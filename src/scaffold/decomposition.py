# -*- coding: utf-8 -*-
"""
AST-driven *modular decomposition* of a monolithic Python script.

Where :mod:`src.scaffold.python_repo_generator` copies a single entry script into
a turn-key project verbatim, this module performs the inverse of monolith growth:
it reads one large script (e.g. ``main.py``) and breaks it apart into a
multi-file, highly decoupled package layout, driven entirely by a
``module_mapping`` declared in the blueprint's ``[scaffold]`` section.

The mapping keys are *target filenames* (``"parser"`` / ``"parser.py"``) and the
values are lists of class / function names that should live in that file::

    module_mapping = {
        "parser":   ["SchemaValidator", "parse_blueprint"],
        "cli":      ["main", "create_parser"],
        "shielder": ["RustSemanticShield"],
    }

For each target the decomposer:

* extracts the matching top-level ``class`` / ``def`` / ``async def`` nodes
  (decorators included) using the AST, preserving original source order;
* duplicates every global module import to the top of each generated file so the
  decoupled modules do not break on missing dependencies;
* writes an (empty) ``__init__.py`` to turn the directory into an importable
  package;
* rewrites the root ``main.py`` into a thin orchestrator that pulls the moved
  symbols back in via package-relative imports (``from .parser import
  SchemaValidator``).

Error handling guards against missing AST nodes (a name mapped but not defined),
cross-module import collisions (one symbol mapped to two files), unparseable
sources and empty mappings.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.decomposition.splitter import hoist_future_flags
from src.scaffold.import_pruner import prune_dead_imports, render_imports

Logger = Callable[[str], None]

# Node types that represent an extractable, top-level definition.
_DEF_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _is_future_import(line: str) -> bool:
    """True if a rendered import line is a ``from __future__`` statement.

    ``from __future__`` imports are compiler flags that must occupy the very
    first lines of a module (only comments, blank lines, the module docstring
    and other future statements may precede them), so they are tracked and
    hoisted separately from the general import block.
    """
    return line.lstrip().startswith("from __future__")


_TYPE_HINT_FALLBACK_TYPING = "from typing import Optional, Union, Any, List, Dict, Callable, Iterator"
_TYPE_HINT_FALLBACK_PATHLIB = "from pathlib import Path"
_TYPE_HINT_RE: re.Pattern[str] = re.compile(
    r"(?:\b(?:Optional|Union|List|Dict|Callable|Iterator|Any)\s*\[)|(?::\s*Path\b)"
)


def _has_typing_coverage(source: str) -> bool:
    """True when the source imports ``typing`` or ``pathlib`` directly."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("typing", "pathlib"):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("typing", "pathlib"):
                return True
    return False


def _needs_type_hint_fallback(source: str) -> bool:
    """True when the source uses common type-hint patterns without coverage."""
    if not _TYPE_HINT_RE.search(source):
        return False
    return not _has_typing_coverage(source)


def _hoist_future_imports(source: str) -> str:
    """Return *source* with every ``from __future__`` line hoisted to the top.

    Delegates to the centralized :func:`hoist_future_flags` helper while
    keeping any leading encoding cookie on line 1 or 2 (Python requires it
    there). This makes the operation idempotent: re-running the scaffolder on
    its own output never duplicates or pushes the future imports down.
    """
    lines = source.splitlines(keepends=True)
    if not lines:
        return source

    encoding_lines: List[str] = []
    rest_lines: List[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (
            i < 2
            and stripped.startswith("#")
            and "coding" in stripped
        ):
            encoding_lines.append(line)
        else:
            rest_lines.append(line)

    hoisted = hoist_future_flags("".join(rest_lines))
    if encoding_lines:
        return "".join(encoding_lines) + hoisted
    return hoisted


def _inject_type_hint_fallbacks(source: str) -> str:
    """Return *source* with typing/pathlib fallback imports if needed.

    Fallbacks are placed directly below any ``__future__`` imports so the
    generated module compiles even when the parent source's typing imports were
    not carried over.
    """
    if not _needs_type_hint_fallback(source):
        return source

    typing_present = _TYPE_HINT_FALLBACK_TYPING in source
    pathlib_present = _TYPE_HINT_FALLBACK_PATHLIB in source
    if typing_present and pathlib_present:
        return source

    lines = source.splitlines(keepends=True)
    insert_pos = 0
    seen_future = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from __future__"):
            insert_pos = i + 1
            seen_future = True
        elif seen_future and stripped == "":
            insert_pos = i + 1
        elif seen_future:
            break

    if not seen_future:
        # No __future__ header: place fallback after the leading docstring/comments.
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped == ""
                or stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                insert_pos = i + 1
            else:
                break

    additions: List[str] = []
    if not typing_present:
        additions.append(_TYPE_HINT_FALLBACK_TYPING + "\n")
    if not pathlib_present:
        additions.append(_TYPE_HINT_FALLBACK_PATHLIB + "\n")
    if not additions:
        return source

    additions.append("\n")
    for line in reversed(additions):
        lines.insert(insert_pos, line)

    return "".join(lines)


class DecompositionError(ValueError):
    """Base error for any failure during modular decomposition."""


class MissingASTNodeError(DecompositionError):
    """A name in ``module_mapping`` has no matching top-level definition."""


class ImportCollisionError(DecompositionError):
    """A single symbol is mapped to more than one target module."""


@dataclass
class DecomposedModule:
    """A single generated module file and the symbols it now owns."""

    filename: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    source: str = ""
    cross_imports: List[str] = field(default_factory=list)

    @property
    def module_name(self) -> str:
        return self.filename[:-3] if self.filename.endswith(".py") else self.filename

    @property
    def symbols(self) -> List[str]:
        return list(self.classes) + list(self.functions)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "module": self.module_name,
            "classes": list(self.classes),
            "functions": list(self.functions),
            "cross_imports": list(self.cross_imports),
        }


@dataclass
class DecompositionResult:
    """The full outcome of a modular decomposition run."""

    root: Path
    modules: List[DecomposedModule] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    orchestrator: str = "main.py"
    package_init: str = "__init__.py"

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "files": list(self.files),
            "modules": [m.to_dict() for m in self.modules],
            "imports": list(self.imports),
            "orchestrator": self.orchestrator,
            "package_init": self.package_init,
            "mode": "modular_package",
        }


@dataclass
class MergedSource:
    """The in-memory result of merging several source files' ASTs."""

    source: str
    files: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)


def _strip_top_level_imports(text: str, tree: ast.Module) -> str:
    """Return ``text`` with its top-level import statements removed."""
    drop: set = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno
            end = getattr(node, "end_lineno", None) or node.lineno
            drop.update(range(start, end + 1))
    lines = text.splitlines(keepends=True)
    return "".join(line for idx, line in enumerate(lines, start=1) if idx not in drop)


def merge_source_asts(
    sources: List[Tuple[str, str]],
    logger: Optional[Logger] = None,
) -> MergedSource:
    """Merge multiple Python source files into a single structural schema.

    Hoists and de-duplicates every top-level import (``from __future__`` first),
    then concatenates each file's remaining top-level body under a provenance
    banner.  The merged source has consistent line numbering so downstream
    ``ast.get_source_segment`` extraction keeps working.
    """
    future_lines: List[str] = []
    import_lines: List[str] = []
    seen_imports: set = set()
    body_chunks: List[str] = []
    merged_files: List[str] = []
    definitions: List[str] = []

    for label, text in sources:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise DecompositionError(
                f"cannot parse source '{label}' for AST merge: "
                f"{exc.msg} (line {exc.lineno})"
            ) from exc

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = (ast.get_source_segment(text, node) or "").strip()
                if not segment or segment in seen_imports:
                    continue
                seen_imports.add(segment)
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    future_lines.append(segment)
                else:
                    import_lines.append(segment)
            elif isinstance(node, _DEF_NODES):
                definitions.append(node.name)

        body = _strip_top_level_imports(text, tree).strip("\n")
        chunk = f"# --- merged from {label} ---\n{body}\n" if body else ""
        if chunk:
            body_chunks.append(chunk)
        merged_files.append(label)
        if logger is not None:
            n_defs = sum(1 for n in tree.body if isinstance(n, _DEF_NODES))
            logger(f"[Ingest     ] Merged AST from {label} ({n_defs} top-level def(s))")

    header = future_lines + import_lines
    parts: List[str] = []
    if header:
        parts.append("\n".join(header) + "\n")
    parts.extend(body_chunks)
    merged = "\n".join(parts) if parts else ""
    return MergedSource(
        source=_hoist_future_imports(merged),
        files=merged_files,
        definitions=definitions,
    )


def resolve_cross_imports(
    body_source: str,
    own_module: str,
    symbol_to_target: Dict[str, str],
    *,
    package_relative: bool = True,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Compute relative imports a module needs for symbols moved elsewhere.

    Walks ``body_source`` for referenced names; any name that now lives in a
    *different* generated module is grouped into a clean ``from .module import …``
    line so the decoupled file does not raise ``NameError`` at runtime.

    Returns ``(import_lines, [(symbol, module_name), …])``.
    """
    try:
        tree = ast.parse(body_source)
    except SyntaxError:
        return [], []

    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    prefix = "." if package_relative else ""

    needed: Dict[str, List[str]] = {}
    pairs: List[Tuple[str, str]] = []
    for symbol in sorted(used):
        target = symbol_to_target.get(symbol)
        if target is None:
            continue
        module = target[:-3] if target.endswith(".py") else target
        own = own_module[:-3] if own_module.endswith(".py") else own_module
        if module == own:
            continue  # symbol is defined in this very module
        needed.setdefault(module, []).append(symbol)
        pairs.append((symbol, module))

    import_lines = [
        f"from {prefix}{module} import {', '.join(symbols)}"
        for module, symbols in needed.items()
    ]
    return import_lines, pairs


class ModularDecomposer:
    """Break a monolithic Python script into a decoupled package layout."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        verbose: bool = True,
        *,
        package_relative: bool = True,
        prune_imports: bool = False,
    ) -> None:
        self._logger = logger
        self.verbose = verbose
        self.package_relative = package_relative
        self.prune_imports = prune_imports

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose and self._logger is not None:
            self._logger(message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        source: str,
        module_mapping: Dict[str, List[str]],
        *,
        source_filename: str = "main.py",
        dest_dir: Path,
    ) -> DecompositionResult:
        """Decompose ``source`` into ``dest_dir`` per ``module_mapping``.

        Writes one file per mapped target, an ``__init__.py`` package marker, and
        a rewritten orchestrator named after ``source_filename``.  Raises a
        :class:`DecompositionError` subclass on any structural problem.
        """
        dest_dir = Path(dest_dir)
        orchestrator_name = Path(source_filename).name or "main.py"

        normalized = self._normalize_mapping(module_mapping)
        if not normalized:
            raise DecompositionError(
                "module_mapping is empty — nothing to decompose; declare at least "
                "one target file -> [symbols] entry in the [scaffold] block"
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise DecompositionError(
                f"cannot parse source '{orchestrator_name}' for decomposition: "
                f"{exc.msg} (line {exc.lineno})"
            ) from exc

        source_lines = source.splitlines(keepends=True)
        symbol_to_target = self._build_symbol_index(normalized)

        # Capture global imports (duplicated into every generated module) and the
        # universe of top-level definitions available for extraction.
        future_imports, top_imports = self._collect_imports(tree, source)
        defs_by_name = self._collect_definitions(tree)

        self._assert_all_present(symbol_to_target, defs_by_name, orchestrator_name)

        # Group the extracted nodes by their target, preserving source order, and
        # record the line ranges so they can be removed from the orchestrator.
        target_nodes: Dict[str, List[ast.AST]] = {t: [] for t in normalized}
        extracted_ranges: List[Tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, _DEF_NODES) and node.name in symbol_to_target:
                target = symbol_to_target[node.name]
                target_nodes[target].append(node)
                extracted_ranges.append(self._node_line_range(node))

        # De-duplicate while preserving order (matters for merged multi-file
        # sources where the same import can appear in several inputs).
        header_imports: List[str] = []
        _seen_imports: set = set()
        for line in list(future_imports) + list(top_imports):
            key = line.strip()
            if key and key not in _seen_imports:
                _seen_imports.add(key)
                header_imports.append(line)

        dest_dir.mkdir(parents=True, exist_ok=True)
        modules: List[DecomposedModule] = []
        written: List[str] = []

        for target, names in normalized.items():
            nodes = target_nodes[target]
            module = self._render_module(
                source=source,
                source_lines=source_lines,
                header_imports=header_imports,
                nodes=nodes,
                target=target,
                dest_dir=dest_dir,
                symbol_to_target=symbol_to_target,
            )
            (dest_dir / module.filename).write_text(module.source, encoding="utf-8")
            modules.append(module)
            written.append(module.filename)

        # Package boundary — an empty, idiomatic __init__.py.
        (dest_dir / "__init__.py").write_text("", encoding="utf-8")
        written.append("__init__.py")
        self._log("[Scaffold   ] Initialized package boundary __init__.py")

        # Root orchestrator: original script minus the moved defs, plus relative
        # imports that pull them back from the generated modules.
        orchestrator_source = self._build_orchestrator(
            tree=tree,
            source_lines=source_lines,
            extracted_ranges=extracted_ranges,
            normalized=normalized,
            target_nodes=target_nodes,
        )
        (dest_dir / orchestrator_name).write_text(orchestrator_source, encoding="utf-8")
        written.append(orchestrator_name)
        self._log(
            f"[Scaffold   ] Rewrote orchestrator entrypoint {dest_dir}/{orchestrator_name}"
        )

        return DecompositionResult(
            root=dest_dir,
            modules=modules,
            files=written,
            imports=list(header_imports),
            orchestrator=orchestrator_name,
            package_init="__init__.py",
        )

    # ------------------------------------------------------------------
    # Mapping / index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_mapping(module_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalise to ``{filename.py: [unique, ordered, symbols]}``."""
        if not isinstance(module_mapping, dict):
            raise DecompositionError("module_mapping must be a dict of filename -> [symbols]")
        normalized: Dict[str, List[str]] = {}
        for raw_key, raw_value in module_mapping.items():
            key = str(raw_key).strip()
            if not key:
                continue
            filename = key if key.endswith(".py") else f"{key}.py"
            if filename in ("__init__.py",):
                raise DecompositionError(
                    "module_mapping may not target '__init__.py' — it is generated "
                    "automatically as the package boundary"
                )
            if isinstance(raw_value, str):
                items = [v.strip() for v in raw_value.split(",")]
            elif isinstance(raw_value, (list, tuple)):
                items = [str(v).strip() for v in raw_value]
            else:
                raise DecompositionError(
                    f"module_mapping['{key}'] must be a list of symbol names"
                )
            # De-duplicate while preserving order; drop empties.
            seen: set = set()
            symbols: List[str] = []
            for item in items:
                if item and item not in seen:
                    seen.add(item)
                    symbols.append(item)
            if not symbols:
                raise DecompositionError(
                    f"module_mapping['{key}'] lists no symbols to extract"
                )
            normalized[filename] = symbols
        return normalized

    @staticmethod
    def _build_symbol_index(normalized: Dict[str, List[str]]) -> Dict[str, str]:
        """Map ``symbol -> target file``, rejecting cross-module collisions."""
        index: Dict[str, str] = {}
        for target, symbols in normalized.items():
            for symbol in symbols:
                existing = index.get(symbol)
                if existing is not None and existing != target:
                    raise ImportCollisionError(
                        f"symbol '{symbol}' is mapped to multiple modules "
                        f"('{existing}' and '{target}'); each symbol may live in "
                        "exactly one generated module"
                    )
                index[symbol] = target
        return index

    @staticmethod
    def _assert_all_present(
        symbol_to_target: Dict[str, str],
        defs_by_name: Dict[str, ast.AST],
        source_filename: str,
    ) -> None:
        missing = sorted(name for name in symbol_to_target if name not in defs_by_name)
        if missing:
            raise MissingASTNodeError(
                "module_mapping references names with no matching top-level class "
                f"or function definition in '{source_filename}': "
                f"{', '.join(missing)}"
            )

    # ------------------------------------------------------------------
    # AST extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_imports(tree: ast.Module, source: str) -> Tuple[List[str], List[str]]:
        """Return ``(__future__ imports, other top-level imports)`` as source text."""
        future_imports: List[str] = []
        top_imports: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                segment = ast.get_source_segment(source, node)
                if segment:
                    future_imports.append(segment)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = ast.get_source_segment(source, node)
                if segment:
                    top_imports.append(segment)
        return future_imports, top_imports

    @staticmethod
    def _collect_definitions(tree: ast.Module) -> Dict[str, ast.AST]:
        defs: Dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, _DEF_NODES):
                defs[node.name] = node
        return defs

    @staticmethod
    def _node_line_range(node: ast.AST) -> Tuple[int, int]:
        """1-indexed inclusive ``(start, end)`` lines, decorators included."""
        start = node.lineno
        for decorator in getattr(node, "decorator_list", []) or []:
            start = min(start, decorator.lineno)
        end = getattr(node, "end_lineno", None) or node.lineno
        return start, end

    def _segment_for(self, source_lines: List[str], node: ast.AST) -> str:
        start, end = self._node_line_range(node)
        return "".join(source_lines[start - 1:end]).rstrip("\n")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_module(
        self,
        *,
        source: str,
        source_lines: List[str],
        header_imports: List[str],
        nodes: List[ast.AST],
        target: str,
        dest_dir: Path,
        symbol_to_target: Dict[str, str],
    ) -> DecomposedModule:
        classes: List[str] = []
        functions: List[str] = []
        body_segments: List[str] = []

        for node in nodes:
            segment = self._segment_for(source_lines, node)
            body_segments.append(segment)
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                kind = "class"
            else:
                functions.append(node.name)
                kind = "function"
            self._log(
                f"[Decomposing] Extracted {kind} '{node.name}' -> {dest_dir}/{target}"
            )

        body_text = "\n\n\n".join(body_segments)

        # Intra-module interlinking: pull in symbols that now live in sibling
        # generated modules so the decoupled file does not raise NameError.
        cross_lines, cross_pairs = resolve_cross_imports(
            body_text, target, symbol_to_target, package_relative=self.package_relative
        )
        for symbol, module in cross_pairs:
            self._log(
                f"[Interlink  ] {target}: injected 'from .{module} import {symbol}'"
            )

        all_imports = list(header_imports) + list(cross_lines)
        # Strict __future__ hoisting: separate compiler-flag imports from the
        # general import buffer.  They must land on the absolute first lines of
        # the file (before the banner/docstring) and must never be pruned, since
        # pruning treats them as "unused" and the compiler rejects a mislocated
        # future statement with a hard SyntaxError.
        future_lines: List[str] = []
        other_imports: List[str] = []
        _seen_future: set = set()
        for line in all_imports:
            if _is_future_import(line):
                if line.strip() not in _seen_future:
                    _seen_future.add(line.strip())
                    future_lines.append(line)
            else:
                other_imports.append(line)
        if self.prune_imports and other_imports:
            other_imports = self._prune_module_imports(
                import_lines=other_imports,
                body_text=body_text,
                target=target,
            )

        banner = (
            f'"""Module \'{target}\' — generated by Aero Universal modular '
            'decomposition."""\n'
        )
        parts: List[str] = []
        if future_lines:
            parts.append("\n".join(future_lines) + "\n")
        parts.append(banner)
        if other_imports:
            parts.append("\n".join(other_imports) + "\n")
        if body_segments:
            parts.append(body_text + "\n")

        return DecomposedModule(
            filename=target,
            classes=classes,
            functions=functions,
            source=_inject_type_hint_fallbacks(
                _hoist_future_imports("\n".join(parts))
            ),
            cross_imports=list(cross_lines),
        )

    def _prune_module_imports(
        self,
        *,
        import_lines: List[str],
        body_text: str,
        target: str,
    ) -> List[str]:
        """Drop imports unused by this module's body via static AST analysis."""
        probe = "\n".join(import_lines) + "\n\n\n" + body_text + "\n"
        try:
            module_ast = ast.parse(probe)
        except SyntaxError:
            # Never let an analysis hiccup corrupt a valid module — keep imports.
            self._log(f"[Optimize   ] {target}: skipped pruning (probe parse failed)")
            return import_lines

        outcome = prune_dead_imports(module_ast)
        if outcome.skipped_dynamic:
            self._log(
                f"[Optimize   ] {target}: skipped pruning (dynamic import lookup detected)"
            )
            return import_lines
        if not outcome.changed:
            return import_lines

        new_lines = render_imports(outcome.kept_imports)
        before = len("\n".join(import_lines))
        after = len("\n".join(new_lines))
        saved = max(0, before - after)
        for name in outcome.pruned:
            self._log(f"[Optimize   ] Pruned unused '{name}' import from {target}")
        self._log(
            f"[Optimize   ] {target}: pruned {len(outcome.pruned)} import(s), "
            f"saved {saved} byte(s)"
        )
        return new_lines

    def _render_orchestrator_imports(
        self,
        normalized: Dict[str, List[str]],
        target_nodes: Dict[str, List[ast.AST]],
    ) -> str:
        prefix = "." if self.package_relative else ""
        lines = [
            "",
            "# --- Aero Universal modular decomposition: re-exported package modules ---",
        ]
        for target, names in normalized.items():
            module = target[:-3] if target.endswith(".py") else target
            extracted = {n.name for n in target_nodes[target]}
            ordered = [name for name in names if name in extracted]
            if not ordered:
                continue
            lines.append(f"from {prefix}{module} import {', '.join(ordered)}")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _build_orchestrator(
        self,
        *,
        tree: ast.Module,
        source_lines: List[str],
        extracted_ranges: List[Tuple[int, int]],
        normalized: Dict[str, List[str]],
        target_nodes: Dict[str, List[ast.AST]],
    ) -> str:
        drop: set = set()
        for start, end in extracted_ranges:
            drop.update(range(start, end + 1))

        anchor = self._import_anchor_line(tree, len(source_lines))
        import_block = self._render_orchestrator_imports(normalized, target_nodes)

        out: List[str] = []
        inserted = False
        for index, line in enumerate(source_lines, start=1):
            if index == anchor and not inserted:
                out.append(import_block)
                inserted = True
            if index in drop:
                continue
            out.append(line)
        if not inserted:
            out.append(import_block)
        return _inject_type_hint_fallbacks(
            _hoist_future_imports("".join(out))
        )

    @staticmethod
    def _import_anchor_line(tree: ast.Module, total_lines: int) -> int:
        """First top-level line that is neither the docstring nor an import.

        The generated ``from .module import ...`` block is inserted here so it
        sits with the orchestrator's existing imports while keeping any module
        docstring and ``from __future__`` import first.
        """
        body = tree.body
        i = 0
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            i = 1
        while i < len(body) and isinstance(body[i], (ast.Import, ast.ImportFrom)):
            i += 1
        if i < len(body):
            node = body[i]
            start = node.lineno
            for decorator in getattr(node, "decorator_list", []) or []:
                start = min(start, decorator.lineno)
            return start
        return total_lines + 1
