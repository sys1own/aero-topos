"""Deterministic, language-agnostic dependency & complexity evaluator.

This replaces regex/keyword scanning with a Tree-sitter **query** engine that
reads declarative S-expression patterns from external ``.scm`` files
(``core/analysis/queries/<language>/<name>.scm``).  Everything here is
deterministic: same inputs -> byte-identical outputs.

Capabilities
------------
* **Polyglot import extraction** -- runs the per-language ``imports`` query to
  isolate import roots, resolves relative module paths against the workspace, and
  inserts the resulting edges into the DAG tracked in ``blueprint.aero``.
* **Cyclomatic complexity** -- runs the ``branches`` query (``[ (if_statement)
  (for_statement) ... ] @branch_node``) over each function subtree and computes
  ``M = C + 1`` where ``C`` is the matched branch count.
* **Automated decomposition** -- when a function's ``M`` exceeds
  ``max_module_complexity`` (or the file crosses ``auto_split_threshold`` lines),
  the function's byte span is lifted into its own generated module and a relative
  import is inserted back into the original file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.parser.universal import load_language, detect_language

_PathLike = Union[str, Path]

_QUERIES_DIR = Path(__file__).resolve().parent / "queries"

# Per-language decomposition templates (extension + import statement form).
_DECOMP_TEMPLATES: Dict[str, Dict[str, str]] = {
    "python": {"ext": ".py", "import": "from {module} import {name}"},
    "rust": {"ext": ".rs", "import": "mod {module};\nuse {module}::{name};"},
}


class InferenceError(Exception):
    """Raised for unrecoverable inference failures (bad language, etc.)."""


# ---------------------------------------------------------------------------
# 1. Tree-sitter query engine (reads external .scm definitions)
# ---------------------------------------------------------------------------
class QueryEngine:
    """Loads and caches Tree-sitter queries from external ``.scm`` files."""

    def __init__(self, queries_dir: _PathLike = _QUERIES_DIR) -> None:
        self.queries_dir = Path(queries_dir)
        self._cache: Dict[tuple, object] = {}

    def query_path(self, language: str, name: str) -> Path:
        return self.queries_dir / language / f"{name}.scm"

    def has_query(self, language: str, name: str) -> bool:
        return self.query_path(language, name).is_file()

    def load(self, language: str, name: str):
        key = (language, name)
        if key in self._cache:
            return self._cache[key]
        path = self.query_path(language, name)
        if not path.is_file():
            raise InferenceError(f"No '{name}' query for language {language!r} ({path})")
        from tree_sitter import Query

        query = Query(load_language(language), path.read_text(encoding="utf-8"))
        self._cache[key] = query
        return query

    def captures(self, language: str, name: str, node) -> Dict[str, list]:
        """Return ``{capture_name: [nodes]}`` for *node*'s subtree."""
        from tree_sitter import QueryCursor

        cursor = QueryCursor(self.load(language, name))
        return cursor.captures(node)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass
class ImportEdge:
    raw: str                       # the import root as written in source
    resolved: Optional[str] = None  # workspace-relative path, if it resolves


@dataclass
class FunctionComplexity:
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    branch_count: int
    complexity: int                # M = branch_count + 1
    line_count: int
    top_level: bool


@dataclass
class FileAnalysis:
    path: str                      # workspace-relative path
    language: str
    line_count: int
    imports: List[ImportEdge] = field(default_factory=list)
    functions: List[FunctionComplexity] = field(default_factory=list)

    @property
    def dependencies(self) -> List[str]:
        return sorted({e.resolved for e in self.imports if e.resolved})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _function_name(func_node, language: str, source: bytes, engine: QueryEngine) -> str:
    # Prefer an explicit @name capture from the functions query.
    caps = engine.captures(language, "functions", func_node)
    for name_node in caps.get("name", []):
        if func_node.start_byte <= name_node.start_byte and name_node.end_byte <= func_node.end_byte:
            return _text(name_node, source)
    # Fallback: first identifier-ish descendant (covers C/C++ declarators).
    queue = list(func_node.children)
    while queue:
        n = queue.pop(0)
        if n.type in ("identifier", "type_identifier", "field_identifier"):
            return _text(n, source)
        queue.extend(n.children)
    return "anonymous"


# ---------------------------------------------------------------------------
# 3. Cyclomatic complexity
# ---------------------------------------------------------------------------
def cyclomatic_complexity(func_node, language: str, engine: QueryEngine) -> int:
    """Count @branch_node matches in *func_node*'s subtree; return M = C + 1."""
    caps = engine.captures(language, "branches", func_node)
    branch_nodes = caps.get("branch_node", [])
    return len(branch_nodes) + 1


# ---------------------------------------------------------------------------
# 2. Import resolution
# ---------------------------------------------------------------------------
def _resolve_python(module_str: str, file_path: Path, workspace: Path) -> Optional[str]:
    if module_str.startswith("."):
        dots = len(module_str) - len(module_str.lstrip("."))
        rest = module_str[dots:]
        base = file_path.parent
        for _ in range(dots - 1):
            base = base.parent
        parts = [p for p in rest.split(".") if p]
    else:
        base = workspace
        parts = [p for p in module_str.split(".") if p]
    if not parts:
        return None
    candidate = base.joinpath(*parts)
    for resolved in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if resolved.is_file():
            try:
                return resolved.resolve().relative_to(workspace).as_posix()
            except ValueError:
                return resolved.as_posix()
    return None


def _resolve_generic(module_str: str, file_path: Path, workspace: Path) -> Optional[str]:
    # Best-effort: treat the string as a path fragment relative to the file/workspace.
    cleaned = module_str.strip().strip('"').strip("<>").strip()
    for base in (file_path.parent, workspace):
        candidate = (base / cleaned)
        if candidate.is_file():
            try:
                return candidate.resolve().relative_to(workspace).as_posix()
            except ValueError:
                return candidate.as_posix()
    return None


_RESOLVERS = {"python": _resolve_python}


def _resolve_import(module_str: str, language: str, file_path: Path, workspace: Path) -> Optional[str]:
    resolver = _RESOLVERS.get(language, _resolve_generic)
    return resolver(module_str, file_path, workspace)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, workspace: _PathLike, queries_dir: _PathLike = _QUERIES_DIR) -> None:
        self.workspace = Path(workspace).resolve()
        self.queries = QueryEngine(queries_dir)

    def _relpath(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.workspace).as_posix()
        except ValueError:
            return path.as_posix()

    def _parse(self, path: Path, language: str):
        from tree_sitter import Parser

        source = path.read_bytes()
        # Build the native parser through the self-healing loader. Any failure
        # (grammar load, ABI drift, parse crash) is normalised into a clean
        # InferenceError so callers like analyze_paths() can skip the file
        # instead of letting a low-level exception cascade up the pipeline.
        try:
            parser = Parser(load_language(language))
            tree = parser.parse(source)
        except Exception as exc:
            raise InferenceError(f"Failed to parse {path} as {language!r}: {exc}") from exc
        if tree is None or tree.root_node is None:
            raise InferenceError(f"Parser returned no tree for {path} ({language!r})")
        return tree, source

    def analyze_file(self, path: _PathLike, language: Optional[str] = None) -> FileAnalysis:
        import re
        from core.analysis.inference import FileAnalysis, FunctionComplexity, ImportEdge

        file_path = Path(path).resolve()
        resolved_language = language or detect_language(file_path)
        if resolved_language is None:
            raise InferenceError(f"Unsupported file extension: {file_path.suffix!r}")

        analysis = FileAnalysis(
            path=self._relpath(file_path),
            language=resolved_language,
            line_count=0,
        )

        try:
            # 1. Attempt standard native Tree-sitter parsing path
            tree, source = self._parse(file_path, resolved_language)
            root = tree.root_node
            analysis.line_count = source.count(b"\n") + (0 if source.endswith(b"\n") or not source else 1)

            if self.queries.has_query(resolved_language, "imports"):
                import_caps = self.queries.captures(resolved_language, "imports", root)
                seen = set()
                for module_node in sorted(import_caps.get("module", []), key=lambda n: n.start_byte):
                    raw = _text(module_node, source)
                    if raw in seen: continue
                    seen.add(raw)
                    resolved = _resolve_import(raw, resolved_language, file_path, self.workspace)
                    analysis.imports.append(ImportEdge(raw=raw, resolved=resolved))

            func_caps = self.queries.captures(resolved_language, "functions", root)
            functions = sorted(func_caps.get("function", []), key=lambda n: n.start_byte)
            for func in functions:
                m = cyclomatic_complexity(func, resolved_language, self.queries)
                name = _function_name(func, resolved_language, source, self.queries)
                analysis.functions.append(
                    FunctionComplexity(
                        name=name,
                        start_byte=func.start_byte,
                        end_byte=func.end_byte,
                        start_line=func.start_point[0] + 1,
                        end_line=func.end_point[0] + 1,
                        branch_count=m - 1,
                        complexity=m,
                        line_count=func.end_point[0] - func.start_point[0] + 1,
                        top_level=func.parent is not None and func.parent == root,
                    )
                )
            return analysis

        except Exception:
            # 2. Resilient Path: Fallback to high-fidelity structural token regex matching
            try:
                source_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                raise InferenceError(f"Failed to read file source: {e}")

            lines = source_text.splitlines()
            analysis.line_count = len(lines)

            if resolved_language == "python":
                func_pattern = re.compile(r"^[ \t]*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
                import_pattern = re.compile(r"^[ \t]*(?:import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)|from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import)")
            else:
                func_pattern = re.compile(r"(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)")
                import_pattern = re.compile(r"^[ \t]*use\s+([a-zA-Z_][a-zA-Z0-9_::\*]*)")

            for line in lines:
                match = import_pattern.match(line)
                if match:
                    raw_import = next((g for g in match.groups() if g), "unknown")
                    analysis.imports.append(ImportEdge(raw=raw_import, resolved=None))

            for idx, line in enumerate(lines):
                match = func_pattern.match(line)
                if match:
                    func_name = match.group(1)
                    complexity_score = 1
                    scope_indent = len(line) - len(line.lstrip())
                    func_line_count = 1
                    
                    for sub_line in lines[idx+1:]:
                        if sub_line.strip():
                            sub_indent = len(sub_line) - len(sub_line.lstrip())
                            if resolved_language == "python" and sub_indent <= scope_indent:
                                break
                            if "}" in sub_line and resolved_language != "python":
                                break
                        func_line_count += 1
                        
                        tokens = sub_line.split()
                        for branch_token in ["if", "elif", "for", "while", "and", "or", "match"]:
                            if branch_token in tokens:
                                complexity_score += 1

                    analysis.functions.append(
                        FunctionComplexity(
                            name=func_name,
                            start_byte=0,
                            end_byte=0,
                            start_line=idx + 1,
                            end_line=idx + func_line_count,
                            branch_count=complexity_score - 1,
                            complexity=complexity_score,
                            line_count=func_line_count,
                            top_level=True
                        )
                    )
            return analysis

    def analyze_paths(self, paths) -> List[FileAnalysis]:
        results = []
        for p in paths:
            try:
                results.append(self.analyze_file(p))
            except InferenceError:
                continue
        return sorted(results, key=lambda a: a.path)

    def iter_source_files(self) -> List[Path]:
        files = []
        for candidate in sorted(self.workspace.rglob("*")):
            if candidate.is_file() and detect_language(candidate) and self.queries.has_query(
                detect_language(candidate), "functions"
            ):
                files.append(candidate)
        return files

    # -- DAG ------------------------------------------------------------------
    def build_dag(self, analyses: List[FileAnalysis]) -> Dict[str, List[str]]:
        """Module -> sorted unique resolved dependencies (deterministic)."""
        dag: Dict[str, List[str]] = {}
        for analysis in sorted(analyses, key=lambda a: a.path):
            dag[analysis.path] = analysis.dependencies
        return dag

    def write_dag_to_blueprint(self, blueprint_path: _PathLike, dag: Dict[str, List[str]]) -> None:
        """Insert/replace the ``[dag]`` table in ``blueprint.aero`` deterministically."""
        bp_path = Path(blueprint_path)
        text = bp_path.read_text(encoding="utf-8") if bp_path.is_file() else ""
        text = _strip_table(text, "dag")

        # An edge that points at another node in this graph is rewritten to that
        # node's sanitized key so the table stays internally consistent; edges to
        # external/unsanitized references are preserved verbatim.
        node_keys = set(dag)
        lines = ["[dag]"]
        for module in sorted(dag):
            deps = dag[module]
            array = "[" + ", ".join(
                _toml_str(_toml_key(d) if d in node_keys else d) for d in deps
            ) + "]"
            lines.append('"' + _toml_key(module) + '" = ' + array)
        block = "\n".join(lines) + "\n"

        if text and not text.endswith("\n"):
            text += "\n"
        if text and not text.endswith("\n\n"):
            text += "\n"
        text += block
        bp_path.parent.mkdir(parents=True, exist_ok=True)
        bp_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Automated decomposition
# ---------------------------------------------------------------------------
@dataclass
class DecompositionAction:
    function: str
    new_module: str               # workspace-relative path of the generated module
    import_line: str
    reason: str                   # "complexity" | "file_size"


@dataclass
class DecompositionPlan:
    path: str
    applied: bool
    actions: List[DecompositionAction] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def decompose_file(
    engine: InferenceEngine,
    path: _PathLike,
    max_module_complexity: int,
    auto_split_threshold: int,
    *,
    apply: bool = False,
) -> DecompositionPlan:
    """Split over-complex / oversized functions into their own modules.

    Deterministic: candidate functions are processed in a fixed order and the
    generated module names derive only from the file stem + function name.
    """
    file_path = Path(path).resolve()
    analysis = engine.analyze_file(file_path)
    language = analysis.language
    template = _DECOMP_TEMPLATES.get(language)

    plan = DecompositionPlan(path=engine._relpath(file_path), applied=False)
    if template is None:
        plan.skipped.append(f"decomposition not supported for {language!r}")
        return plan

    file_too_big = analysis.line_count > auto_split_threshold
    # Candidates: top-level functions that are over-complex, or (if the file is
    # oversized) all top-level functions. Deterministic order by start byte.
    candidates = []
    for fn in analysis.functions:
        if not fn.top_level or fn.name == "anonymous":
            continue
        if fn.complexity > max_module_complexity:
            candidates.append((fn, "complexity"))
        elif file_too_big:
            candidates.append((fn, "file_size"))

    if not candidates:
        return plan

    source = file_path.read_bytes()
    stem = file_path.stem
    ext = template["ext"]
    used_names = set()

    # Build the actions (sorted by start byte for stable output).
    candidates.sort(key=lambda c: c[0].start_byte)
    for fn, reason in candidates:
        module_name = f"{stem}_{fn.name}"
        if module_name in used_names:
            continue
        used_names.add(module_name)
        import_line = template["import"].format(module=module_name, name=fn.name)
        plan.actions.append(
            DecompositionAction(
                function=fn.name,
                new_module=engine._relpath(file_path.parent / (module_name + ext)),
                import_line=import_line,
                reason=reason,
            )
        )

    if not apply:
        return plan

    # Apply: write new modules, then excise spans (reverse byte order), then
    # prepend imports. Reverse order keeps byte offsets valid during excision.
    name_to_fn = {f"{stem}_{fn.name}": fn for fn, _ in candidates}
    for action in plan.actions:
        fn = name_to_fn[Path(action.new_module).stem]
        body = source[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
        module_file = file_path.parent / (Path(action.new_module).stem + ext)
        module_file.write_text(body.rstrip("\n") + "\n", encoding="utf-8")

    new_source = source
    for fn, _ in sorted(candidates, key=lambda c: c[0].start_byte, reverse=True):
        new_source = new_source[:fn.start_byte] + new_source[fn.end_byte:]

    text = new_source.decode("utf-8", "replace")
    # Collapse the blank gap left by excision, then prepend imports.
    header = "\n".join(a.import_line for a in plan.actions) + "\n"
    text = header + text.lstrip("\n")
    file_path.write_text(text, encoding="utf-8")
    plan.applied = True
    return plan


# ---------------------------------------------------------------------------
# TOML helpers (deterministic, comment-preserving for untouched sections)
# ---------------------------------------------------------------------------
def _toml_str(value: str) -> str:
    """Return a TOML-compatible double-quoted string for *value*."""
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\x00", "\\x00")
        .replace("\x01", "\\x01")
        .replace("\x02", "\\x02")
        .replace("\x03", "\\x03")
        .replace("\x04", "\\x04")
        .replace("\x05", "\\x05")
        .replace("\x06", "\\x06")
        .replace("\x07", "\\x07")
        .replace("\x08", "\\b")
        .replace("\x0b", "\\x0b")
        .replace("\x0c", "\\x0c")
        .replace("\x0e", "\\x0e")
        .replace("\x0f", "\\x0f")
        .replace("\x10", "\\x10")
        .replace("\x11", "\\x11")
        .replace("\x12", "\\x12")
        .replace("\x13", "\\x13")
        .replace("\x14", "\\x14")
        .replace("\x15", "\\x15")
        .replace("\x16", "\\x16")
        .replace("\x17", "\\x17")
        .replace("\x18", "\\x18")
        .replace("\x19", "\\x19")
        .replace("\x1a", "\\x1a")
        .replace("\x1b", "\\x1b")
        .replace("\x1c", "\\x1c")
        .replace("\x1d", "\\x1d")
        .replace("\x1e", "\\x1e")
        .replace("\x1f", "\\x1f")
        .replace("\x7f", "\\x7f")
        + '"'
    )


def _toml_key(value: str) -> str:
    """Return a TOML-compatible key for *value*.

    Valid TOML bare keys may contain only ``A-Z``, ``a-z``, ``0-9``, ``_`` and
    ``-``. Module names that come from file paths or decomposed identifiers can
    carry path separators, dots, or other characters, so we sanitize them into a
    safe bare key. Underscore runs are preserved (they are common in decomposed
    module names such as ``main__build_dsl_targets``). If sanitization empties
    the string, we fall back to a generic bare key so the document never has an
    invalid key.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_")
    if not safe:
        return "module"
    return safe


def _strip_table(text: str, name: str) -> str:
    """Remove a top-level ``[name]`` table (and its keys) from *text*."""
    out: List[str] = []
    skip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            skip = (
                stripped == f"[{name}]"
                or stripped.startswith(f"[{name}.")
                or stripped.startswith(f"[[{name}")
            )
        if not skip:
            out.append(line)
    result = "\n".join(out).rstrip("\n")
    return result + "\n" if result else ""
