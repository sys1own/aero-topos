"""Deterministic, non-probabilistic self-healing compilation wrapper.

This module runs a module's build, intercepts *structured* diagnostics, and
applies strictly rule-based Tree-sitter rewrites to repair a bounded set of
well-understood errors.  Nothing here is probabilistic: every fix is a
template-driven structural edit selected by error code / category, so the same
(source, error) pair always yields the same healed source.

Repair model (the H function)::

    H(CST_orig, e) -> CST_healed
        CST_healed = Replace(P(e, CST_orig), R(e))

where ``e`` is an isolated error span, ``P`` is the Tree-sitter pattern matcher
that locates the construct to edit, and ``R`` is the strict template fix.

Execution is bounded to 3 successive build attempts per module.  If errors
persist after the 3rd attempt the module's source is rolled back to its original
clean state, the failure is flagged in ``blueprint.aero``, and we exit safely so
a partial/corrupt edit is never left behind.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from core.parser.universal import _load_language, detect_language, extract_symbols, parse_source

_PathLike = Union[str, Path]

MAX_BUILD_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Unified diagnostic model (compiler JSON + LSP proxy)
# ---------------------------------------------------------------------------
@dataclass
class Diagnostic:
    message: str
    file: str
    code: Optional[str] = None
    severity: str = "error"
    line: int = 1
    column: int = 1
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    source: str = "compiler"

    @classmethod
    def from_lsp(cls, lsp_diag, source_text: str = "") -> "Diagnostic":
        from core.toolchain.lsp_proxy import Diagnostic as LspDiagnostic  # local import

        assert isinstance(lsp_diag, LspDiagnostic)
        return cls(
            message=lsp_diag.message,
            file=lsp_diag.uri,
            code=str(lsp_diag.code) if lsp_diag.code is not None else None,
            severity={1: "error", 2: "warning"}.get(lsp_diag.severity or 1, "info"),
            line=lsp_diag.start.line + 1,
            column=lsp_diag.start.character + 1,
            start_byte=lsp_diag.start.byte,
            end_byte=lsp_diag.end.byte,
            source="lsp",
        )


# ---------------------------------------------------------------------------
# 1. Structured diagnostic interception (compiler JSON parsers)
# ---------------------------------------------------------------------------
def parse_rustc_json(stream: str, only_errors: bool = True) -> List[Diagnostic]:
    """Parse ``rustc --error-format=json`` (JSONL) into diagnostics."""
    diags: List[Diagnostic] = []
    for line in stream.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("$message_type") not in (None, "diagnostic"):
            continue
        if "message" not in obj or "spans" not in obj:
            continue
        level = obj.get("level", "error")
        if only_errors and level != "error":
            continue
        code = (obj.get("code") or {}).get("code") if obj.get("code") else None
        spans = obj.get("spans") or []
        primary = next((s for s in spans if s.get("is_primary")), spans[0] if spans else None)
        if primary is None:
            continue
        diags.append(Diagnostic(
            message=obj.get("message", ""),
            file=primary.get("file_name", ""),
            code=code,
            severity=level,
            line=primary.get("line_start", 1),
            column=primary.get("column_start", 1),
            start_byte=primary.get("byte_start"),
            end_byte=primary.get("byte_end"),
            source="rustc",
        ))
    return diags


def parse_gcc_json(stream: str, only_errors: bool = True) -> List[Diagnostic]:
    """Parse ``gcc/clang -fdiagnostics-format=json`` output into diagnostics."""
    text = stream.strip()
    if not text:
        return []
    # gcc emits a JSON array (sometimes several, one per TU); be lenient.
    payloads = []
    for chunk in re.findall(r"\[.*?\]", text, re.DOTALL):
        try:
            payloads.extend(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    diags: List[Diagnostic] = []
    for item in payloads:
        kind = item.get("kind", "error")
        if only_errors and kind != "error":
            continue
        locations = item.get("locations") or []
        caret = (locations[0].get("caret") if locations else {}) or {}
        finish = (locations[0].get("finish") if locations else {}) or {}
        diags.append(Diagnostic(
            message=item.get("message", ""),
            file=caret.get("file", ""),
            code=item.get("option"),
            severity=kind,
            line=caret.get("line", 1),
            column=caret.get("column", 1),
            start_byte=caret.get("byte-column"),
            end_byte=finish.get("byte-column"),
            source="gcc",
        ))
    return diags


# ---------------------------------------------------------------------------
# 2. Error categorisation (the matrix keys)
# ---------------------------------------------------------------------------
class Category(str, Enum):
    MISSING_IMPORT = "missing_import"
    IMMUTABLE_ASSIGNMENT = "immutable_assignment"
    MISSING_SEMICOLON = "missing_semicolon"
    MISSING_CRATE = "missing_crate"
    DUPLICATE_DEFINITION = "duplicate_definition"
    UNRESOLVED_PARENT_IMPORT = "unresolved_parent_import"
    MISMATCHED_SLICE_TYPE = "mismatched_slice_type"
    PYO3_GLOB_IMPORT = "pyo3_glob_import"


def categorize(diag: Diagnostic) -> Optional[Category]:
    code = (diag.code or "").upper()
    msg = diag.message.lower()

    # --- Rust-specific advanced rules (checked first for specificity) --------

    # PyO3 declarative #[pymodule] cannot re-export glob (`use super::*;`) imports.
    if "pymodule" in msg and "glob" in msg:
        return Category.PYO3_GLOB_IMPORT

    # E0433 / E0405: missing crate or module (Cargo.toml dependency injection)
    if code in {"E0433", "E0405"}:
        return Category.MISSING_CRATE

    # E0428: duplicate symbol definitions
    if code == "E0428" or "is defined multiple times" in msg:
        return Category.DUPLICATE_DEFINITION

    # E0432 with super:: path: private-item visibility promotion
    if (code == "E0432" or "unresolved import" in msg) and "super::" in msg:
        return Category.UNRESOLVED_PARENT_IMPORT

    # E0308 with slice/array mismatch
    if code == "E0308" and ("&[" in diag.message and "[" in diag.message):
        # Heuristic: the message contains both a slice type and an array type.
        if re.search(r"expected `&\[", diag.message) and re.search(r"found `&\[\[", diag.message):
            return Category.MISMATCHED_SLICE_TYPE

    # --- Original broad categories -------------------------------------------

    if code == "E0432" or "unresolved import" in msg \
            or "could not be resolved" in msg or "reportmissingimports" in code.lower() \
            or ("is not defined" in msg) or "reportundefinedvariable" in (diag.code or "").lower():
        return Category.MISSING_IMPORT
    if code in {"E0384", "E0594", "E0596"} \
            or ("cannot assign" in msg and "immutable" in msg) \
            or ("cannot borrow" in msg and "as mutable" in msg):
        return Category.IMMUTABLE_ASSIGNMENT
    if "expected ';'" in msg or "expected semi" in msg or code == "EXPECTED_SEMICOLON" \
            or ("expected" in msg and "`;`" in msg):
        return Category.MISSING_SEMICOLON
    return None


# ---------------------------------------------------------------------------
# Structural edits
# ---------------------------------------------------------------------------
@dataclass
class Edit:
    start: int
    end: int
    text: str


def _apply_edits(source: bytes, edits: Sequence[Edit]) -> bytes:
    """Apply non-overlapping edits in reverse byte order (deterministic)."""
    result = source
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        result = result[:edit.start] + edit.text.encode("utf-8") + result[edit.end:]
    return result


def _symbol_from_message(message: str) -> Optional[str]:
    m = re.search(r"`([^`]+)`", message) or re.search(r'"([^"]+)"', message)
    if not m:
        return None
    token = m.group(1)
    # For import paths (a::b::c / a.b.c) keep the leaf identifier.
    for sep in ("::", "."):
        if sep in token:
            token = token.split(sep)[-1]
    return token or None


# ---------------------------------------------------------------------------
# 3. The Self-Healing Mapping Matrix (P + R per category)
# ---------------------------------------------------------------------------
class SymbolIndex:
    """Maps declared function/type names to their defining module path (the DAG)."""

    def __init__(self) -> None:
        self.by_symbol: Dict[str, str] = {}

    @classmethod
    def build(cls, workspace: Path) -> "SymbolIndex":
        index = cls()
        for path in sorted(workspace.rglob("*")):
            if not path.is_file():
                continue
            language = detect_language(path)
            if language is None:
                continue
            try:
                text = path.read_text(encoding="utf-8")
                uast = parse_source(text, language)
            except Exception:
                continue
            symbols = extract_symbols(uast, text)
            rel = path.resolve().relative_to(workspace).as_posix()
            for name in symbols["functions"] + symbols["types"]:
                index.by_symbol.setdefault(name, rel)
        return index

    def module_for(self, symbol: str) -> Optional[str]:
        return self.by_symbol.get(symbol)


def _module_ref(rel_path: str, language: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", rel_path)
    if language == "python":
        return stem.replace("/", ".")
    if language == "rust":
        return "crate::" + stem.split("/")[-1]
    return stem


def heal_missing_import(source: bytes, diag: Diagnostic, language: str, index: Optional[SymbolIndex]) -> List[Edit]:
    symbol = _symbol_from_message(diag.message)
    if not symbol:
        return []
    module = index.module_for(symbol) if index else None
    if language == "python":
        if module:
            ref = _module_ref(module, "python")
            line = f"from {ref} import {symbol}\n" if "." in ref else f"import {ref}\n"
        else:
            line = f"import {symbol}\n"
    elif language == "rust":
        if module:
            line = f"use {_module_ref(module, 'rust')}::{symbol};\n"
        else:
            line = f"use crate::{symbol};\n"
    else:
        return []
    return [Edit(0, 0, line)]


def heal_immutable_assignment(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    if language != "rust":
        return []
    name = _symbol_from_message(diag.message)
    if not name:
        return []
    from tree_sitter import Parser, Query, QueryCursor

    lang_obj = _load_language("rust")
    tree = Parser(lang_obj).parse(source)
    query = Query(lang_obj, "(let_declaration pattern: (identifier) @name) @decl")
    caps = QueryCursor(query).captures(tree.root_node)
    decls = caps.get("decl", [])
    names = caps.get("name", [])
    for decl in sorted(decls, key=lambda n: n.start_byte):
        if any(c.type == "mutable_specifier" for c in decl.children):
            continue
        for name_node in names:
            if decl.start_byte <= name_node.start_byte and name_node.end_byte <= decl.end_byte:
                if source[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace") == name:
                    return [Edit(name_node.start_byte, name_node.start_byte, "mut ")]
    return []


def heal_missing_semicolon(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    if diag.start_byte is None:
        return []
    pos = max(0, min(diag.start_byte, len(source)))
    # Append the delimiter at the broken statement's byte boundary, unless one is
    # already present there.
    if pos < len(source) and source[pos:pos + 1] == b";":
        return []
    return [Edit(pos, pos, ";")]


# ---------------------------------------------------------------------------
# 3b. Advanced Rust compiler error handlers
# ---------------------------------------------------------------------------

def _crate_name_from_message(message: str) -> Optional[str]:
    """Extract the crate/module name from an E0433/E0405 diagnostic message.

    Typical formats:
        ``cannot find crate `foo` in ...``
        ``failed to resolve: use of undeclared crate or module `bar```
        ``cannot find trait `Baz` in ...``
    """
    m = re.search(r"(?:crate or module|crate|module|trait)\s+`([^`]+)`", message)
    if m:
        token = m.group(1)
        if "::" in token:
            return token.split("::")[0]
        return token
    return _symbol_from_message(message)


def _parse_cargo_dependencies(cargo_text: str) -> Dict[str, Tuple[int, int]]:
    """Return a mapping of crate name → (line_start, line_end) within [dependencies].

    Handles both inline (``crate = "version"``) and table-style
    (``crate = { version = "...", features = [...] }``) entries.
    """
    deps: Dict[str, Tuple[int, int]] = {}
    in_deps = False
    lines = cargo_text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("["):
            in_deps = stripped == "[dependencies]"
            i += 1
            continue
        if in_deps:
            m = re.match(r'^(\w[\w-]*)\s*=', lines[i])
            if m:
                name = m.group(1)
                start = i
                # Multi-line table values: if the value opens with '{' but
                # doesn't close on the same line, consume continuation lines.
                if '{' in lines[i] and '}' not in lines[i]:
                    while i < len(lines) - 1 and '}' not in lines[i]:
                        i += 1
                deps[name] = (start, i)
        i += 1
    return deps


def heal_missing_crate(
    source: bytes, diag: Diagnostic, language: str,
    *, cargo_toml_path: Optional[Path] = None,
) -> List[Edit]:
    """E0433 / E0405: inject a missing crate into ``Cargo.toml``'s [dependencies].

    If a ``Cargo.toml`` path is provided (or can be inferred from the diagnostic
    file path), the engine adds ``<crate> = "*"`` under ``[dependencies]``.
    The edit targets the Cargo.toml bytes, not the Rust source itself; the caller
    must apply the returned edits to the **manifest file**, not the source file.

    Smart merging: if the crate already exists in [dependencies] (with version
    pins, feature flags, etc.), the existing entry is preserved as-is.  Only
    truly absent crates are appended.
    """
    if language != "rust":
        return []
    crate_name = _crate_name_from_message(diag.message)
    if not crate_name:
        return []

    # Resolve the manifest path.
    manifest = cargo_toml_path
    if manifest is None:
        src_path = Path(diag.file)
        for parent in (src_path.parent, *src_path.parents):
            candidate = parent / "Cargo.toml"
            if candidate.is_file():
                manifest = candidate
                break

    if manifest is not None and manifest.is_file():
        cargo_text = manifest.read_text(encoding="utf-8")
        existing_deps = _parse_cargo_dependencies(cargo_text)

        # If the crate is already declared, preserve its version/features.
        if crate_name in existing_deps:
            return []

        dep_line = f'{crate_name} = "*"'
        if "[dependencies]" in cargo_text:
            idx = cargo_text.index("[dependencies]") + len("[dependencies]")
            nl = cargo_text.find("\n", idx)
            if nl == -1:
                nl = len(cargo_text)
            insert_pos = nl + 1
            new_cargo = cargo_text[:insert_pos] + dep_line + "\n" + cargo_text[insert_pos:]
        else:
            new_cargo = cargo_text.rstrip("\n") + "\n\n[dependencies]\n" + dep_line + "\n"
        manifest.write_text(new_cargo, encoding="utf-8")

    return []


def _return_type_text(node: Any, source: bytes) -> str:
    """Return the textual return type of a function item, or ``""``."""
    rt = node.child_by_field_name("return_type")
    if rt is None:
        return ""
    return source[rt.start_byte:rt.end_byte].decode("utf-8", "replace")


def _returns_nested_array(node: Any, source: bytes) -> bool:
    """True if the function returns a 2D stack array, e.g. ``[[Complex; 4]; 4]``."""
    return bool(re.search(r"\[\s*\[", _return_type_text(node, source)))


def _returns_flat_sequence(node: Any, source: bytes) -> bool:
    """True if the function returns a contiguous flat/heap sequence.

    Covers ``Vec<T>`` (1D heap), ``&[T]`` / ``[T]`` slices and ``Box<[T]>`` —
    the continuous-memory shapes that flat-slice call sites (``&[Complex]``)
    expect.
    """
    text = _return_type_text(node, source)
    if re.search(r"\[\s*\[", text):  # nested array is *not* flat
        return False
    return bool(
        re.search(r"\bVec\s*<", text)
        or re.search(r"&\s*\[", text)
        or re.search(r"\bBox\s*<\s*\[", text)
        or re.fullmatch(r"\[\s*[^\[\];]+;\s*[^\[\];]+\]", text.strip())  # 1D [T; N]
    )


def heal_duplicate_definition(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    """E0428: resolve a duplicate symbol, type-awarely where possible.

    Uses Tree-sitter to locate every top-level definition of ``symbol``.

    Type-aware path: when one variant returns a 2D stack array
    (``[[Complex; 4]; 4]``) and another returns a contiguous flat/heap shape
    (``Vec<Complex>`` / ``&[Complex]``), the 2D matrix variant is *renamed*
    (``<symbol>_legacy``) rather than deleted.  This preserves the
    continuous-memory variant that flat-slice call sites (e.g.
    ``apply_two_qudit_unitary`` expecting ``&[Complex]``) depend on, avoiding a
    type-breaking cascade.

    Fallback: when the variants are not type-discriminable, the *later*
    duplicate (by byte offset) is commented out, preserving the earlier
    definition downstream code may already depend on.
    """
    if language != "rust":
        return []
    symbol = _symbol_from_message(diag.message)
    if not symbol:
        return []

    from tree_sitter import Parser

    lang_obj = _load_language("rust")
    tree = Parser(lang_obj).parse(source)

    # Walk the top-level children looking for items that declare ``symbol``.
    defs: List[Any] = []
    name_nodes: List[Any] = []
    for child in tree.root_node.children:
        name_node = _find_name_in_item(child, symbol, source)
        if name_node is not None:
            defs.append(child)
            name_nodes.append(name_node)

    if len(defs) < 2:
        return []

    # Type-aware resolution: rename 2D matrix variant(s) to <symbol>_legacy and
    # keep the flat continuous-memory variant the call sites expect.
    nested = [
        (node, nm)
        for node, nm in zip(defs, name_nodes)
        if node.type == "function_item" and _returns_nested_array(node, source)
    ]
    flat = [
        node
        for node in defs
        if node.type == "function_item" and _returns_flat_sequence(node, source)
    ]
    if nested and flat and len(nested) < len(defs):
        edits: List[Edit] = []
        for _node, nm in nested:
            edits.append(Edit(nm.start_byte, nm.end_byte, f"{symbol}_legacy"))
        return edits

    # Fallback: comment out the *later* duplicate (preserve the first).
    dup = defs[-1]
    dup_text = source[dup.start_byte:dup.end_byte].decode("utf-8", "replace")
    commented = "/* [auto-healed: duplicate removed]\n" + dup_text + "\n*/"
    return [Edit(dup.start_byte, dup.end_byte, commented)]


def _find_name_in_item(node: Any, symbol: str, source: bytes) -> Optional[Any]:
    """Return the name child node if ``node`` declares ``symbol``, else None."""
    # Item types that carry a name: fn, struct, enum, type_alias, const, static,
    # trait, impl (impl doesn't always have a direct name node, skip for now).
    for child in node.children:
        if child.type == "identifier" or child.type == "type_identifier":
            text = source[child.start_byte:child.end_byte].decode("utf-8", "replace")
            if text == symbol:
                return child
    return None


_PYMODULE_ITEM_TYPES = ("mod_item", "function_item")
_NAMED_ITEM_TYPES = (
    "function_item",
    "struct_item",
    "enum_item",
    "union_item",
    "type_item",
    "const_item",
    "static_item",
    "trait_item",
)


def _find_pymodule_item(root: Any, source: bytes) -> Optional[Any]:
    """Locate the item carrying a ``#[pymodule]`` attribute.

    In tree-sitter-rust the attribute is an ``attribute_item`` sibling that
    *precedes* the annotated ``mod``/``fn`` item, so we track pending attributes
    while scanning the top-level children.
    """
    pending_pymodule = False
    for child in root.children:
        if child.type == "attribute_item":
            text = source[child.start_byte:child.end_byte].decode("utf-8", "replace")
            if "pymodule" in text:
                pending_pymodule = True
            continue
        if child.type in _PYMODULE_ITEM_TYPES and pending_pymodule:
            return child
        # Some grammars nest the attribute inside the item; check there too.
        if child.type in _PYMODULE_ITEM_TYPES:
            for sub in child.children:
                if sub.type == "attribute_item":
                    text = source[sub.start_byte:sub.end_byte].decode("utf-8", "replace")
                    if "pymodule" in text:
                        return child
        pending_pymodule = False
    return None


def _find_super_glob_use(node: Any, source: bytes) -> Optional[Any]:
    """Find a ``use super::*;`` declaration anywhere inside ``node``."""
    stack = list(node.children)
    while stack:
        current = stack.pop(0)
        if current.type == "use_declaration":
            text = source[current.start_byte:current.end_byte].decode("utf-8", "replace")
            if "super" in text and "*" in text:
                return current
        stack.extend(current.children)
    return None


def _item_name(node: Any, source: bytes) -> Optional[str]:
    """Return the declared name of a named top-level item, if any."""
    name = node.child_by_field_name("name")
    if name is not None:
        return source[name.start_byte:name.end_byte].decode("utf-8", "replace")
    for child in node.children:
        if child.type in ("identifier", "type_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return None


def _collect_top_level_item_names(root: Any, source: bytes, exclude: Any) -> List[str]:
    """Collect names of all top-level structs, enums and functions (in order)."""
    names: List[str] = []
    seen: set = set()
    for child in root.children:
        if child is exclude:
            continue
        if child.type not in _NAMED_ITEM_TYPES:
            continue
        name = _item_name(child, source)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def heal_pyo3_glob_import(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    """Expand a glob ``use super::*;`` inside a ``#[pymodule]`` into named imports.

    PyO3's declarative ``#[pymodule]`` macro rejects glob re-exports.  This locates
    the ``#[pymodule]`` block via Tree-sitter, finds the recursive wildcard
    statement (``use super::*;``), queries the outer file's AST for every
    top-level structure / enum / function, and rewrites the glob into an explicit
    named import map: ``use super::{ItemA, ItemB, ItemC};``.
    """
    if language != "rust":
        return []

    from tree_sitter import Parser

    lang_obj = _load_language("rust")
    root = Parser(lang_obj).parse(source).root_node

    pymodule = _find_pymodule_item(root, source)
    if pymodule is None:
        return []

    glob = _find_super_glob_use(pymodule, source)
    if glob is None:
        return []

    names = _collect_top_level_item_names(root, source, exclude=pymodule)
    if not names:
        return []

    replacement = "use super::{" + ", ".join(names) + "};"
    return [Edit(glob.start_byte, glob.end_byte, replacement)]


def heal_unresolved_parent_import(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    """E0432 with ``super::``: promote a private item to ``pub(crate)`` visibility.

    Locates the parent scope source file (inferred from the import path), finds
    the referenced private item, and prepends ``pub(crate) `` to its definition.
    Because the parent file is a *different* file, this function edits the source
    of the file containing the target symbol in-place and returns an empty list
    (the edits are applied as a side-effect so the next build picks them up).
    """
    if language != "rust":
        return []

    # Extract the symbol from the ``super::<symbol>`` pattern.
    m = re.search(r"super::(`?)(\w+)`?", diag.message)
    if not m:
        return []
    symbol = m.group(2)

    # Resolve the parent module file.
    src_path = Path(diag.file)
    parent_candidates = [
        src_path.parent.parent / "mod.rs",
        src_path.parent.parent / (src_path.parent.name + ".rs"),
        src_path.parent / "mod.rs",
    ]
    parent_file: Optional[Path] = None
    for candidate in parent_candidates:
        if candidate.is_file():
            parent_file = candidate
            break

    if parent_file is None:
        return []

    parent_source = parent_file.read_bytes()

    from tree_sitter import Parser

    lang_obj = _load_language("rust")
    tree = Parser(lang_obj).parse(parent_source)

    for child in tree.root_node.children:
        name_node = _find_name_in_item(child, symbol, parent_source)
        if name_node is None:
            continue
        # Check if already pub.
        first_token = parent_source[child.start_byte:child.start_byte + 4]
        if first_token.startswith(b"pub"):
            continue
        # Prepend pub(crate) to the item.
        new_source = (
            parent_source[:child.start_byte]
            + b"pub(crate) "
            + parent_source[child.start_byte:]
        )
        parent_file.write_bytes(new_source)
        break

    return []


def heal_mismatched_slice_type(source: bytes, diag: Diagnostic, language: str) -> List[Edit]:
    """E0308: inject ``.flatten()`` when an array-of-arrays is passed as a flat slice.

    Pattern: ``expected `&[T]`, found `&[[T; N]; M]``

    Locates the call-site expression at the diagnostic byte span and appends
    ``.flatten()`` (or wraps with ``.as_flattened()``) to coerce the nested array
    into a contiguous slice reference.
    """
    if language != "rust":
        return []
    if diag.start_byte is None or diag.end_byte is None:
        return []

    # Validate this is genuinely a nested-array-to-slice mismatch.
    if not re.search(r"expected `&\[", diag.message):
        return []
    if not re.search(r"found `&\[\[", diag.message):
        return []

    start = diag.start_byte
    end = diag.end_byte

    if start >= len(source) or end > len(source):
        return []

    expr_bytes = source[start:end]

    # If the expression already has .flatten() or .as_flattened(), skip.
    if b".flatten()" in expr_bytes or b".as_flattened()" in expr_bytes:
        return []

    # Find the end of the full expression (scan to whitespace / comma / semicolon / paren).
    scan_pos = end
    while scan_pos < len(source) and source[scan_pos:scan_pos + 1] not in (
        b",", b";", b")", b"]", b"}", b" ", b"\n", b"\t",
    ):
        scan_pos += 1

    # If the reference token is `&`, we need to wrap: `&<expr>.flatten()`.
    # But more commonly the span covers the whole argument; just append.
    return [Edit(scan_pos, scan_pos, ".flatten()")]


# ---------------------------------------------------------------------------
# 4. Bounded correction loop + rollback
# ---------------------------------------------------------------------------
@dataclass
class HealReport:
    path: str
    success: bool
    attempts: int
    rolled_back: bool = False
    applied: List[str] = field(default_factory=list)   # categories applied per attempt
    final_diagnostics: List[Diagnostic] = field(default_factory=list)
    reason: Optional[str] = None


BuildFn = Callable[[Path], List[Diagnostic]]


def _plan_edits(source: bytes, diags: Sequence[Diagnostic], language: str,
                index: Optional[SymbolIndex]) -> List[Edit]:
    """H's pattern+template stage: produce non-overlapping edits for this attempt."""
    edits: List[Edit] = []
    occupied: List[range] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end < r.start or start > r.stop) for r in occupied)

    # Deterministic order: by category then byte position.
    ordered = sorted(
        [(categorize(d), d) for d in diags],
        key=lambda cd: (cd[0].value if cd[0] else "zzz", cd[1].start_byte or 0),
    )
    for category, diag in ordered:
        if category is None:
            continue
        if category == Category.MISSING_IMPORT:
            new = heal_missing_import(source, diag, language, index)
        elif category == Category.IMMUTABLE_ASSIGNMENT:
            new = heal_immutable_assignment(source, diag, language)
        elif category == Category.MISSING_SEMICOLON:
            new = heal_missing_semicolon(source, diag, language)
        elif category == Category.MISSING_CRATE:
            new = heal_missing_crate(source, diag, language)
        elif category == Category.DUPLICATE_DEFINITION:
            new = heal_duplicate_definition(source, diag, language)
        elif category == Category.UNRESOLVED_PARENT_IMPORT:
            new = heal_unresolved_parent_import(source, diag, language)
        elif category == Category.MISMATCHED_SLICE_TYPE:
            new = heal_mismatched_slice_type(source, diag, language)
        elif category == Category.PYO3_GLOB_IMPORT:
            new = heal_pyo3_glob_import(source, diag, language)
        else:
            new = []
        for edit in new:
            if overlaps(edit.start, edit.end):
                continue
            edits.append(edit)
            occupied.append(range(edit.start, max(edit.start, edit.end)))
    return edits


def heal_module(
    path: _PathLike,
    build_fn: BuildFn,
    *,
    language: Optional[str] = None,
    workspace: Optional[_PathLike] = None,
    blueprint_path: Optional[_PathLike] = None,
    max_attempts: int = MAX_BUILD_ATTEMPTS,
    symbol_index: Optional[SymbolIndex] = None,
) -> HealReport:
    """Run the bounded self-healing loop for a single module.

    ``build_fn(path)`` must return the structured diagnostics for the current
    on-disk source (an empty list means a clean build).  After at most
    ``max_attempts`` builds the source is either healed or rolled back.
    """
    file_path = Path(path).resolve()
    resolved_language = language or detect_language(file_path)
    if resolved_language is None:
        raise ValueError(f"Unsupported file extension: {file_path.suffix!r}")

    ws = Path(workspace).resolve() if workspace else file_path.parent
    if symbol_index is None and resolved_language in ("python", "rust"):
        symbol_index = SymbolIndex.build(ws)

    original = file_path.read_bytes()

    # Snapshot manifests that side-effect healers may modify (e.g. Cargo.toml)
    # so we can restore them cleanly on rollback without losing user config.
    manifest_snapshots: Dict[Path, bytes] = {}
    for manifest_name in ("Cargo.toml",):
        for search_dir in (file_path.parent, *file_path.parents):
            candidate = search_dir / manifest_name
            if candidate.is_file():
                manifest_snapshots[candidate] = candidate.read_bytes()
                break

    report = HealReport(path=file_path.as_posix(), success=False, attempts=0)

    for attempt in range(1, max_attempts + 1):
        report.attempts = attempt
        diags = build_fn(file_path)
        if not diags:
            report.success = True
            return report
        report.final_diagnostics = diags

        # Last attempt failed -> do not edit further; fall through to rollback.
        if attempt == max_attempts:
            break

        source = file_path.read_bytes()
        edits = _plan_edits(source, diags, resolved_language, symbol_index)
        if not edits:
            report.reason = "no applicable healing rule"
            break
        healed = _apply_edits(source, edits)
        if healed == source:
            report.reason = "healing made no progress"
            break
        file_path.write_bytes(healed)
        for category, diag in ((categorize(d), d) for d in diags):
            if category:
                report.applied.append(category.value)

    # Exhausted / stuck -> rollback to the original clean state and flag.
    # Restore both source and any manifests modified by side-effect healers.
    file_path.write_bytes(original)
    for mpath, mdata in manifest_snapshots.items():
        if mpath.is_file():
            mpath.write_bytes(mdata)
    report.rolled_back = True
    if report.reason is None:
        report.reason = f"unresolved after {report.attempts} build attempt(s)"
    if blueprint_path is not None:
        rel = _safe_rel(file_path, ws)
        flag_healing_failure(blueprint_path, rel, report.reason, report.final_diagnostics)
    return report


def _safe_rel(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# blueprint.aero failure flagging
# ---------------------------------------------------------------------------
def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strip_table(text: str, name: str) -> str:
    out: List[str] = []
    skip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            skip = (stripped == f"[{name}]" or stripped.startswith(f"[{name}.")
                    or stripped.startswith(f"[[{name}"))
        if not skip:
            out.append(line)
    result = "\n".join(out).rstrip("\n")
    return result + "\n" if result else ""


def flag_healing_failure(
    blueprint_path: _PathLike, module: str, reason: str, diagnostics: Sequence[Diagnostic]
) -> None:
    """Record a self-healing failure under ``[self_healing]`` in the blueprint."""
    bp_path = Path(blueprint_path)
    text = bp_path.read_text(encoding="utf-8") if bp_path.is_file() else ""

    # Parse any existing failures so we update the table idempotently.
    existing: Dict[str, str] = {}
    try:
        from src.blueprint.loader import _toml as _t

        if text:
            parsed = _t.loads(text)
            for key, value in (parsed.get("self_healing") or {}).items():
                if isinstance(value, str):
                    existing[key] = value
    except Exception:
        pass

    codes = ",".join(sorted({d.code for d in diagnostics if d.code})) or "unknown"
    existing[module] = f"rolled_back: {reason} (codes: {codes})"

    text = _strip_table(text, "self_healing")
    lines = ["[self_healing]"]
    for key in sorted(existing):
        lines.append(f"{_toml_str(key)} = {_toml_str(existing[key])}")
    block = "\n".join(lines) + "\n"
    if text and not text.endswith("\n"):
        text += "\n"
    if text and not text.endswith("\n\n"):
        text += "\n"
    text += block
    bp_path.parent.mkdir(parents=True, exist_ok=True)
    bp_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Real compiler build functions (structured JSON interception)
# ---------------------------------------------------------------------------
def make_rust_build_fn(timeout: float = 60.0) -> BuildFn:
    """A build_fn that compiles a Rust file and returns structured diagnostics."""
    import tempfile

    def build(path: Path) -> List[Diagnostic]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = subprocess.run(
                ["rustc", "--error-format=json", "--edition", "2021", str(path), "-o", str(out)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=timeout, env=os.environ.copy(),
            )
            return parse_rustc_json(proc.stderr)

    return build


def make_c_build_fn(compiler: str = "cc", timeout: float = 60.0) -> BuildFn:
    """A build_fn that syntax-checks a C/C++ file and returns structured diagnostics."""
    def build(path: Path) -> List[Diagnostic]:
        proc = subprocess.run(
            [compiler, "-fsyntax-only", "-fdiagnostics-format=json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout, env=os.environ.copy(),
        )
        return parse_gcc_json(proc.stderr)

    return build
