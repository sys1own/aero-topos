"""
AeroNova Polyglot AST Parsing Engine
Provides an absolute, multi-era self-healing parsing layer capable of resolving
PyCapsule/ctypes struct offsets and falling back to zero-dependency flat parsers.
"""
import os
import sys
import ast
import re
import hashlib
import time
import warnings
import ctypes
import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core execution boundaries
MAX_FILE_BYTES = 1_000_000
PARSE_TIMEOUT_SECONDS = 5.0

LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".c": "c", ".h": "c",
    ".f": "fortran", ".f90": "fortran", ".f95": "fortran",
    ".f03": "fortran", ".for": "fortran",
    ".cob": "cobol", ".cbl": "cobol", ".cpy": "cobol",
}

GRAMMAR_MODULES: Dict[str, str] = {
    "python": "tree_sitter_python",
    "rust": "tree_sitter_rust",
    "cpp": "tree_sitter_cpp",
    "c": "tree_sitter_c",
    "fortran": "tree_sitter_fortran",
    "cobol": "tree_sitter_cobol",
}

_LANGUAGE_CACHE: Dict[str, Any] = {}

class UniversalParseError(Exception):
    """Raised when no grammar can be resolved/loaded for a language."""
    pass

class NodeCategory(str, Enum):
    DECLARATION = "DECLARATION"
    STATEMENT = "STATEMENT"
    EXPRESSION = "EXPRESSION"
    IDENTIFIER = "IDENTIFIER"
    LITERAL = "LITERAL"
    TYPE = "TYPE"
    KEYWORD = "KEYWORD"
    COMMENT = "COMMENT"
    PUNCTUATION = "PUNCTUATION"
    MODULE = "MODULE"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

CANONICAL_KIND: Dict[str, Dict[str, str]] = {
    "python": {
        "function_definition": "function_declaration",
        "class_definition": "type_declaration",
        "import_statement": "import_declaration",
        "import_from_statement": "import_declaration",
        "module": "translation_unit",
    },
    "rust": {
        "function_item": "function_declaration",
        "struct_item": "type_declaration",
        "enum_item": "type_declaration",
        "union_item": "type_declaration",
        "trait_item": "type_declaration",
        "type_item": "type_declaration",
        "use_declaration": "import_declaration",
        "mod_item": "module_declaration",
        "source_file": "translation_unit",
    },
    "cpp": {
        "function_definition": "function_declaration",
        "class_specifier": "type_declaration",
        "struct_specifier": "type_declaration",
        "namespace_definition": "module_declaration",
        "preproc_include": "import_declaration",
        "translation_unit": "translation_unit",
    },
    "c": {
        "function_definition": "function_declaration",
        "struct_specifier": "type_declaration",
        "union_specifier": "type_declaration",
        "preproc_include": "import_declaration",
        "translation_unit": "translation_unit",
    },
    "fortran": {
        "function": "function_declaration",
        "subroutine": "function_declaration",
        "module": "module_declaration",
        "derived_type_definition": "type_declaration",
        "translation_unit": "translation_unit",
    },
    "cobol": {
        "method_definition": "function_declaration",
        "paragraph": "function_declaration",
        "program_definition": "module_declaration",
    },
}

_COMMENT_TYPES = {"comment", "line_comment", "block_comment", "doc_comment"}
_DECLARATION_SUFFIXES = ("_definition", "_declaration", "_item", "_specifier", "_declarator")
_STATEMENT_SUFFIXES = ("_statement",)
_EXPRESSION_SUFFIXES = ("_expression", "_expr")
_LITERAL_HINTS = ("literal", "number", "string", "char", "integer", "float", "boolean", "true", "false")
_IDENTIFIER_HINTS = ("identifier",)
_TYPE_HINTS = ("type_identifier", "primitive_type", "type")

def supported_extensions() -> List[str]:
    return sorted(LANGUAGE_BY_EXTENSION)

def supported_languages() -> List[str]:
    return sorted(set(LANGUAGE_BY_EXTENSION.values()))

def detect_language(path: Union[str, Path]) -> Optional[str]:
    return LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower())

def _canonical_kind(language: str, node_type: str) -> Optional[str]:
    return CANONICAL_KIND.get(language, {}).get(node_type)

def _categorize(language: str, node_type: str, named: bool, is_error: bool, is_missing: bool) -> NodeCategory:
    if is_error or is_missing or node_type == "ERROR": return NodeCategory.ERROR
    if node_type in _COMMENT_TYPES: return NodeCategory.COMMENT
    canonical = _canonical_kind(language, node_type)
    if canonical:
        if canonical.endswith("_declaration"): return NodeCategory.DECLARATION
        if canonical.endswith("translation_unit"): return NodeCategory.MODULE
    if not named: return NodeCategory.KEYWORD if node_type.isalpha() else NodeCategory.PUNCTUATION
    lowered = node_type.lower()
    if any(lowered.endswith(s) for s in _DECLARATION_SUFFIXES): return NodeCategory.DECLARATION
    if any(lowered.endswith(s) for s in _STATEMENT_SUFFIXES): return NodeCategory.STATEMENT
    if any(lowered.endswith(s) for s in _EXPRESSION_SUFFIXES): return NodeCategory.EXPRESSION
    if lowered in _TYPE_HINTS: return NodeCategory.TYPE
    if any(h in lowered for h in _IDENTIFIER_HINTS): return NodeCategory.IDENTIFIER
    if any(h in lowered for h in _LITERAL_HINTS): return NodeCategory.LITERAL
    return NodeCategory.UNKNOWN

def _load_language(language: str):
    if language in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[language]
    module_name = GRAMMAR_MODULES.get(language)
    if module_name is None:
        raise UniversalParseError(f"No grammar module registered for language {language!r}")
    try:
        grammar = importlib.import_module(module_name)
        if type(grammar).__name__ == "Language" or hasattr(grammar, "query"):
            _LANGUAGE_CACHE[language] = grammar
            return grammar
        raw_lang = grammar.language()
        if type(raw_lang).__name__ == "Language" or hasattr(raw_lang, "query"):
            _LANGUAGE_CACHE[language] = raw_lang
            return raw_lang

        from tree_sitter import Language as TSLanguage
        if isinstance(raw_lang, int):
            try:
                ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
                ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
                capsule = ctypes.pythonapi.PyCapsule_New(raw_lang, b"tree_sitter.Language", None)
                lang_obj = TSLanguage(capsule, language)
                _LANGUAGE_CACHE[language] = lang_obj
                return lang_obj
            except Exception: pass
        try:
            lang_obj = TSLanguage(raw_lang, language)
        except TypeError:
            lang_obj = TSLanguage(raw_lang)
        _LANGUAGE_CACHE[language] = lang_obj
        return lang_obj
    except Exception as exc:
        raise UniversalParseError(f"Failed to load grammar module {module_name}: {exc}")

def _build_parser(language: str):
    from tree_sitter import Parser
    parser = Parser()
    lang = _load_language(language)
    try:
        parser.language = lang
    except (TypeError, AttributeError):
        parser.set_language(lang)
    return parser

def _flatten(root, language: str, source: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    nodes: List[Dict[str, Any]] = []
    counts = {"error": 0, "missing": 0}
    def visit(node, parent_id: Optional[int]) -> int:
        node_id = len(nodes)
        is_error = bool(node.is_error) or node.type == "ERROR"
        is_missing = bool(node.is_missing)
        if is_error: counts["error"] += 1
        if is_missing: counts["missing"] += 1
        category = _categorize(language, node.type, node.is_named, is_error, is_missing)
        record = {
            "id": node_id, "parent": parent_id, "children": [], "type": node.type,
            "grammar": getattr(node, "grammar_name", node.type), "category": category.value,
            "canonical_kind": _canonical_kind(language, node.type), "named": bool(node.is_named),
            "start_byte": node.start_byte, "end_byte": node.end_byte,
            "start_point": [node.start_point[0], node.start_point[1]], "end_point": [node.end_point[0], node.end_point[1]],
            "is_error": is_error, "is_missing": is_missing,
        }
        if is_error or is_missing:
            record["canonical_kind"] = "missing_placeholder" if is_missing else "error_placeholder"
        nodes.append(record)
        for child in node.children:
            child_id = visit(child, node_id)
            record["children"].append(child_id)
        if not node.children:
            record["text"] = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
        return node_id
    visit(root, None)
    return nodes, counts

# --- FALLBACK STRUCTURAL ENGINES ---
def _parse_fallback_python(source: str) -> Dict[str, Any]:
    parsed = ast.parse(source)
    lines = source.splitlines()
    def get_byte(l, c):
        if l < 0 or l >= len(lines): return 0
        return sum(len(line) + 1 for line in lines[:l]) + c
    def walk(node):
        ntype = type(node).__name__
        sl = getattr(node, "lineno", 1) - 1
        sc = getattr(node, "col_offset", 0)
        el = getattr(node, "end_lineno", sl + 1) - 1
        ec = getattr(node, "end_col_offset", sc)
        sb = get_byte(sl, sc)
        eb = get_byte(el, ec)
        text = ""
        try:
            if sl == el: text = lines[sl][sc:ec]
            else: text = "\n".join([lines[sl][sc:]] + lines[sl+1:el] + [lines[el][:ec]])
        except Exception: pass
        children = []
        for f, v in ast.iter_fields(node):
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, ast.AST): children.append(walk(item))
            elif isinstance(v, ast.AST): children.append(walk(v))
        return {"type": f"ast_{ntype}", "text": text, "start_byte": sb, "end_byte": max(sb, eb), "start_point": [max(0, sl), max(0, sc)], "end_point": [max(0, el), max(0, ec)], "children": children, "named": True}
    return {"type": "module", "text": source, "start_byte": 0, "end_byte": len(source.encode("utf-8")), "start_point": [0,0], "end_point": [max(0, len(lines)-1), max(0, len(lines[-1]) if lines else 0)], "children": [walk(parsed)], "named": True}

def _parse_fallback_lexical(source: str, language: str) -> Dict[str, Any]:
    lines = source.splitlines()
    s_bytes = source.encode("utf-8")
    root = {"type": "translation_unit" if language != "python" else "module", "text": source, "start_byte": 0, "end_byte": len(s_bytes), "start_point": [0,0], "end_point": [max(0, len(lines)-1), max(0, len(lines[-1]) if lines else 0)], "children": [], "named": True}
    comment_rx = re.compile(r"(\/\/.*|\/\*[\s\S]*?\*\/|#.*)")
    string_rx = re.compile(r"(\"[^\"]*\"|'[^']*')")
    word_rx = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)")
    brace_rx = re.compile(r"([{}()\[\];])")
    combined_rx = re.compile(f"{comment_rx.pattern}|{string_rx.pattern}|{word_rx.pattern}|{brace_rx.pattern}")
    curr_byte = 0
    stack = [root]
    for line_num, line_str in enumerate(lines):
        l_bytes = line_str.encode("utf-8")
        for match in combined_rx.finditer(line_str):
            tok = match.group(0)
            sb = curr_byte + len(line_str[:match.start()].encode("utf-8"))
            eb = curr_byte + len(line_str[:match.end()].encode("utf-8"))
            ntype = "token"
            if tok in ["{", "(", "["]: ntype = "block_start"
            elif tok in ["}", ")", "]"]: ntype = "block_end"
            elif tok == ";": ntype = "delimiter"
            elif tok.startswith(("//", "/*", "#")): ntype = "comment"
            elif tok.startswith(("\"", "'")): ntype = "string"
            else: ntype = "keyword" if tok in ["fn", "def", "function", "class", "struct", "impl", "let", "const", "var"] else "identifier"
            node = {"type": ntype, "text": tok, "start_byte": sb, "end_byte": eb, "start_point": [line_num, match.start()], "end_point": [line_num, match.end()], "children": [], "named": True}
            if ntype == "block_start":
                block = {"type": "lexical_block", "text": tok, "start_byte": sb, "end_byte": eb, "start_point": [line_num, match.start()], "end_point": [line_num, match.end()], "children": [node], "named": True}
                stack[-1]["children"].append(block)
                stack.append(block)
            elif ntype == "block_end":
                stack[-1]["children"].append(node)
                if len(stack) > 1:
                    closed = stack.pop()
                    closed["end_byte"] = eb
                    closed["end_point"] = [line_num, match.end()]
                    try: closed["text"] = s_bytes[closed["start_byte"]:eb].decode("utf-8", "replace")
                    except Exception: pass
            else:
                stack[-1]["children"].append(node)
        curr_byte += len(l_bytes) + 1
    return root

def _flatten_fallback_tree(tree: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    flat_nodes = []
    def walk(current, parent_id: Optional[int] = None) -> int:
        node_id = len(flat_nodes)
        category = _categorize(language, current["type"], current.get("named", True), False, False)
        record = {
            "id": node_id, "parent": parent_id, "children": [], "type": current["type"],
            "grammar": current["type"], "category": category.value,
            "canonical_kind": _canonical_kind(language, current["type"]), "named": current.get("named", True),
            "start_byte": current.get("start_byte", 0), "end_byte": current.get("end_byte", 0),
            "start_point": current.get("start_point", (0, 0)), "end_point": current.get("end_point", (0, 0)),
            "is_error": False, "is_missing": False
        }
        flat_nodes.append(record)
        for child in current.get("children", []):
            child_id = walk(child, node_id)
            record["children"].append(child_id)
        if not current.get("children") and "text" in current:
            record["text"] = current["text"]
        return node_id
    walk(tree, None)
    return flat_nodes

# --- PUBLIC INTERFACES ---
def semantic_hash_from_nodes(nodes: List[Dict[str, Any]]) -> str:
    tokens = []
    for node in nodes:
        if node["category"] in ("COMMENT", "PUNCTUATION") or not node["named"]: continue
        tokens.append(node.get("canonical_kind") or node["type"])
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()

def _node_name(node_id: int, nodes: List[Dict[str, Any]], source_text: str) -> Optional[str]:
    node = nodes[node_id]
    queue = list(node["children"])
    while queue:
        cid = queue.pop(0)
        child = nodes[cid]
        if child["type"] in ("identifier", "type_identifier", "name", "field_identifier"):
            return child.get("text") or source_text[child["start_byte"]:child["end_byte"]]
        queue.extend(child["children"])
    return None

def extract_symbols(uast: Dict[str, Any], source: str = "") -> Dict[str, List[str]]:
    nodes = uast["nodes"]
    functions, types = [], []
    for node in nodes:
        kind = node.get("canonical_kind")
        if kind == "function_declaration":
            name = _node_name(node["id"], nodes, source)
            if name: functions.append(name)
        elif kind == "type_declaration":
            name = _node_name(node["id"], nodes, source)
            if name: types.append(name)
    return {"functions": sorted(set(functions)), "types": sorted(set(types))}

def _empty_uast(language: str, source: bytes, file_path: Optional[str], flags: Dict[str, Any]) -> Dict[str, Any]:
    return {"metadata": {"language": language, "file_path": file_path, "file_hash": hashlib.sha256(source).hexdigest(), "source_bytes": len(source), "byte_span": [0, len(source)], "node_count": 0, "parser_flags": flags}, "root": None, "nodes": [], "semantic_hash": hashlib.sha256(b"").hexdigest()}

def _fallback_uast(source: bytes, language: str, file_path: Optional[str], flags: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Build a fallback UAST when the native grammar is unavailable or fails.

    Uses the lightweight ``ast``-based (Python) or lexical (other languages)
    tree builder, flattens it into the standard node array, and stamps the
    parser flags with ``fallback=True`` and the supplied *reason*.
    """
    source_str = source.decode("utf-8", "replace")
    fallback_tree = _parse_fallback_python(source_str) if language == "python" else _parse_fallback_lexical(source_str, language)
    nodes = _flatten_fallback_tree(fallback_tree, language)
    flags = dict(flags, fallback=True, fallback_reason=reason)
    uast = {"metadata": {"language": language, "file_path": file_path, "file_hash": hashlib.sha256(source).hexdigest(), "source_bytes": len(source), "byte_span": [0, len(source)], "node_count": len(nodes), "parser_flags": flags}, "root": 0 if nodes else None, "nodes": nodes}
    uast["semantic_hash"] = semantic_hash_from_nodes(nodes)
    return uast


def parse_bytes(source: bytes, language: str, *, file_path: Optional[str] = None) -> Dict[str, Any]:
    if language not in supported_languages():
        raise UniversalParseError(f"Unsupported language: {language!r}")
    base_flags = {"fallback": False, "fallback_reason": None, "has_error": False, "error_nodes": 0, "missing_nodes": 0, "error_locations": [], "truncated": False, "parse_seconds": 0.0}
    if len(source) > MAX_FILE_BYTES:
        return _empty_uast(language, source, file_path, dict(base_flags, fallback=True, fallback_reason="file_too_large", truncated=True))
    
    start = time.monotonic()
    try:
        parser = _build_parser(language)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: parser.timeout_micros = int(PARSE_TIMEOUT_SECONDS * 1_000_000)
            except Exception: pass
        tree = parser.parse(source)
        if tree is None or tree.root_node is None or (tree.root_node.child_count == 0 and len(source) > 10):
            raise ValueError("Corrupted or misaligned binary grammar layout structure.")
        elapsed = time.monotonic() - start
        nodes, counts = _flatten(tree.root_node, language, source)
        err_locs = [{"node_id": n["id"], "kind": "missing" if n["is_missing"] else "error", "start_byte": n["start_byte"], "end_byte": n["end_byte"], "start_point": n["start_point"]} for n in nodes if n["is_error"] or n["is_missing"]]
        flags = dict(base_flags, has_error=bool(tree.root_node.has_error), error_nodes=counts["error"], missing_nodes=counts["missing"], error_locations=err_locs, parse_seconds=round(elapsed, 4))
        uast = {"metadata": {"language": language, "file_path": file_path, "file_hash": hashlib.sha256(source).hexdigest(), "source_bytes": len(source), "byte_span": [tree.root_node.start_byte, tree.root_node.end_byte], "node_count": len(nodes), "parser_flags": flags}, "root": 0 if nodes else None, "nodes": nodes}
        uast["semantic_hash"] = semantic_hash_from_nodes(nodes)
        return uast
    except Exception as exc:
        elapsed = time.monotonic() - start
        flags = dict(base_flags, parse_seconds=round(elapsed, 4))
        return _fallback_uast(source, language, file_path, flags, f"fallback_activated:{type(exc).__name__}")

def parse_source(source: str, language: str, *, file_path: Optional[str] = None) -> Dict[str, Any]:
    return parse_bytes(source.encode("utf-8"), language, file_path=file_path)

def parse_file(path: Union[str, Path], language: Optional[str] = None) -> Dict[str, Any]:
    file_path = Path(path)
    resolved = language or detect_language(file_path)
    if resolved is None: raise UniversalParseError(f"Unsupported file extension: {file_path.suffix!r}")
    return parse_bytes(file_path.read_bytes(), resolved, file_path=str(file_path))

def semantic_hash(source: str, language: str) -> str:
    return parse_source(source, language)["semantic_hash"]


def load_language(language: Optional[str] = None, *args, **kwargs):
    """Return the loaded Tree-sitter ``Language`` object for *language*.

    This is the public entry point used by the inference engine and the
    self-healing toolchain to build native parsers/queries.  The language name
    is resolved through the same cached grammar loader used internally, so a
    successfully registered grammar yields a real ``Language`` (never ``None``).
    """
    if language is None:
        raise UniversalParseError("load_language requires a language name")
    return _load_language(language)
