"""
Semantic Proximity Mapping Engine.

Cross-language analysis with Unified AST (UAST) generation.
Parses Python (via ``ast``), Rust, C, C++ and Fortran (via ``tree-sitter``)
source files, builds a unified directed graph, detects FFI bindings (PyO3,
``extern "C"``, ``cffi``/``ctypes`` and Fortran ``bind(c)``), surfaces GPU
kernels as dedicated nodes, and performs cross-language data-flow analysis.

The UAST normalises three cross-language concepts into a common vocabulary so
the rest of the tool can reason about a multi-million-line, multi-language
physics codebase uniformly:

* ``uast_function``    -- a function/subroutine definition.
* ``uast_call``        -- a function call site.
* ``uast_global``      -- a module/global variable or constant.
* ``uast_type``        -- a type/struct/class/typedef definition.
* ``uast_gpu_kernel``  -- a GPU kernel (``__global__`` / ``.cu`` entry point).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("analysis.semantic_mapper")


# ---------------------------------------------------------------------------
# Schema-tolerant node accessors
# ---------------------------------------------------------------------------
# The parsing layer can hand back either a native tree-sitter ``Node`` (with
# ``.type``/``.children`` attributes) or a uniform dictionary record (from the
# zero-dependency fallback engine, using ``node["type"]``/``node["children"]``).
# These accessors normalise both shapes and degrade gracefully when a field is
# missing or an unexpected type is encountered, so a single malformed node can
# never cascade an exception up into the orchestration loops.

def _ts_get(node: Any, key: str, default: Any) -> Any:
    try:
        if isinstance(node, dict):
            value = node.get(key, default)
        else:
            value = getattr(node, key, default)
    except Exception:
        return default
    return default if value is None else value


def _node_type(node: Any) -> str:
    return str(_ts_get(node, "type", "") or "")


def _node_children(node: Any) -> List[Any]:
    children = _ts_get(node, "children", [])
    return list(children) if isinstance(children, (list, tuple)) else []


def _node_start_byte(node: Any) -> int:
    try:
        return int(_ts_get(node, "start_byte", 0))
    except (TypeError, ValueError):
        return 0


def _node_end_byte(node: Any) -> int:
    try:
        return int(_ts_get(node, "end_byte", 0))
    except (TypeError, ValueError):
        return 0


def _node_start_point(node: Any) -> Tuple[int, int]:
    point = _ts_get(node, "start_point", (0, 0))
    try:
        return int(point[0]), int(point[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return 0, 0


class UnifiedASTNode:
    """Represents a node in the Unified AST."""

    def __init__(
        self,
        node_id: str,
        node_type: str,
        language: str,
        source_location: Tuple[str, int, int],
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.id = node_id
        self.type = node_type
        self.language = language
        self.source_location = source_location
        self.data = data or {}
        self.children: List[UnifiedASTNode] = []
        self.parent: Optional[UnifiedASTNode] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "language": self.language,
            "source_location": list(self.source_location),
            "data": self.data,
            "metadata": self.metadata,
            "child_count": len(self.children),
        }


class SemanticMapper:
    """
    Maps semantic relationships across language boundaries.

    Generates a Unified AST (UAST) ``networkx.DiGraph`` from Python and Rust
    source trees, detecting PyO3 FFI bindings and establishing cross-language
    data-flow edges.
    """

    # Map of UAST language name -> source file extensions.
    _LANGUAGE_EXTENSIONS = {
        "c": ("*.c", "*.h"),
        "cpp": ("*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh"),
        "fortran": ("*.f90", "*.f", "*.f03", "*.f08", "*.for"),
    }

    # GPU kernel extensions handled as dedicated kernel nodes (feature #5).
    _GPU_KERNEL_EXTENSIONS = ("*.cu", "*.cuh", "*.hip")

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.uast: nx.DiGraph = nx.DiGraph()
        self.ffi_registry: Dict[str, Dict[str, Any]] = {}
        self._function_signatures: Dict[str, Dict[str, Any]] = {}
        self.parsers = self._init_tree_sitter_parsers()
        # Backward-compatible alias used by older callers/tests.
        self.rust_parser = self.parsers.get("rust")

    # ------------------------------------------------------------------
    # tree-sitter parser bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def _init_tree_sitter_parsers() -> Dict[str, Any]:
        from core.parser.universal import _build_parser

        parsers: Dict[str, Any] = {}
        for lang in ("rust", "c", "cpp", "fortran"):
            try:
                parsers[lang] = _build_parser(lang)
            except Exception:
                # A missing/unbuildable grammar should degrade gracefully so the
                # remaining languages still load; the factory is the single
                # source of truth, we just skip whatever it can't resolve.
                continue
        return parsers

    @staticmethod
    def _init_rust_parser() -> Any:
        from core.parser.universal import _build_parser

        try:
            return _build_parser("rust")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_uast(self, project_root: Path) -> nx.DiGraph:
        """Build the Unified AST from the entire project."""
        analysis_cfg = self.config.get("analysis", {}).get("semantic_proximity_mapping", {})
        source_roots = analysis_cfg.get("source_roots", {})

        python_root = project_root / source_roots.get("python", "src/python")
        rust_root = project_root / source_roots.get("rust", "src/native")

        for py_file in self._glob_safe(python_root, "**/*.py"):
            self._parse_python_file(py_file)

        for py_file in self._glob_safe(project_root, "builder_brains/**/*.py"):
            self._parse_python_file(py_file)

        for rs_file in self._glob_safe(rust_root, "**/*.rs"):
            self._parse_rust_file(rs_file)

        # C / C++ / Fortran roots (feature #1).  Each language may declare its
        # own root; we fall back to a conventional default and always also scan
        # the shared ``src`` tree so legacy layouts keep working.
        default_roots = {"c": "src/c", "cpp": "src/cpp", "fortran": "src/fortran"}
        for lang, (default_dir) in default_roots.items():
            lang_root = project_root / source_roots.get(lang, default_dir)
            for pattern in self._LANGUAGE_EXTENSIONS[lang]:
                for src_file in self._glob_safe(lang_root, f"**/{pattern}"):
                    self._parse_tree_sitter_file(src_file, lang)

        # GPU kernels are surfaced as dedicated nodes (feature #5).  Patterns are
        # resolved relative to the project root so users can point at e.g.
        # ``src/kernels/*.cu``.
        gpu_cfg = self.config.get("gpu", {})
        kernel_patterns = list(gpu_cfg.get("kernel_sources", []))
        for pattern in kernel_patterns:
            for kernel_file in self._glob_safe(project_root, pattern):
                self._parse_gpu_kernel_file(kernel_file)
        # Always also pick up conventionally-named kernels even without config.
        for pattern in self._GPU_KERNEL_EXTENSIONS:
            for kernel_file in self._glob_safe(project_root, f"**/{pattern}"):
                self._parse_gpu_kernel_file(kernel_file)

        self._detect_pyo3_bindings()
        self._detect_c_abi_bindings()
        self._detect_python_c_bindings()
        self._create_cross_language_edges()
        self._create_gpu_kernel_edges()
        self._run_data_flow_analysis()
        return self.uast

    def get_ffi_bindings(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.ffi_registry)

    def get_statistics(self) -> Dict[str, Any]:
        nodes_by_language: Dict[str, int] = {}
        unified_node_counts: Dict[str, int] = {}
        for _, d in self.uast.nodes(data=True):
            lang = d.get("language", "unknown")
            nodes_by_language[lang] = nodes_by_language.get(lang, 0) + 1
            uast_kind = d.get("metadata", {}).get("uast_kind")
            if uast_kind:
                unified_node_counts[uast_kind] = unified_node_counts.get(uast_kind, 0) + 1

        def _count_edges(edge_type: str) -> int:
            return sum(
                1
                for _, _, d in self.uast.edges(data=True)
                if d.get("edge_type") == edge_type
            )

        ffi_by_type: Dict[str, int] = {}
        for info in self.ffi_registry.values():
            ffi_by_type[info.get("type", "unknown")] = ffi_by_type.get(info.get("type", "unknown"), 0) + 1

        return {
            "total_nodes": self.uast.number_of_nodes(),
            "total_edges": self.uast.number_of_edges(),
            "python_nodes": nodes_by_language.get("python", 0),
            "rust_nodes": nodes_by_language.get("rust", 0),
            "c_nodes": nodes_by_language.get("c", 0),
            "cpp_nodes": nodes_by_language.get("cpp", 0),
            "fortran_nodes": nodes_by_language.get("fortran", 0),
            "gpu_kernel_nodes": nodes_by_language.get("gpu", 0),
            "nodes_by_language": nodes_by_language,
            "unified_node_counts": unified_node_counts,
            "ffi_bindings": len(self.ffi_registry),
            "ffi_bindings_by_type": ffi_by_type,
            "ffi_edges": _count_edges("pyo3_ffi_bridge") + _count_edges("c_abi_bridge"),
            "gpu_kernel_edges": _count_edges("gpu_kernel"),
            "dataflow_edges": _count_edges("data_flow"),
        }

    def export_graph(self, path: Path) -> None:
        data: Dict[str, Any] = {"nodes": [], "edges": []}
        for node_id, node_data in self.uast.nodes(data=True):
            entry: Dict[str, Any] = {"id": node_id}
            for k, v in node_data.items():
                try:
                    json.dumps(v)
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)
            data["nodes"].append(entry)
        for src, dst, edge_data in self.uast.edges(data=True):
            entry = {"source": src, "target": dst}
            for k, v in edge_data.items():
                try:
                    json.dumps(v)
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)
            data["edges"].append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Python parsing
    # ------------------------------------------------------------------

    def _parse_python_file(self, file_path: Path) -> None:
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (OSError, SyntaxError) as exc:
            logger.debug("Skipping Python file %s: %s", file_path, exc)
            return

        # Pre-compute which assignment statements are module-level so they can be
        # classified as unified globals.
        top_level_assigns = {
            id(stmt)
            for stmt in getattr(tree, "body", [])
            if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        }

        for node in ast.walk(tree):
            lineno = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)
            node_id = self._generate_node_id(file_path, node)
            node_type = node.__class__.__name__
            src_snippet = None
            try:
                src_snippet = ast.unparse(node)
            except (ValueError, TypeError) as exc:
                logger.debug("ast.unparse failed for %s node in %s: %s", node_type, file_path, exc)

            uast_node = UnifiedASTNode(
                node_id=node_id,
                node_type=node_type,
                language="python",
                source_location=(str(file_path), lineno, col),
                data={"source": src_snippet},
            )
            uast_kind = self._python_uast_kind(node, top_level_assigns)
            if uast_kind:
                uast_node.metadata["uast_kind"] = uast_kind
                uast_node.metadata["name"] = self._python_node_name(node)
            self.uast.add_node(node_id, **uast_node.to_dict())

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._store_function_signature(file_path, node)

    @staticmethod
    def _python_uast_kind(node: Any, top_level_assigns: Set[int]) -> Optional[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "uast_function"
        if isinstance(node, ast.Call):
            return "uast_call"
        if isinstance(node, ast.ClassDef):
            return "uast_type"
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and id(node) in top_level_assigns:
            return "uast_global"
        return None

    @staticmethod
    def _python_node_name(node: Any) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
            return "<call>"
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            return targets[0] if targets else "<assign>"
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return ""

    def _store_function_signature(self, file_path: Path, node: ast.FunctionDef) -> None:
        sig_key = f"{file_path}::{node.name}"
        args = [a.arg for a in node.args.args]
        self._function_signatures[sig_key] = {
            "name": node.name,
            "file": str(file_path),
            "line": node.lineno,
            "args": args,
        }

    # ------------------------------------------------------------------
    # Rust parsing
    # ------------------------------------------------------------------

    def _parse_rust_file(self, file_path: Path) -> None:
        if not self.rust_parser:
            return
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = self.rust_parser.parse(source.encode("utf-8"))
            self._walk_rust_tree(tree.root_node, file_path, source)
        except OSError as exc:
            logger.debug("Cannot read Rust file %s: %s", file_path, exc)
        except Exception as exc:
            logger.warning("Failed to parse Rust file %s: %s", file_path, exc)

    def _walk_rust_tree(self, node: Any, file_path: Path, source: str) -> None:
        node_id = self._generate_rust_node_id(file_path, node)
        text = source[node.start_byte : node.end_byte]
        uast_node = UnifiedASTNode(
            node_id=node_id,
            node_type=node.type,
            language="rust",
            source_location=(str(file_path), node.start_point[0], node.start_point[1]),
            data={"text": text},
        )
        uast_kind = self._classify_ts_node(node, "rust")
        if uast_kind:
            uast_node.metadata["uast_kind"] = uast_kind
            uast_node.metadata["name"] = self._ts_node_name(node, source, "rust")
        self.uast.add_node(node_id, **uast_node.to_dict())

        # PyO3 attributes (``#[pyfunction]`` / ``#[pyfn(...)]``) are siblings of
        # the ``function_item`` they decorate, so inspect the previous sibling.
        if node.type == "function_item" and self._rust_node_has_pyo3_attr(node):
            match = re.search(r"\bfn\s+(\w+)", text)
            if match:
                func_name = match.group(1)
                self.ffi_registry[func_name] = {
                    "type": "pyo3_function",
                    "function_name": func_name,
                    "rust_node": node_id,
                    "file": str(file_path),
                    "line": node.start_point[0],
                }

        for child in node.children:
            child_id = self._generate_rust_node_id(file_path, child)
            self.uast.add_edge(node_id, child_id, edge_type="ast_child")
            self._walk_rust_tree(child, file_path, source)

    # ------------------------------------------------------------------
    # Generic tree-sitter parsing (C / C++ / Fortran)
    # ------------------------------------------------------------------

    def _parse_tree_sitter_file(self, file_path: Path, language: str) -> None:
        parser = self.parsers.get(language)
        if not parser:
            return
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = parser.parse(source.encode("utf-8"))
            self._walk_ts_tree(tree.root_node, file_path, source, language)
        except OSError as exc:
            logger.debug("Cannot read %s file %s: %s", language, file_path, exc)
        except Exception as exc:
            logger.warning("Failed to parse %s file %s: %s", language, file_path, exc)

    def _walk_ts_tree(self, node: Any, file_path: Path, source: str, language: str) -> None:
        node_id = self._generate_rust_node_id(file_path, node)
        start_byte, end_byte = _node_start_byte(node), _node_end_byte(node)
        text = source[start_byte:end_byte] if 0 <= start_byte <= end_byte else ""
        start_point = _node_start_point(node)
        uast_node = UnifiedASTNode(
            node_id=node_id,
            node_type=_node_type(node),
            language=language,
            source_location=(str(file_path), start_point[0], start_point[1]),
            data={"text": text if len(text) <= 4096 else text[:4096]},
        )
        uast_kind = self._classify_ts_node(node, language)
        if uast_kind:
            uast_node.metadata["uast_kind"] = uast_kind
            uast_node.metadata["name"] = self._ts_node_name(node, source, language)
        self.uast.add_node(node_id, **uast_node.to_dict())

        for child in _node_children(node):
            child_id = self._generate_rust_node_id(file_path, child)
            self.uast.add_edge(node_id, child_id, edge_type="ast_child")
            self._walk_ts_tree(child, file_path, source, language)

    # Per-language tree-sitter node-type -> unified kind mapping.
    _TS_KIND_MAP = {
        "rust": {
            "function_item": "uast_function",
            "call_expression": "uast_call",
            "macro_invocation": "uast_call",
            "static_item": "uast_global",
            "const_item": "uast_global",
            "struct_item": "uast_type",
            "enum_item": "uast_type",
            "type_item": "uast_type",
            "trait_item": "uast_type",
        },
        "c": {
            "function_definition": "uast_function",
            "call_expression": "uast_call",
            "type_definition": "uast_type",
            "struct_specifier": "uast_type",
            "enum_specifier": "uast_type",
            "union_specifier": "uast_type",
        },
        "cpp": {
            "function_definition": "uast_function",
            "call_expression": "uast_call",
            "type_definition": "uast_type",
            "struct_specifier": "uast_type",
            "class_specifier": "uast_type",
            "enum_specifier": "uast_type",
        },
        "fortran": {
            "function": "uast_function",
            "subroutine": "uast_function",
            "call_expression": "uast_call",
            "subroutine_call": "uast_call",
            "derived_type_definition": "uast_type",
        },
    }

    def _classify_ts_node(self, node: Any, language: str) -> Optional[str]:
        kind_map = self._TS_KIND_MAP.get(language, {})
        ntype = _node_type(node)
        if ntype in kind_map:
            # C/C++ top-level ``declaration`` nodes become globals only when they
            # are not function declarators (handled via dedicated node types).
            return kind_map[ntype]
        if language in ("c", "cpp") and ntype == "declaration":
            if not self._declaration_is_function(node):
                return "uast_global"
        if language == "fortran" and ntype == "variable_declaration":
            return "uast_global"
        return None

    @staticmethod
    def _declaration_is_function(node: Any) -> bool:
        for child in _node_children(node):
            if _node_type(child) == "function_declarator":
                return True
        return False

    @staticmethod
    def _ts_node_name(node: Any, source: str, language: str) -> str:
        """Best-effort extraction of the identifier a node defines or calls."""

        def text_of(n: Any) -> str:
            start, end = _node_start_byte(n), _node_end_byte(n)
            return source[start:end] if 0 <= start <= end else ""

        def find_first(n: Any, types: Tuple[str, ...], max_depth: int = 3) -> Optional[Any]:
            stack = [(n, 0)]
            while stack:
                cur, depth = stack.pop()
                if cur is not node and _node_type(cur) in types:
                    return cur
                if depth < max_depth:
                    for child in _node_children(cur):
                        stack.append((child, depth + 1))
            return None

        ntype = _node_type(node)
        if language == "rust":
            ident = find_first(node, ("identifier", "type_identifier", "field_identifier"))
            return text_of(ident) if ident else ""
        if language in ("c", "cpp"):
            if ntype in ("function_definition", "declaration"):
                declarator = find_first(node, ("function_declarator",))
                if declarator:
                    ident = find_first(declarator, ("identifier", "field_identifier", "type_identifier", "qualified_identifier"))
                    if ident:
                        return text_of(ident)
                ident = find_first(node, ("identifier", "type_identifier"))
                return text_of(ident) if ident else ""
            ident = find_first(node, ("identifier", "type_identifier", "field_identifier"))
            return text_of(ident) if ident else ""
        if language == "fortran":
            name_node = find_first(node, ("name",))
            if name_node:
                return text_of(name_node)
            ident = find_first(node, ("identifier",))
            return text_of(ident) if ident else ""
        return ""

    # ------------------------------------------------------------------
    # GPU kernels (feature #5)
    # ------------------------------------------------------------------

    def _parse_gpu_kernel_file(self, file_path: Path) -> None:
        """Register GPU kernels as dedicated UAST nodes.

        CUDA/HIP kernels are not fed through a full grammar; instead each
        ``__global__`` entry point becomes a ``uast_gpu_kernel`` node so that the
        host->device call edges can be modelled with the special ``gpu_kernel``
        edge type.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Cannot read GPU kernel file %s: %s", file_path, exc)
            return

        file_node_id = hashlib.md5(f"{file_path}:gpu_translation_unit".encode()).hexdigest()[:16]
        file_node = UnifiedASTNode(
            node_id=file_node_id,
            node_type="gpu_translation_unit",
            language="gpu",
            source_location=(str(file_path), 0, 0),
            data={"path": str(file_path)},
        )
        self.uast.add_node(file_node_id, **file_node.to_dict())

        kernel_pattern = re.compile(
            r"__global__\s+[\w:<>\*&\s]+?\b(\w+)\s*\(", re.MULTILINE
        )
        for match in kernel_pattern.finditer(source):
            kernel_name = match.group(1)
            line = source[: match.start()].count("\n")
            node_id = hashlib.md5(
                f"{file_path}:gpu_kernel:{kernel_name}:{line}".encode()
            ).hexdigest()[:16]
            kernel_node = UnifiedASTNode(
                node_id=node_id,
                node_type="gpu_kernel",
                language="gpu",
                source_location=(str(file_path), line, 0),
                data={"kernel_name": kernel_name},
            )
            kernel_node.metadata["uast_kind"] = "uast_gpu_kernel"
            kernel_node.metadata["name"] = kernel_name
            self.uast.add_node(node_id, **kernel_node.to_dict())
            self.uast.add_edge(file_node_id, node_id, edge_type="ast_child")
            self.ffi_registry[kernel_name] = {
                "type": "gpu_kernel",
                "function_name": kernel_name,
                "kernel_node": node_id,
                "file": str(file_path),
                "line": line,
            }

        # Capture in-file launch sites (``kernel<<<grid, block>>>(...)``) as
        # unified call nodes so host->device edges can be created even when the
        # launch lives inside the kernel translation unit itself.
        for match in re.finditer(r"\b(\w+)\s*<<<", source):
            launch_name = match.group(1)
            line = source[: match.start()].count("\n")
            call_id = hashlib.md5(
                f"{file_path}:gpu_launch:{launch_name}:{match.start()}".encode()
            ).hexdigest()[:16]
            call_node = UnifiedASTNode(
                node_id=call_id,
                node_type="gpu_launch",
                language="gpu",
                source_location=(str(file_path), line, 0),
                data={"text": source[match.start() : match.start() + 80]},
            )
            call_node.metadata["uast_kind"] = "uast_call"
            call_node.metadata["name"] = launch_name
            self.uast.add_node(call_id, **call_node.to_dict())
            self.uast.add_edge(file_node_id, call_id, edge_type="ast_child")

    # ------------------------------------------------------------------
    # FFI detection
    # ------------------------------------------------------------------

    def _detect_pyo3_bindings(self) -> None:
        for node_id, data in list(self.uast.nodes(data=True)):
            if data.get("language") != "rust":
                continue
            if data.get("type") == "function_item":
                text = data.get("data", {}).get("text", "")
                if "#[pyfunction]" in text or "#[pyfn" in text:
                    match = re.search(r"\bfn\s+(\w+)", text)
                    if match:
                        func_name = match.group(1)
                        self.ffi_registry[func_name] = {
                            "type": "pyo3_function",
                            "function_name": func_name,
                            "rust_node": node_id,
                            "source_location": data.get("source_location"),
                        }

    @staticmethod
    def _rust_node_has_pyo3_attr(node: Any) -> bool:
        sibling = node.prev_named_sibling
        # Skip over doc comments / other attributes to find an attribute_item.
        while sibling is not None and sibling.type in ("attribute_item", "line_comment", "block_comment"):
            if sibling.type == "attribute_item" and (
                "pyfunction" in sibling.text.decode("utf-8", "replace")
                or "pyfn" in sibling.text.decode("utf-8", "replace")
            ):
                return True
            sibling = sibling.prev_named_sibling
        return False

    def _register_pyo3_function(self, file_path: Path, node: Any, source: str) -> None:
        for child in node.children:
            if child.type == "function_item":
                text = source[child.start_byte : child.end_byte]
                match = re.search(r"\bfn\s+(\w+)", text)
                if match:
                    func_name = match.group(1)
                    self.ffi_registry[func_name] = {
                        "type": "pyo3_function",
                        "function_name": func_name,
                        "file": str(file_path),
                        "line": node.start_point[0],
                    }

    def _detect_c_abi_bindings(self) -> None:
        """Detect C-compatible bindings across Rust, C/C++ and Fortran.

        Recognises ``extern "C"`` in Rust, ``extern "C"`` linkage blocks and
        ``extern`` declarations in C/C++, and Fortran ``bind(c)`` interfaces.
        """
        for node_id, data in list(self.uast.nodes(data=True)):
            lang = data.get("language")
            ntype = data.get("type", "")
            text = data.get("data", {}).get("text", "") or ""
            meta = data.get("metadata", {})

            if lang == "rust":
                if ntype in ("function_item", "function_signature_item") and 'extern "C"' in text:
                    match = re.search(r"\bfn\s+(\w+)", text)
                    name = match.group(1) if match else meta.get("name")
                    if name:
                        self.ffi_registry[name] = {
                            "type": "rust_extern_c",
                            "function_name": name,
                            "rust_node": node_id,
                            "abi": "C",
                            "source_location": data.get("source_location"),
                        }
            elif lang in ("c", "cpp"):
                if ntype == "linkage_specification" and '"C"' in text:
                    for name in re.findall(r"\b(\w+)\s*\(", text):
                        self.ffi_registry.setdefault(
                            name,
                            {
                                "type": "c_abi",
                                "function_name": name,
                                "c_node": node_id,
                                "abi": "C",
                                "source_location": data.get("source_location"),
                            },
                        )
                elif ntype == "declaration" and text.lstrip().startswith("extern") and "(" in text:
                    match = re.search(r"\b(\w+)\s*\(", text)
                    if match:
                        name = match.group(1)
                        self.ffi_registry.setdefault(
                            name,
                            {
                                "type": "c_extern",
                                "function_name": name,
                                "c_node": node_id,
                                "abi": "C",
                                "source_location": data.get("source_location"),
                            },
                        )
            elif lang == "fortran":
                if ntype in ("function", "subroutine", "function_statement", "subroutine_statement"):
                    if re.search(r"bind\s*\(\s*c", text, re.IGNORECASE):
                        bind_match = re.search(
                            r'bind\s*\(\s*c\s*,\s*name\s*=\s*["\'](\w+)["\']',
                            text,
                            re.IGNORECASE,
                        )
                        name = meta.get("name") or ""
                        bind_name = bind_match.group(1) if bind_match else name
                        key = bind_name or name
                        if key:
                            self.ffi_registry[key] = {
                                "type": "fortran_c_abi",
                                "function_name": name,
                                "bind_name": bind_name,
                                "fortran_node": node_id,
                                "abi": "C",
                                "source_location": data.get("source_location"),
                            }

    def _detect_python_c_bindings(self) -> None:
        """Detect ``ctypes``/``cffi`` C bindings on the Python side."""
        for node_id, data in list(self.uast.nodes(data=True)):
            if data.get("language") != "python":
                continue
            if data.get("type") != "Call":
                continue
            src = data.get("data", {}).get("source", "") or ""
            if re.search(r"\b(CDLL|WinDLL|cdll\.LoadLibrary|windll\.LoadLibrary)\b", src):
                self.ffi_registry.setdefault(
                    f"__ctypes_{node_id}",
                    {"type": "python_ctypes", "loader": src[:120], "python_node": node_id},
                )
            if re.search(r"\bffi\.dlopen\b|\bcffi\.FFI\b|\bFFI\(\)", src):
                self.ffi_registry.setdefault(
                    f"__cffi_{node_id}",
                    {"type": "python_cffi", "loader": src[:120], "python_node": node_id},
                )

    _FFI_TARGET_KEYS = ("rust_node", "c_node", "fortran_node")
    _PYTHON_SIDE_FFI_TYPES = {"python_ctypes", "python_cffi"}

    def _create_cross_language_edges(self) -> None:
        python_calls: List[Tuple[str, Dict[str, Any]]] = []
        for node_id, data in self.uast.nodes(data=True):
            if data.get("language") == "python" and data.get("type") == "Call":
                python_calls.append((node_id, data))

        for node_id, data in python_calls:
            src = data.get("data", {}).get("source", "") or ""
            for ffi_name, ffi_info in self.ffi_registry.items():
                ftype = ffi_info.get("type")
                if ftype == "gpu_kernel" or ftype in self._PYTHON_SIDE_FFI_TYPES:
                    continue
                target = next(
                    (ffi_info[k] for k in self._FFI_TARGET_KEYS if ffi_info.get(k)),
                    None,
                )
                if not target or not self.uast.has_node(target):
                    continue
                candidate_names = [
                    ffi_name,
                    ffi_info.get("function_name"),
                    ffi_info.get("bind_name"),
                ]
                if any(name and name in src for name in candidate_names):
                    edge_type = "pyo3_ffi_bridge" if ftype == "pyo3_function" else "c_abi_bridge"
                    self.uast.add_edge(
                        node_id,
                        target,
                        edge_type=edge_type,
                        ffi_function=ffi_name,
                    )

    def _create_gpu_kernel_edges(self) -> None:
        """Link host call sites to GPU kernel nodes via ``gpu_kernel`` edges."""
        kernels = {
            name: info
            for name, info in self.ffi_registry.items()
            if info.get("type") == "gpu_kernel"
        }
        if not kernels:
            return
        for node_id, data in self.uast.nodes(data=True):
            if data.get("metadata", {}).get("uast_kind") != "uast_call":
                continue
            call_name = data.get("metadata", {}).get("name", "")
            payload = data.get("data", {})
            text = payload.get("source") or payload.get("text") or ""
            for kernel_name, kernel_info in kernels.items():
                launches = bool(re.search(rf"\b{re.escape(kernel_name)}\s*<<<", text))
                if call_name == kernel_name or launches:
                    kernel_node = kernel_info.get("kernel_node")
                    if kernel_node and self.uast.has_node(kernel_node):
                        self.uast.add_edge(
                            node_id,
                            kernel_node,
                            edge_type="gpu_kernel",
                            kernel=kernel_name,
                        )

    # ------------------------------------------------------------------
    # Data-flow analysis
    # ------------------------------------------------------------------

    def _run_data_flow_analysis(self) -> None:
        definitions: Dict[str, str] = {}
        uses: Dict[str, List[str]] = {}

        for node_id, data in self.uast.nodes(data=True):
            if data.get("language") != "python":
                continue
            ntype = data.get("type", "")
            src = data.get("data", {}).get("source", "") or ""

            if ntype == "Assign":
                match = re.match(r"^(\w+)\s*=", src)
                if match:
                    definitions[match.group(1)] = node_id

            if ntype == "Name":
                var_name = src.strip()
                if var_name:
                    uses.setdefault(var_name, []).append(node_id)

        for var_name, use_nodes in uses.items():
            if var_name in definitions:
                def_node = definitions[var_name]
                for use_node in use_nodes:
                    if use_node != def_node:
                        self.uast.add_edge(
                            def_node,
                            use_node,
                            edge_type="data_flow",
                            variable=var_name,
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _glob_safe(root: Path, pattern: str) -> List[Path]:
        if not root.exists():
            return []
        return sorted(root.glob(pattern))

    @staticmethod
    def _generate_node_id(file_path: Path, node: Any) -> str:
        lineno = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        name = getattr(node, "name", None) or getattr(node, "id", None) or node.__class__.__name__
        raw = f"{file_path}:{lineno}:{col}:{name}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _generate_rust_node_id(file_path: Path, node: Any) -> str:
        point = _node_start_point(node)
        raw = f"{file_path}:{point[0]}:{point[1]}:{_node_type(node)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
