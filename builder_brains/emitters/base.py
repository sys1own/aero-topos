# -*- coding: utf-8 -*-
"""Generic code-generation interface for polyglot UAST emission.

The :class:`BaseEmitter` accepts a linearized UAST node list (as produced by
:mod:`core.parser.universal`) together with an optional scope graph dictionary,
and translates the language-agnostic representation back into target-language
source code without losing functions, parameters, or structural scopes.

Subclasses override the per-node-kind emit hooks to produce language-specific
syntax (indentation, braces vs. colons, type annotations, etc.).
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EmitterError(Exception):
    """Raised when code emission encounters an unrecoverable structural issue."""


class BaseEmitter(ABC):
    """Abstract base for target-language code emitters.

    Parameters
    ----------
    nodes : list[dict]
        The flat 1-D UAST node array (each element carries ``id``, ``parent``,
        ``children``, ``type``, ``category``, ``canonical_kind``, ``text``, etc.).
    scope_graph : dict
        A mapping from node IDs to scope metadata (variable bindings, symbol
        visibility, nesting depth).  May be empty for simple translations.
    source_language : str
        The language from which the UAST was originally parsed (informational).
    """

    # Subclasses MUST set this to their canonical language name.
    target_language: str = ""

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        scope_graph: Optional[Dict[str, Any]] = None,
        source_language: str = "",
    ) -> None:
        self.nodes = nodes
        self.scope_graph = scope_graph or {}
        self.source_language = source_language
        self._output_lines: List[str] = []
        self._indent_level: int = 0
        self._symbols: Dict[str, str] = {}  # node_id -> emitted name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(self) -> str:
        """Walk the UAST and return the fully rendered target-language source."""
        self._output_lines = []
        self._indent_level = 0
        self._symbols = {}

        self._emit_preamble()

        root_children = self._root_children()
        for node_id in root_children:
            self._emit_node(node_id)

        self._emit_postamble()
        return "\n".join(self._output_lines) + "\n"

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _emit_preamble(self) -> None:
        """Emit file-level header (imports, pragma, etc.)."""

    def _emit_postamble(self) -> None:
        """Emit file-level footer if needed."""

    @abstractmethod
    def _emit_function_declaration(self, node: Dict[str, Any]) -> None:
        """Emit a complete function declaration from the UAST node."""

    @abstractmethod
    def _emit_type_declaration(self, node: Dict[str, Any]) -> None:
        """Emit a type/class/struct declaration from the UAST node."""

    @abstractmethod
    def _emit_import_declaration(self, node: Dict[str, Any]) -> None:
        """Emit an import/use/include statement."""

    @abstractmethod
    def _emit_statement(self, node: Dict[str, Any]) -> None:
        """Emit a generic statement."""

    @abstractmethod
    def _emit_expression(self, node: Dict[str, Any]) -> None:
        """Emit an expression."""

    def _emit_variable_declaration(self, node: Dict[str, Any]) -> None:
        """Emit a variable/let binding. Default delegates to _emit_statement."""
        self._emit_statement(node)

    def _emit_module_declaration(self, node: Dict[str, Any]) -> None:
        """Emit a module/namespace block. Default delegates to _emit_statement."""
        self._emit_statement(node)

    # ------------------------------------------------------------------
    # Tree traversal helpers
    # ------------------------------------------------------------------

    def _emit_node(self, node_id: int) -> None:
        """Dispatch a single node to the appropriate emit hook."""
        node = self.nodes[node_id]
        canonical = node.get("canonical_kind") or ""
        category = node.get("category", "")

        if canonical == "function_declaration":
            self._emit_function_declaration(node)
        elif canonical == "type_declaration":
            self._emit_type_declaration(node)
        elif canonical == "import_declaration":
            self._emit_import_declaration(node)
        elif canonical == "module_declaration":
            self._emit_module_declaration(node)
        elif category == "DECLARATION":
            self._emit_variable_declaration(node)
        elif category == "STATEMENT":
            self._emit_statement(node)
        elif category == "EXPRESSION":
            self._emit_expression(node)
        else:
            # Recurse into children for structural containers.
            for child_id in node.get("children", []):
                self._emit_node(child_id)

    def _root_children(self) -> List[int]:
        """Return the IDs of root-level children (translation_unit descendants)."""
        if not self.nodes:
            return []
        root = self.nodes[0]
        return root.get("children", [])

    def _get_node(self, node_id: int) -> Dict[str, Any]:
        return self.nodes[node_id]

    def _child_nodes(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.nodes[cid] for cid in node.get("children", [])]

    def _find_child_by_type(
        self, node: Dict[str, Any], node_type: str
    ) -> Optional[Dict[str, Any]]:
        for cid in node.get("children", []):
            child = self.nodes[cid]
            if child["type"] == node_type:
                return child
        return None

    def _find_children_by_type(
        self, node: Dict[str, Any], node_type: str
    ) -> List[Dict[str, Any]]:
        return [
            self.nodes[cid]
            for cid in node.get("children", [])
            if self.nodes[cid]["type"] == node_type
        ]

    def _find_child_by_category(
        self, node: Dict[str, Any], category: str
    ) -> Optional[Dict[str, Any]]:
        for cid in node.get("children", []):
            child = self.nodes[cid]
            if child.get("category") == category:
                return child
        return None

    def _node_text(self, node: Dict[str, Any]) -> str:
        """Reconstruct source text for a leaf node or subtree."""
        if "text" in node:
            return node["text"]
        parts: List[str] = []
        for cid in node.get("children", []):
            parts.append(self._node_text(self.nodes[cid]))
        return " ".join(parts)

    def _extract_name(self, node: Dict[str, Any]) -> str:
        """Extract the identifier name from a declaration node."""
        for cid in node.get("children", []):
            child = self.nodes[cid]
            if child["type"] in ("identifier", "type_identifier", "name", "field_identifier"):
                return child.get("text", "")
        return "unknown"

    def _extract_parameters(self, node: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract (name, type_hint) pairs from a function's parameter list."""
        params: List[Tuple[str, str]] = []
        param_list = self._find_child_by_type(node, "parameters") or \
                     self._find_child_by_type(node, "parameter_list") or \
                     self._find_child_by_type(node, "formal_parameters")
        if param_list is None:
            return params

        for cid in param_list.get("children", []):
            child = self.nodes[cid]
            if not child.get("named", False):
                continue
            if child["type"] in ("parameter", "typed_parameter", "identifier",
                                 "simple_parameter", "default_parameter"):
                name = self._extract_name(child) or child.get("text", "")
                type_hint = self._extract_type_annotation(child)
                if name and name not in ("(", ")", ","):
                    params.append((name, type_hint))
        return params

    def _extract_type_annotation(self, node: Dict[str, Any]) -> str:
        """Extract type annotation string from a parameter or declaration node."""
        type_node = self._find_child_by_type(node, "type") or \
                    self._find_child_by_type(node, "type_identifier") or \
                    self._find_child_by_type(node, "primitive_type") or \
                    self._find_child_by_type(node, "type_annotation")
        if type_node:
            return self._node_text(type_node)
        return ""

    def _extract_return_type(self, node: Dict[str, Any]) -> str:
        """Extract the return type annotation from a function declaration."""
        ret = self._find_child_by_type(node, "return_type") or \
              self._find_child_by_type(node, "type")
        if ret:
            return self._node_text(ret)
        return ""

    def _extract_body_statements(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract child statements from a function/block body."""
        body = self._find_child_by_type(node, "block") or \
               self._find_child_by_type(node, "compound_statement") or \
               self._find_child_by_type(node, "function_body")
        if body is None:
            return self._child_nodes(node)
        return self._child_nodes(body)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _write(self, line: str) -> None:
        """Append an indented line to the output."""
        prefix = "    " * self._indent_level
        self._output_lines.append(prefix + line)

    def _write_raw(self, line: str) -> None:
        """Append a line without auto-indentation."""
        self._output_lines.append(line)

    def _blank(self) -> None:
        """Append a blank line."""
        self._output_lines.append("")

    def _indent(self) -> None:
        self._indent_level += 1

    def _dedent(self) -> None:
        self._indent_level = max(0, self._indent_level - 1)


# ---------------------------------------------------------------------------
# Emitter registry
# ---------------------------------------------------------------------------

class EmitterRegistry:
    """Central registry mapping language names to emitter classes."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, language: str, emitter_cls: type) -> None:
        cls._registry[language.lower()] = emitter_cls

    @classmethod
    def get(cls, language: str) -> Optional[type]:
        return cls._registry.get(language.lower())

    @classmethod
    def supported_languages(cls) -> List[str]:
        return sorted(cls._registry.keys())


def get_emitter(
    language: str,
    nodes: List[Dict[str, Any]],
    scope_graph: Optional[Dict[str, Any]] = None,
    source_language: str = "",
) -> BaseEmitter:
    """Instantiate the appropriate emitter for *language*.

    Raises :class:`EmitterError` if no emitter is registered for the language.
    """
    emitter_cls = EmitterRegistry.get(language)
    if emitter_cls is None:
        raise EmitterError(
            f"No emitter registered for target language {language!r}. "
            f"Supported: {EmitterRegistry.supported_languages()}"
        )
    return emitter_cls(nodes, scope_graph=scope_graph, source_language=source_language)
