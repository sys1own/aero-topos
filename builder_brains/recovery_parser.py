"""In-memory syntactic recovery for Tree-sitter parse trees.

AeroNova's compilation pipeline drops an entire file tree when Tree-sitter
encounters raw syntax errors, because the resulting ``ERROR`` / ``MISSING``
nodes block structural decomposition. This module performs a lookahead token
injection pass that patches those errors programmatically -- in memory -- so a
best-effort tree can be recovered before the formal parsing pass runs.

The recovery is intentionally conservative: it only injects tokens for a small
set of well-understood, language-specific failure shapes (missing block colons
in Python, missing terminators / unclosed scopes in Rust and C++). When no rule
matches, the original tree and source are returned untouched.
"""

import tree_sitter


class AeroTreeRecoveryParser:
    """Attempt programmatic recovery of Tree-sitter parses containing errors.

    The parser walks the error/missing nodes produced by Tree-sitter and applies
    language-specific patches to the source bytes, then reparses. All injection
    is done on an in-memory copy of the source; the caller's bytes are never
    mutated in place.
    """

    def __init__(
        self,
        language: tree_sitter.Language,
        parser: tree_sitter.Parser,
        language_name: str | None = None,
    ):
        self.language = language
        self.parser = parser
        # ``Language.name`` is ``None`` for many grammar bindings, so callers may
        # pass an explicit name to drive rule dispatch. Falls back to the
        # grammar's own name attribute when not provided.
        self.language_name = language_name
        self.max_lookahead_steps = 5
        self.syntax_rules = {
            "python": {
                "parent_types": [
                    "function_definition",
                    "class_definition",
                    "if_statement",
                    "for_statement",
                    "while_statement",
                ],
                "missing_token": b":",
                "default_indent_fill": b"\n    pass\n",
            },
            "rust": {
                "missing_terminator": b";",
                "missing_scope_close": b"}",
            },
            "cpp": {
                "missing_terminator": b";",
                "missing_scope_close": b"}",
            },
        }

    def attempt_recovery(
        self, raw_source: bytes
    ) -> tuple[tree_sitter.Tree, bytes, bool]:
        """Parse ``raw_source`` and try to repair any syntax errors in place.

        Returns a ``(tree, source, mutated)`` tuple where ``tree`` is the parse
        tree (reparsed when a patch was applied), ``source`` is the possibly
        modified source bytes, and ``mutated`` indicates whether any injection
        occurred. If the initial parse is error-free, the original tree and
        source are returned with ``mutated`` set to ``False``.
        """
        tree = self.parser.parse(raw_source)
        if not tree.root_node.has_error:
            return tree, raw_source, False

        modified_source = bytearray(raw_source)
        errors = self._gather_error_nodes(tree.root_node)
        # Patch from the end of the buffer backwards so each injection does not
        # shift the byte offsets of error nodes we have yet to process.
        errors.sort(key=lambda node: node.start_byte, reverse=True)
        mutated = False

        for err_node in errors:
            patch = self._calculate_recovery_patch(err_node, raw_source)
            if patch:
                start, end = err_node.start_byte, err_node.end_byte
                modified_source[start:end] = patch
                mutated = True

        if mutated:
            reparsed_bytes = bytes(modified_source)
            new_tree = self.parser.parse(reparsed_bytes)
            return new_tree, reparsed_bytes, True

        return tree, raw_source, False

    def _gather_error_nodes(
        self, node: tree_sitter.Node
    ) -> list[tree_sitter.Node]:
        """Recursively collect every ``ERROR`` or missing node in the subtree."""
        nodes = []
        if node.type == "ERROR" or node.is_missing:
            nodes.append(node)
        for child in node.children:
            nodes.extend(self._gather_error_nodes(child))
        return nodes

    def _calculate_recovery_patch(
        self, node: tree_sitter.Node, raw_source: bytes
    ) -> bytes | None:
        """Compute the injection bytes for a single error node, or ``None``.

        Dispatches on the configured language name and applies the matching
        language-specific recovery rule. Returns ``None`` when no rule applies.
        """
        lang_name = self.language_name
        if not lang_name and hasattr(self.language, "name"):
            lang_name = self.language.name
        if not lang_name:
            lang_name = "python"
        rules = self.syntax_rules.get(lang_name, {})
        if not rules:
            return None

        if lang_name == "python":
            parent = node.parent
            if parent and parent.type in rules["parent_types"]:
                # A block-introducing construct is missing its trailing colon and
                # body; inject the colon plus a default indented ``pass`` block.
                sibling_text = raw_source[parent.start_byte : node.start_byte]
                if b"(" in sibling_text and not sibling_text.rstrip().endswith(b":"):
                    return rules["missing_token"] + rules["default_indent_fill"]
            # A bare newline with an empty statement body -- fill in a body.
            if node.type == "ERROR" and raw_source[node.start_byte - 1 : node.start_byte] == b"\n":
                return rules["default_indent_fill"]

        elif lang_name in ("rust", "cpp"):
            if node.is_missing and node.type == ";":
                return rules["missing_terminator"]
            if node.type == "ERROR":
                # Peek ahead a short window; an opening brace with no matching
                # close within the window signals an unclosed scope.
                context_window = raw_source[node.start_byte : node.start_byte + 32]
                if b"{" in context_window and b"}" not in context_window:
                    return rules["missing_scope_close"]
        return None
