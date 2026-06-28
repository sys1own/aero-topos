# -*- coding: utf-8 -*-
"""Tests for the language-agnostic Tree-sitter engine (``core/parser/universal``)."""

import unittest

from core.parser.universal import (
    MAX_FILE_BYTES,
    NodeCategory,
    UniversalParseError,
    _fallback_uast,
    _flatten_fallback_tree,
    detect_language,
    extract_symbols,
    parse_source,
    semantic_hash,
    supported_languages,
)


def _grammar_available(language: str) -> bool:
    try:
        parse_source("", language)
        return True
    except UniversalParseError:
        return False


_PY = 'def add(a, b):\n    return a + b  # sum\n\nclass Point:\n    pass\n'
_RS = '// header\nstruct Point { x: i32 }\n\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n'
_CPP = 'class Foo {\npublic:\n    int bar() { return 1; }\n};\n\nint baz() { return 0; }\n'


class TestGrammarLoading(unittest.TestCase):
    def test_detect_language_by_extension(self):
        self.assertEqual(detect_language("a.py"), "python")
        self.assertEqual(detect_language("a.rs"), "rust")
        self.assertEqual(detect_language("a.cpp"), "cpp")
        self.assertEqual(detect_language("a.f90"), "fortran")
        self.assertEqual(detect_language("a.cbl"), "cobol")
        self.assertIsNone(detect_language("a.unknownext"))

    def test_supported_languages_include_core_set(self):
        langs = set(supported_languages())
        self.assertTrue({"python", "rust", "cpp", "c", "fortran"}.issubset(langs))

    def test_unknown_grammar_raises(self):
        with self.assertRaises(UniversalParseError):
            parse_source("x", "klingon")


class TestUASTSchema(unittest.TestCase):
    def setUp(self):
        self.uast = parse_source(_PY, "python", file_path="m.py")

    def test_metadata_block(self):
        md = self.uast["metadata"]
        self.assertEqual(md["language"], "python")
        self.assertEqual(md["file_path"], "m.py")
        self.assertEqual(len(md["file_hash"]), 64)
        self.assertEqual(md["source_bytes"], len(_PY.encode("utf-8")))
        self.assertEqual(md["node_count"], len(self.uast["nodes"]))
        self.assertIn("parser_flags", md)

    def test_flat_node_array_pointers(self):
        nodes = self.uast["nodes"]
        self.assertEqual(self.uast["root"], 0)
        self.assertIsNone(nodes[0]["parent"])
        # Every node has a unique id equal to its index, valid spans, a type.
        for i, node in enumerate(nodes):
            self.assertEqual(node["id"], i)
            self.assertLessEqual(node["start_byte"], node["end_byte"])
            self.assertIsInstance(node["type"], str)
            # parent/children pointers are consistent
            for child_id in node["children"]:
                self.assertEqual(nodes[child_id]["parent"], i)

    def test_taxonomy_categories_present(self):
        cats = {n["category"] for n in self.uast["nodes"]}
        self.assertIn(NodeCategory.DECLARATION.value, cats)
        self.assertIn(NodeCategory.STATEMENT.value, cats)
        self.assertIn(NodeCategory.COMMENT.value, cats)

    def test_leaf_tokens_carry_text(self):
        # At least the identifier 'add' should appear as a leaf token text.
        leaf_texts = [n.get("text") for n in self.uast["nodes"] if not n["children"]]
        self.assertIn("add", leaf_texts)


class TestCrossLanguageAlignment(unittest.TestCase):
    def _kinds(self, source, language):
        uast = parse_source(source, language)
        return {n.get("canonical_kind") for n in uast["nodes"]}

    def test_function_declaration_aligns_across_languages(self):
        self.assertIn("function_declaration", self._kinds(_PY, "python"))
        if _grammar_available("rust"):
            self.assertIn("function_declaration", self._kinds(_RS, "rust"))
        if _grammar_available("cpp"):
            self.assertIn("function_declaration", self._kinds(_CPP, "cpp"))

    def test_type_declaration_aligns(self):
        self.assertIn("type_declaration", self._kinds(_PY, "python"))
        if _grammar_available("rust"):
            self.assertIn("type_declaration", self._kinds(_RS, "rust"))


class TestSemanticHashing(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(semantic_hash(_PY, "python"), semantic_hash(_PY, "python"))
        self.assertEqual(len(semantic_hash(_PY, "python")), 64)

    def test_comment_and_whitespace_invariant_python(self):
        noisy = 'def add(a, b):\n    # totally different\n    return a + b\n\n\nclass Point:\n\n    pass\n'
        self.assertEqual(semantic_hash(_PY, "python"), semantic_hash(noisy, "python"))

    @unittest.skipUnless(_grammar_available("rust"), "rust grammar missing")
    def test_comment_invariant_rust(self):
        noisy = 'struct Point { x: i32 }\n/* block */\npub fn add(a: i32, b: i32) -> i32 {\n    // inline\n    a + b\n}\n'
        self.assertEqual(semantic_hash(_RS, "rust"), semantic_hash(noisy, "rust"))

    def test_structural_change_changes_hash(self):
        changed = _PY + "\ndef sub(a, b):\n    return a - b\n"
        self.assertNotEqual(semantic_hash(_PY, "python"), semantic_hash(changed, "python"))


class TestSymbolExtraction(unittest.TestCase):
    def test_python_symbols(self):
        uast = parse_source(_PY, "python")
        syms = extract_symbols(uast, _PY)
        self.assertIn("add", syms["functions"])
        self.assertIn("Point", syms["types"])

    @unittest.skipUnless(_grammar_available("rust"), "rust grammar missing")
    def test_rust_symbols(self):
        uast = parse_source(_RS, "rust")
        syms = extract_symbols(uast, _RS)
        self.assertIn("add", syms["functions"])
        self.assertIn("Point", syms["types"])


class TestEdgeCases(unittest.TestCase):
    def test_error_and_missing_nodes_become_placeholders(self):
        uast = parse_source("def f(:\n", "python")
        flags = uast["metadata"]["parser_flags"]
        self.assertTrue(flags["has_error"])
        self.assertGreaterEqual(flags["error_nodes"] + flags["missing_nodes"], 1)
        self.assertTrue(flags["error_locations"])
        # The rest of the tree is preserved (placeholders carry offsets).
        placeholders = [
            n for n in uast["nodes"]
            if n["canonical_kind"] in ("error_placeholder", "missing_placeholder")
        ]
        self.assertTrue(placeholders)
        for p in placeholders:
            self.assertIn("start_point", p)

    def test_large_file_is_bypassed(self):
        big = "x = 1\n" * (MAX_FILE_BYTES // 6 + 10)
        uast = parse_source(big, "python")
        flags = uast["metadata"]["parser_flags"]
        self.assertTrue(flags["fallback"])
        self.assertEqual(flags["fallback_reason"], "file_too_large")
        self.assertEqual(uast["nodes"], [])

    def test_valid_file_not_flagged(self):
        uast = parse_source(_PY, "python")
        flags = uast["metadata"]["parser_flags"]
        self.assertFalse(flags["fallback"])
        self.assertFalse(flags["has_error"])


class TestFallbackFlattening(unittest.TestCase):
    def test_python_fallback_tree_is_flattened(self):
        uast = _fallback_uast(
            _PY.encode("utf-8"), "python", "m.py",
            {"fallback": False, "fallback_reason": None, "has_error": False,
             "error_nodes": 0, "missing_nodes": 0, "error_locations": [],
             "truncated": False, "parse_seconds": 0.0},
            "forced_fallback",
        )
        nodes = uast["nodes"]
        self.assertGreater(len(nodes), 0)
        self.assertEqual(uast["root"], 0)
        self.assertEqual(nodes[0]["parent"], None)
        for i, node in enumerate(nodes):
            self.assertEqual(node["id"], i)
            for child_id in node["children"]:
                self.assertEqual(nodes[child_id]["parent"], i)
        self.assertEqual(uast["metadata"]["node_count"], len(nodes))
        self.assertTrue(uast["metadata"]["parser_flags"]["fallback"])

    def test_lexical_fallback_tree_is_flattened(self):
        rust_source = "fn main() { let x = 42; }\n"
        uast = _fallback_uast(
            rust_source.encode("utf-8"), "rust", "main.rs",
            {"fallback": False, "fallback_reason": None, "has_error": False,
             "error_nodes": 0, "missing_nodes": 0, "error_locations": [],
             "truncated": False, "parse_seconds": 0.0},
            "forced_fallback",
        )
        nodes = uast["nodes"]
        self.assertGreater(len(nodes), 0)
        self.assertEqual(uast["root"], 0)
        # Leaf tokens carry text and parent pointers are consistent.
        for i, node in enumerate(nodes):
            self.assertEqual(node["id"], i)
            for child_id in node["children"]:
                self.assertEqual(nodes[child_id]["parent"], i)
        leaf_texts = [n.get("text") for n in nodes if not n["children"]]
        self.assertIn("main", leaf_texts)

    def test_flattener_handles_missing_fields_gracefully(self):
        tree = {
            "type": "root",
            "children": [
                {"type": "child", "children": []},
                {"type": "leaf", "text": "x"},
            ],
        }
        nodes = _flatten_fallback_tree(tree, "python")
        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes[0]["children"], [1, 2])
        self.assertEqual(nodes[1]["parent"], 0)
        self.assertEqual(nodes[2]["parent"], 0)
        self.assertEqual(nodes[2].get("text"), "x")


if __name__ == "__main__":
    unittest.main()
