# -*- coding: utf-8 -*-
"""Unit tests for builder_brains.compactor."""

import ast
import unittest

from builder_brains.compactor import (
    StructuralTokenizer,
    tokenize_source,
    DeadCodeEliminator,
    _collect_loaded_names,
    eliminate_dead_code,
)


class TestStructuralTokenizer(unittest.TestCase):
    def test_basic_tokenization(self):
        source = "x = 1\ny = 2\n"
        tree = ast.parse(source)
        tok = StructuralTokenizer()
        tok.visit(tree)
        vec = tok.get_token_vector()
        self.assertIn("Module", vec)
        self.assertIn("Assign", vec)
        self.assertEqual(vec["Assign"], 2)

    def test_depth_stats(self):
        source = "def foo():\n    return 1\n"
        tree = ast.parse(source)
        tok = StructuralTokenizer()
        tok.visit(tree)
        stats = tok.get_depth_stats()
        self.assertIn("max_observed_depth", stats)
        self.assertIn("total_nodes", stats)
        self.assertIn("unique_node_types", stats)
        self.assertGreater(stats["total_nodes"], 0)
        self.assertGreater(stats["max_observed_depth"], 0)

    def test_max_depth_limit(self):
        # Build deeply nested code
        source = "x = " + "[" * 50 + "1" + "]" * 50
        tree = ast.parse(source)
        tok = StructuralTokenizer(max_depth=10)
        tok.visit(tree)
        stats = tok.get_depth_stats()
        # Should have visited but stopped at max_depth
        self.assertGreater(stats["total_nodes"], 0)

    def test_function_scope(self):
        source = "def foo():\n    pass\ndef bar():\n    pass\n"
        tree = ast.parse(source)
        tok = StructuralTokenizer()
        tok.visit(tree)
        vec = tok.get_token_vector()
        self.assertEqual(vec["FunctionDef"], 2)

    def test_class_scope(self):
        source = "class Foo:\n    pass\n"
        tree = ast.parse(source)
        tok = StructuralTokenizer()
        tok.visit(tree)
        vec = tok.get_token_vector()
        self.assertEqual(vec["ClassDef"], 1)


class TestTokenizeSource(unittest.TestCase):
    def test_returns_expected_keys(self):
        result = tokenize_source("x = 1")
        self.assertIn("token_vector", result)
        self.assertIn("depth_stats", result)

    def test_complex_source(self):
        source = """
import os
class MyClass:
    def method(self):
        if True:
            return os.getcwd()
        else:
            raise ValueError()
"""
        result = tokenize_source(source)
        self.assertIn("ClassDef", result["token_vector"])
        self.assertIn("FunctionDef", result["token_vector"])
        self.assertIn("Import", result["token_vector"])


class TestCollectLoadedNames(unittest.TestCase):
    def test_collects_used_names(self):
        source = "x = 1\ny = x + 2\nprint(y)\n"
        tree = ast.parse(source)
        names = _collect_loaded_names(tree)
        self.assertIn("x", names)
        self.assertIn("y", names)
        self.assertIn("print", names)


class TestDeadCodeEliminator(unittest.TestCase):
    def test_removes_after_return(self):
        source = "def foo():\n    return 1\n    x = 2\n    y = 3\n"
        tree = ast.parse(source)
        elim = DeadCodeEliminator()
        elim.set_loaded_names(_collect_loaded_names(tree))
        new_tree = elim.visit(tree)
        code = ast.unparse(new_tree)
        self.assertIn("return 1", code)
        self.assertNotIn("x = 2", code)
        self.assertGreater(elim.removed_nodes, 0)

    def test_removes_after_raise(self):
        source = "def bar():\n    raise Exception()\n    cleanup()\n"
        tree = ast.parse(source)
        elim = DeadCodeEliminator()
        elim.set_loaded_names(_collect_loaded_names(tree))
        new_tree = elim.visit(tree)
        code = ast.unparse(new_tree)
        self.assertNotIn("cleanup", code)

    def test_removes_unused_imports(self):
        source = "import os\nimport sys\nprint(os.getcwd())\n"
        tree = ast.parse(source)
        loaded = _collect_loaded_names(tree)
        elim = DeadCodeEliminator()
        elim.set_loaded_names(loaded)
        new_tree = elim.visit(tree)
        code = ast.unparse(new_tree)
        self.assertIn("import os", code)
        # sys is unused
        self.assertNotIn("import sys", code)

    def test_preserves_used_code(self):
        source = "x = 1\ny = x + 2\nprint(y)\n"
        tree = ast.parse(source)
        loaded = _collect_loaded_names(tree)
        elim = DeadCodeEliminator()
        elim.set_loaded_names(loaded)
        new_tree = elim.visit(tree)
        code = ast.unparse(new_tree)
        self.assertIn("x = 1", code)
        self.assertIn("y = x + 2", code)

    def test_run_passes(self):
        source = "import os\nimport sys\nx = os.getcwd()\n_unused = 42\n"
        tree = ast.parse(source)
        elim = DeadCodeEliminator(elimination_depth=2)
        result_tree = elim.run_passes(tree)
        code = ast.unparse(result_tree)
        self.assertIn("import os", code)
        # _unused starts with _ and is not loaded
        self.assertNotIn("_unused", code)


class TestEliminateDeadCode(unittest.TestCase):
    def test_returns_expected_structure(self):
        source = "def foo():\n    return 1\n    x = 2\n"
        result = eliminate_dead_code(source)
        self.assertIn("cleaned_source", result)
        self.assertIn("removed_node_count", result)
        self.assertIn("original_node_count", result)
        self.assertIn("final_node_count", result)
        self.assertGreater(result["removed_node_count"], 0)


if __name__ == "__main__":
    unittest.main()
