"""Tests for the Semantic Proximity Mapping Engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis.semantic_mapper import SemanticMapper, UnifiedASTNode

_CONFIG = {
    "analysis": {
        "semantic_proximity_mapping": {
            "ffi_type": "PyO3",
            "source_roots": {
                "python": "src/python",
                "rust": "src/native",
            },
            "uast_generation": {
                "language_parsers": {"python": "native_ast", "rust": "tree_sitter"},
            },
        }
    }
}


class TestUnifiedASTNode(unittest.TestCase):
    def test_to_dict(self):
        node = UnifiedASTNode(
            node_id="abc123",
            node_type="FunctionDef",
            language="python",
            source_location=("test.py", 10, 0),
            data={"source": "def foo(): pass"},
        )
        d = node.to_dict()
        self.assertEqual(d["id"], "abc123")
        self.assertEqual(d["type"], "FunctionDef")
        self.assertEqual(d["language"], "python")


class TestSemanticMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = SemanticMapper(_CONFIG)

    def test_build_uast_empty_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            uast = self.mapper.build_uast(Path(tmp))
            self.assertEqual(uast.number_of_nodes(), 0)

    def test_build_uast_with_python_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brains = root / "builder_brains"
            brains.mkdir()
            (brains / "example.py").write_text(
                "def hello():\n    return 42\n\nx = hello()\n"
            )
            uast = self.mapper.build_uast(root)
            self.assertGreater(uast.number_of_nodes(), 0)

    def test_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brains = root / "builder_brains"
            brains.mkdir()
            (brains / "tiny.py").write_text("a = 1\n")
            self.mapper.build_uast(root)
            stats = self.mapper.get_statistics()
            self.assertIn("total_nodes", stats)
            self.assertIn("python_nodes", stats)
            self.assertGreater(stats["python_nodes"], 0)

    def test_export_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brains = root / "builder_brains"
            brains.mkdir()
            (brains / "tiny.py").write_text("a = 1\n")
            self.mapper.build_uast(root)
            export_path = root / "uast.json"
            self.mapper.export_graph(export_path)
            self.assertTrue(export_path.exists())
            import json
            data = json.loads(export_path.read_text())
            self.assertIn("nodes", data)
            self.assertIn("edges", data)

    def test_data_flow_edges_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brains = root / "builder_brains"
            brains.mkdir()
            (brains / "flow.py").write_text("x = 10\ny = x + 1\n")
            self.mapper.build_uast(root)
            dataflow = [
                (s, t, d)
                for s, t, d in self.mapper.uast.edges(data=True)
                if d.get("edge_type") == "data_flow"
            ]
            self.assertGreater(len(dataflow), 0)


if __name__ == "__main__":
    unittest.main()
