"""Tests for context ingestion (analyser, repair, ingestor)."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from src.context.analyser import CodeAnalyser
from src.context.ingest import ContextIngestor
from src.context.repair import CodeRepairer


class TestCodeAnalyser(unittest.TestCase):
    def setUp(self):
        self.analyser = CodeAnalyser()

    def test_undefined_and_unused(self):
        src = "import os\nimport sys\n\ndef f(x):\n    return math.sin(x)\n"
        findings = self.analyser.analyse(src, "m.py", "python")
        self.assertIn("math", findings.undefined_names)
        self.assertIn("os", findings.unused_imports)
        self.assertIn("sys", findings.unused_imports)

    def test_missing_return_type(self):
        src = "def f(x):\n    return x\n\ndef g():\n    pass\n"
        findings = self.analyser.analyse(src, "m.py", "python")
        self.assertIn("f", findings.functions_missing_return_type)
        self.assertIn("g", findings.functions_missing_return_type)

    def test_syntax_error_recorded(self):
        findings = self.analyser.analyse("def (:\n", "bad.py", "python")
        self.assertIsNotNone(findings.syntax_error)


class TestCodeRepairer(unittest.TestCase):
    def setUp(self):
        self.repairer = CodeRepairer()

    def test_auto_import(self):
        src = "def f(x):\n    return math.sqrt(x)\n"
        result = self.repairer.repair(src, ["auto_import"], "m.py")
        self.assertIn("import math", result.source)
        ast.parse(result.source)  # still valid

    def test_remove_unused(self):
        src = "import os\nimport math\n\ndef f(x):\n    return math.sqrt(x)\n"
        result = self.repairer.repair(src, ["remove_unused"], "m.py")
        self.assertNotIn("import os", result.source)
        self.assertIn("import math", result.source)

    def test_type_inference_adds_none(self):
        src = "def g():\n    pass\n"
        result = self.repairer.repair(src, ["type_inference"], "m.py")
        self.assertIn("-> None", result.source)

    def test_type_inference_skips_value_returns(self):
        src = "def g():\n    return 5\n"
        result = self.repairer.repair(src, ["type_inference"], "m.py")
        self.assertNotIn("-> None", result.source)

    def test_broken_repair_rolls_back(self):
        # Unparseable input is never modified.
        src = "def (:\n"
        result = self.repairer.repair(src, ["auto_import", "remove_unused"], "m.py")
        self.assertEqual(result.source, src)
        self.assertEqual(result.changes, [])


class TestContextIngestor(unittest.TestCase):
    def _project(self, tmp: Path):
        ext = tmp / "ext"
        ext.mkdir()
        (ext / "core.py").write_text("def f(a):\n    return math.sin(a)\n")
        (ext / "util.py").write_text("import json\nimport os\nDATA = os.getcwd()\n")
        ws = tmp / "ws"
        ws.mkdir()
        return ext, ws

    def test_ingest_copies_and_repairs(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            ext, ws = self._project(tmp)
            config = {"context": {"sources": [{
                "path": str(ext), "language": "python",
                "repair_rules": ["auto_import", "remove_unused"],
                "target_mapping": "src/python/shbt",
            }]}}
            report = ContextIngestor(config, ws).ingest_all()
            self.assertEqual(report["files_ingested"], 2)
            core = (ws / "src/python/shbt/core.py").read_text()
            self.assertIn("import math", core)
            util = (ws / "src/python/shbt/util.py").read_text()
            self.assertNotIn("import json", util)  # unused removed
            self.assertIn("os.getcwd", util)       # used import kept
            self.assertTrue((ws / ContextIngestor.REPORT_NAME).exists())

    def test_missing_source_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as t:
            ws = Path(t) / "ws"
            ws.mkdir()
            config = {"context": {"sources": [{"path": str(Path(t) / "nope"), "language": "python", "target_mapping": "x"}]}}
            report = ContextIngestor(config, ws).ingest_all()
            self.assertEqual(len(report["errors"]), 1)
            self.assertIn("not found", report["errors"][0])

    def test_report_is_valid_json(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            ext, ws = self._project(tmp)
            config = {"context": [{"path": str(ext), "language": "python", "target_mapping": "imp"}]}
            ContextIngestor(config, ws).ingest_all()
            data = json.loads((ws / ContextIngestor.REPORT_NAME).read_text())
            self.assertIn("sources", data)
            self.assertEqual(data["source_count"], 1)

    def test_symlink_mode(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            ext, ws = self._project(tmp)
            config = {"context": {"sources": [{
                "path": str(ext), "language": "python", "target_mapping": "linked", "mode": "symlink",
            }]}}
            ContextIngestor(config, ws).ingest_all()
            # No repair rules -> symlinked.
            linked = ws / "linked" / "core.py"
            self.assertTrue(linked.exists())


if __name__ == "__main__":
    unittest.main()
