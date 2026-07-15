# -*- coding: utf-8 -*-
"""Tests that ``ScaffoldEngine`` auto-detects and bundles ``.part2.aeroc`` partitions."""

import tempfile
import unittest
from pathlib import Path

from src.scaffold.engine import ScaffoldEngine


class TestScaffoldAerocArtifacts(unittest.TestCase):
    def test_aeroc_entry_bundles_primary_and_part2(self):
        """A .aeroc source entry is packaged with its .part2.aeroc sibling."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "module.aeroc"
            part2 = root / "module.part2.aeroc"
            primary.write_text('{"primary": true}', encoding="utf-8")
            part2.write_text('{"secondary": true}', encoding="utf-8")

            out = root / "dist"
            engine = ScaffoldEngine(verbose=False)
            result = engine.scaffold(str(primary), distribution_directory=out, keep=True)

            self.assertEqual(result.language, "aeroc")
            self.assertIn("build_artifacts/module.aeroc", result.repo["files"])
            self.assertIn("build_artifacts/module.part2.aeroc", result.repo["files"])
            self.assertIn("build_artifacts/module.aeroc", result.repo["aeroc_artifacts"])
            self.assertIn("build_artifacts/module.part2.aeroc", result.repo["aeroc_artifacts"])
            self.assertTrue((out / "build_artifacts" / "module.aeroc").is_file())
            self.assertTrue((out / "build_artifacts" / "module.part2.aeroc").is_file())

    def test_python_source_collects_aeroc_siblings(self):
        """A Python source is scaffolded and any compiled .aeroc / .part2 siblings are copied."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "foo.py"
            src.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
            primary = root / "foo.aeroc"
            part2 = root / "foo.part2.aeroc"
            primary.write_text('{"primary": true}', encoding="utf-8")
            part2.write_text('{"secondary": true}', encoding="utf-8")

            out = root / "dist"
            engine = ScaffoldEngine(verbose=False)
            result = engine.scaffold(str(src), distribution_directory=out, keep=True)

            self.assertEqual(result.language, "python")
            files = result.repo["files"]
            self.assertIn("build_artifacts/foo.aeroc", files)
            self.assertIn("build_artifacts/foo.part2.aeroc", files)
            self.assertTrue((out / "build_artifacts" / "foo.aeroc").is_file())
            self.assertTrue((out / "build_artifacts" / "foo.part2.aeroc").is_file())

    def test_missing_part2_is_not_fatal(self):
        """If a .part2.aeroc sibling is missing, the primary .aeroc is still packaged."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "module.aeroc"
            primary.write_text('{"primary": true}', encoding="utf-8")

            out = root / "dist"
            engine = ScaffoldEngine(verbose=False)
            result = engine.scaffold(str(primary), distribution_directory=out, keep=True)

            self.assertEqual(result.repo["files"], ["build_artifacts/module.aeroc"])
            self.assertTrue((out / "build_artifacts" / "module.aeroc").is_file())
            self.assertFalse((out / "build_artifacts" / "module.part2.aeroc").exists())


if __name__ == "__main__":
    unittest.main()
