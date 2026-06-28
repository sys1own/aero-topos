# -*- coding: utf-8 -*-
"""Tests for the Language Router and Python scaffold path."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main
from blueprint_parser import normalize_optional_sections, parse_blueprint_content
from src.scaffold.engine import ScaffoldEngine
from src.scaffold.language_router import resolve_target_language
from src.scaffold.pipeline import ScaffoldBuildPipeline
from src.scaffold.python_repo_generator import infer_import_dependencies, generate_python_repo, build_python_spec
from src.scaffold.python_validator import PythonValidationRunner
from src.scaffold.source_resolver import resolve_source_entry
from src.scaffold.workspace import TOOL_ROOT

PYTHON_SOURCE = """\
#!/usr/bin/env python3
import json
import sys

def main() -> int:
    print(json.dumps({"status": "ok"}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""

BROKEN_PYTHON = "def broken(\n    return 1\n"


class TestLanguageRouter(unittest.TestCase):
    def test_reads_frameworks_language(self):
        ctx = {"frameworks": {"language": "python"}}
        self.assertEqual(
            resolve_target_language(ctx, source_path=Path("main.py")),
            "python",
        )

    def test_frameworks_language_overrides_rs_extension(self):
        ctx = {"frameworks": {"language": "python"}}
        self.tmp = Path(tempfile.mkdtemp())
        f = self.tmp / "main.py"
        f.write_text(PYTHON_SOURCE, encoding="utf-8")
        entry = resolve_source_entry(str(f))
        self.assertEqual(resolve_target_language(ctx, source_entry=entry), "python")

    def test_defaults_to_rust_without_declaration(self):
        ctx = {"frameworks": {}}
        self.assertEqual(resolve_target_language(ctx, source_path=Path("lib.rs")), "rust")


class TestFrameworksBlueprintParsing(unittest.TestCase):
    def test_parses_language_key_in_ini(self):
        ini = """
[graph]
entrypoint = orchestrator
targets = ["app"]
dependencies = {"app": []}
[compiler]
optimization_level = "O3"
[cortex]
target_accuracy_floor = 0.99
[frameworks]
language = "python"
"""
        sections, _ = parse_blueprint_content(ini)
        normalized = normalize_optional_sections(sections)
        self.assertEqual(normalized["frameworks"]["language"], "python")


class TestPythonRepoGenerator(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def test_generates_native_python_layout(self):
        spec = build_python_spec("aero-factory", PYTHON_SOURCE, "main.py")
        repo = generate_python_repo(spec, self.tmp / "out")
        root = self.tmp / "out"
        self.assertTrue((root / "main.py").exists())
        self.assertTrue((root / "pyproject.toml").exists())
        self.assertTrue((root / "requirements.txt").exists())
        self.assertFalse((root / "Cargo.toml").exists())
        self.assertFalse((root / "src").exists())
        gitignore = (root / ".gitignore").read_text()
        self.assertIn("__pycache__/", gitignore)
        self.assertIn(".venv/", gitignore)
        self.assertIn(".egg-info/", gitignore)

    def test_keeps_original_filename(self):
        spec = build_python_spec("tool", PYTHON_SOURCE, "main.py")
        self.assertEqual(spec.entry_filename, "main.py")

    def test_infers_third_party_imports(self):
        src = "import numpy\nimport pandas as pd\nimport os\n"
        deps = infer_import_dependencies(src)
        self.assertIn("numpy", deps)
        self.assertIn("pandas", deps)
        self.assertNotIn("os", deps)


class TestPythonScaffoldPipeline(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def test_python_blueprint_skips_cargo(self):
        src = self.tmp / "main.py"
        src.write_text(PYTHON_SOURCE, encoding="utf-8")
        dist = self.tmp / "aero_factory_repository"
        context = {
            "frameworks": {"language": "python"},
            "scaffold": {
                "source_entry": str(src),
                "auto_layout": True,
                "distribution_directory": str(dist),
                "name": "aero-factory",
            },
        }
        result = ScaffoldBuildPipeline(verbose=False).run(context, build=True)
        self.assertEqual(result.language, "python")
        self.assertTrue(result.succeeded)
        self.assertTrue((dist / "main.py").exists())
        self.assertFalse((dist / "Cargo.toml").exists())
        self.assertNotIn(str(TOOL_ROOT), str(dist.resolve()))

    def test_python_syntax_error_fails_validation(self):
        src = self.tmp / "broken.py"
        src.write_text(BROKEN_PYTHON, encoding="utf-8")
        dist = self.tmp / "broken_repo"
        context = {
            "frameworks": {"language": "python"},
            "scaffold": {
                "source_entry": str(src),
                "auto_layout": True,
                "distribution_directory": str(dist),
            },
        }
        result = ScaffoldBuildPipeline(verbose=False).run(context, build=True)
        self.assertFalse(result.succeeded)
        errors = result.scaffold.build["attempts"][0]["errors"]
        self.assertTrue(any("SyntaxError" in e for e in errors))

    def test_build_command_python_blueprint(self):
        src = self.tmp / "main.py"
        src.write_text(PYTHON_SOURCE, encoding="utf-8")
        dist = self.tmp / "out_py"
        bp = self.tmp / "blueprint.aero"
        bp.write_text(
            f"""
[graph]
entrypoint = orchestrator
targets = ["app"]
dependencies = {{"app": []}}
[compiler]
optimization_level = "O3"
[cortex]
target_accuracy_floor = 0.99
[frameworks]
language = "python"
[scaffold]
source_entry = "{src.as_posix()}"
auto_layout = true
distribution_directory = "{dist.as_posix()}"
name = "aero-factory"
""",
            encoding="utf-8",
        )
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main.main([
                "build",
                "--blueprint", str(bp),
                "--workspace", str(self.tmp),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue((dist / "main.py").exists())
        self.assertFalse((dist / "Cargo.toml").exists())
        self.assertIn("language         : python", out.getvalue())


class TestPythonValidator(unittest.TestCase):
    def test_compileall_passes_clean_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text(PYTHON_SOURCE, encoding="utf-8")
            result = PythonValidationRunner().validate_workspace(root)
            self.assertTrue(result.succeeded)


if __name__ == "__main__":
    unittest.main()
