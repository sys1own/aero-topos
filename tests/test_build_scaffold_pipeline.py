# -*- coding: utf-8 -*-
"""Tests for blueprint-driven scaffold build pipeline integration."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main
from blueprint_parser import parse_blueprint_content, normalize_optional_sections
from src.scaffold.pipeline import ScaffoldBuildPipeline, should_run_scaffold_pipeline
from src.scaffold.workspace import TOOL_ROOT

RUG_PYO3_SOURCE = """\
use pyo3::prelude::*;
use rug::{Float, Complex};

#[pymodule]
fn anyon_sim(m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }

fn sector_dim(sec: u8) -> usize {
    let q_dim = match sec { 0 => 2, 1 => 3, _ => 5 };
    q_dim
}
"""


class TestScaffoldBlueprintParsing(unittest.TestCase):
    def test_optional_scaffold_section_defaults(self):
        defaults = normalize_optional_sections({})
        self.assertIn("scaffold", defaults)
        self.assertFalse(defaults["scaffold"]["auto_layout"])

    def test_scaffold_section_parsed_from_ini(self):
        ini = """
[graph]
entrypoint = orchestrator
targets = ["core"]
dependencies = {"core": []}
[compiler]
optimization_level = "O3"
[cortex]
target_accuracy_floor = 0.99
[scaffold]
source_entry = "/content/lib.rs"
auto_layout = true
distribution_directory = "/content/anyon_simulator_repository"
compatibility_shims = ["rug_v1_30_patch", "pyo3_usize_alignment"]
name = "anyon_simulator"
"""
        sections, _ = parse_blueprint_content(ini)
        normalized = normalize_optional_sections(sections)
        scaffold = normalized["scaffold"]
        self.assertEqual(scaffold["source_entry"], "/content/lib.rs")
        self.assertTrue(scaffold["auto_layout"])
        self.assertEqual(scaffold["distribution_directory"], "/content/anyon_simulator_repository")
        self.assertEqual(
            scaffold["compatibility_shims"],
            ["rug_v1_30_patch", "pyo3_usize_alignment"],
        )
        self.assertEqual(scaffold["name"], "anyon_simulator")


class TestScaffoldPipelineDetection(unittest.TestCase):
    def test_detects_auto_layout(self):
        ctx = {"scaffold": {"auto_layout": True, "source_entry": "/x/lib.rs"}}
        self.assertTrue(should_run_scaffold_pipeline(ctx))

    def test_detects_source_entry_only(self):
        ctx = {"scaffold": {"source_entry": "/content/lib.rs"}}
        self.assertTrue(should_run_scaffold_pipeline(ctx))

    def test_skips_when_scaffold_empty(self):
        ctx = {"scaffold": {"auto_layout": False, "source_entry": ""}}
        self.assertFalse(should_run_scaffold_pipeline(ctx))


class TestScaffoldBuildPipeline(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def test_end_to_end_out_of_tree_no_cargo(self):
        src = self.tmp / "external" / "lib.rs"
        src.parent.mkdir(parents=True)
        src.write_text(RUG_PYO3_SOURCE, encoding="utf-8")
        dist = self.tmp / "anyon_simulator_repository"

        context = {
            "frameworks": {"language": "rust"},
            "scaffold": {
                "source_entry": str(src),
                "auto_layout": True,
                "distribution_directory": str(dist),
                "compatibility_shims": ["rug_v1_30_patch", "pyo3_usize_alignment"],
                "name": "anyon_simulator",
            }
        }

        msgs: list[str] = []

        def log(msg: str) -> None:
            msgs.append(msg)

        result = ScaffoldBuildPipeline(logger=log, verbose=True).run(
            context, blueprint_dir=self.tmp, build=False
        )

        self.assertTrue(result.succeeded)
        self.assertTrue(dist.exists())
        self.assertTrue((dist / "Cargo.toml").exists())
        self.assertTrue((dist / "src" / "lib.rs").exists())
        self.assertTrue((dist / "test_binding.py").exists())
        self.assertIn("AeroNegMutExt", (dist / "src" / "lib.rs").read_text())
        self.assertIn("let q_dim: usize = match", (dist / "src" / "lib.rs").read_text())
        self.assertNotIn(str(TOOL_ROOT), str(dist.resolve()))
        joined = "\n".join(msgs)
        self.assertIn("[build:1/5]", joined)
        self.assertIn("[build:4/5]", joined)


class TestBuildCommandScaffoldRouting(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def _write_minimal_blueprint(self, scaffold_section: str) -> Path:
        bp = self.tmp / "blueprint.aero"
        bp.write_text(
            f"""
[graph]
entrypoint = orchestrator
targets = ["core"]
dependencies = {{"core": []}}
[compiler]
optimization_level = "O3"
[cortex]
target_accuracy_floor = 0.99
[frameworks]
language = "rust"
[scaffold]
{scaffold_section}
""",
            encoding="utf-8",
        )
        return bp

    def test_build_routes_to_scaffold_pipeline(self):
        src = self.tmp / "lib.rs"
        src.write_text(RUG_PYO3_SOURCE, encoding="utf-8")
        dist = self.tmp / "out_repo"
        bp = self._write_minimal_blueprint(
            f'source_entry = "{src.as_posix()}"\n'
            f"auto_layout = true\n"
            f'distribution_directory = "{dist.as_posix()}"\n'
            'compatibility_shims = ["rug_v1_30_patch", "pyo3_usize_alignment"]\n'
            'name = "anyon"\n'
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = main.main([
                "build",
                "--blueprint", str(bp),
                "--workspace", str(self.tmp),
                "--no-scaffold-build",
            ])

        self.assertEqual(rc, 0)
        self.assertTrue((dist / "Cargo.toml").exists())
        self.assertIn("Isolated scaffold build complete", out.getvalue())
        self.assertIn("[build:1/5]", out.getvalue())

    def test_build_parser_accepts_blueprint_flag(self):
        args = main.create_parser().parse_args([
            "build", "--blueprint", "/tmp/blueprint.aero",
        ])
        self.assertEqual(args.blueprint, "/tmp/blueprint.aero")


if __name__ == "__main__":
    unittest.main()
