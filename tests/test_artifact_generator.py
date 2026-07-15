# -*- coding: utf-8 -*-
"""Integration tests for the template-agnostic ArtifactGenerator."""

import json
import tempfile
import unittest
from pathlib import Path

from translator.artifact_generator import ArtifactGenerator
from translator.ffi_codegen import FfiNode, assign_nodes
from translator.rust_ast import RustFn


def _make_fn(name, return_type="Vec<f64>"):
    return RustFn(
        name=name,
        start_byte=0,
        end_byte=100,
        start_line=1,
        end_line=5,
        signature=f"pub fn {name}(input: &[f64]) -> {return_type} ",
        return_type=return_type,
    )


class TestProjectTypeAgnosticism(unittest.TestCase):
    """The same ArtifactGenerator can drive physics simulators and web apps."""

    def test_physics_simulator_project_renders_matrix_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp) / "ffi"
            tdir.mkdir()
            (tdir / "registry.json").write_text(
                json.dumps({"default": {
                    "wrapper": "wrapper_physics.rs",
                    "legacy": "legacy_physics.rs",
                }}),
                encoding="utf-8",
            )
            (tdir / "wrapper_physics.rs").write_text(
                "pub fn $name(input: &[f64]) -> Vec<f64> { /* SU(N) braid */ }\n",
                encoding="utf-8",
            )
            (tdir / "legacy_physics.rs").write_text(
                "pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> { /* anyon legacy */ }\n",
                encoding="utf-8",
            )

            fn = _make_fn("su2_braid_matrix")
            node = assign_nodes([fn])[0]
            gen = ArtifactGenerator(extra_template_dirs=[tdir])
            wrapper = gen.wrapper(node)
            legacy = gen.legacy(node)

            self.assertIn("su2_braid_matrix", wrapper)
            self.assertIn("SU(N) braid", wrapper)
            self.assertIn("anyon legacy", legacy)

    def test_web_app_project_renders_handler_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp) / "ffi"
            tdir.mkdir()
            (tdir / "registry.json").write_text(
                json.dumps({"default": {
                    "wrapper": "wrapper_web.rs",
                    "legacy": "legacy_web.rs",
                }}),
                encoding="utf-8",
            )
            (tdir / "wrapper_web.rs").write_text(
                "pub fn $name(input: &[f64]) -> Vec<f64> { /* http handler */ }\n",
                encoding="utf-8",
            )
            (tdir / "legacy_web.rs").write_text(
                "pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> { /* request stub */ }\n",
                encoding="utf-8",
            )

            fn = _make_fn("handle_request")
            node = assign_nodes([fn])[0]
            gen = ArtifactGenerator(extra_template_dirs=[tdir])
            wrapper = gen.wrapper(node)
            legacy = gen.legacy(node)

            self.assertIn("handle_request", wrapper)
            self.assertIn("http handler", wrapper)
            self.assertIn("request stub", legacy)

    def test_collision_avoidance_blocks_duplicate_output(self):
        source = "pub fn su3_braid_matrix(input: &[f64]) -> Vec<f64> { vec![] }\n"
        fn = _make_fn("su3_braid_matrix")
        node = assign_nodes([fn])[0]
        gen = ArtifactGenerator()
        report = gen.generate_ffi_artifacts(source, [node], "mod")
        self.assertIn("su3_braid_matrix", report.conflicts)
        self.assertEqual(report.files, {})

    def test_idempotent_for_identical_node(self):
        fn = _make_fn("so10_braid_matrix")
        node = assign_nodes([fn])[0]
        gen = ArtifactGenerator()
        first = gen.wrapper(node)
        second = gen.wrapper(node)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
