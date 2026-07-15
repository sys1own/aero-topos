# -*- coding: utf-8 -*-
"""Unit tests for translator.ffi_codegen."""

import os
import tempfile
import unittest
from pathlib import Path

from translator.ffi_codegen import (
    FfiNode,
    assign_nodes,
    generate_wrapper_fn,
    generate_aero_ffi_module,
    generate_legacy_dispatch,
    generate_single_handle,
    generate_ffi_artifacts,
    AERO_NODE_CAP,
)
from translator.rust_ast import RustFn, RustParam


def _make_fn(name, params=None, return_type="Vec<f64>"):
    return RustFn(
        name=name,
        start_byte=0, end_byte=100,
        start_line=1, end_line=5,
        signature=f"pub fn {name}(state: &[f64], dim: usize, coupling: f64) -> {return_type} ",
        params=params or [],
        return_type=return_type,
    )


class TestAssignNodes(unittest.TestCase):
    def test_assigns_sequential_indices(self):
        fns = [_make_fn("foo"), _make_fn("bar")]
        nodes = assign_nodes(fns)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].index, 0)
        self.assertEqual(nodes[1].index, 1)
        self.assertEqual(nodes[0].hook, "aero_execute_node0")
        self.assertEqual(nodes[1].hook, "aero_execute_node1")

    def test_legacy_naming(self):
        fns = [_make_fn("compute_stuff")]
        nodes = assign_nodes(fns)
        self.assertEqual(nodes[0].legacy, "compute_stuff_legacy")

    def test_stub_expr(self):
        fns = [_make_fn("my_func")]
        nodes = assign_nodes(fns)
        self.assertIn("legacy::my_func_legacy", nodes[0].stub_expr)


class TestGenerateWrapperFn(unittest.TestCase):
    def setUp(self):
        # Each test gets a fresh generator so name-emission state is isolated.
        from translator import ffi_codegen
        ffi_codegen._default_generator = None

    def test_apply_unitary_wrapper(self):
        fn = _make_fn("apply_unitary")
        node = assign_nodes([fn])[0]
        wrapper = generate_wrapper_fn(node)
        self.assertIn("aero_ffi::", wrapper)
        self.assertIn("pub fn apply_unitary", wrapper)
        # Modern standard signature: flat &[f64] in, Vec<f64> out.
        self.assertIn("input: &[f64]", wrapper)
        self.assertIn("-> Vec<f64>", wrapper)

    def test_generic_fallback_wrapper(self):
        fn = _make_fn("unknown_func")
        node = assign_nodes([fn])[0]
        wrapper = generate_wrapper_fn(node)
        self.assertIn("aero_ffi::", wrapper)
        self.assertIn("pub fn unknown_func", wrapper)
        self.assertIn("input: &[f64]", wrapper)


class TestGenerateAeroFFIModule(unittest.TestCase):
    def setUp(self):
        from translator import ffi_codegen
        ffi_codegen._default_generator = None

    def test_generates_valid_module(self):
        fns = [_make_fn("apply_unitary"), _make_fn("evolve_state_rk4")]
        nodes = assign_nodes(fns)
        module = generate_aero_ffi_module(nodes, "test_module")
        self.assertIn("Aero FFI Module", module)
        self.assertIn("test_module", module)
        self.assertIn("aero_execute_node0", module)
        self.assertIn("aero_execute_node1", module)
        self.assertIn("OnceLock", module)
        self.assertIn(f"AERO_NODE_CAP: usize = {AERO_NODE_CAP}", module)

    def test_single_node_module(self):
        fns = [_make_fn("some_fn")]
        nodes = assign_nodes(fns)
        module = generate_aero_ffi_module(nodes, "single_mod")
        self.assertIn("aero_execute_node0", module)


class TestGenerateLegacyDispatch(unittest.TestCase):
    def setUp(self):
        from translator import ffi_codegen
        ffi_codegen._default_generator = None

    def test_known_function_templates(self):
        known_names = ["apply_unitary", "compute_braiding_matrix",
                       "evolve_state_rk4", "topological_invariant"]
        fns = [_make_fn(n) for n in known_names]
        nodes = assign_nodes(fns)
        code = generate_legacy_dispatch(nodes)
        self.assertIn("apply_unitary_legacy", code)
        self.assertIn("compute_braiding_matrix_legacy", code)
        self.assertIn("evolve_state_rk4_legacy", code)
        self.assertIn("topological_invariant_legacy", code)

    def test_unknown_function_passthrough(self):
        fns = [_make_fn("custom_thing")]
        nodes = assign_nodes(fns)
        code = generate_legacy_dispatch(nodes)
        self.assertIn("custom_thing_legacy", code)
        self.assertIn("input.to_vec()", code)


class TestGenerateSingleHandle(unittest.TestCase):
    def setUp(self):
        from translator import ffi_codegen
        ffi_codegen._default_generator = None

    def test_generates_module(self):
        result = generate_single_handle("my_function", "my_module")
        self.assertIn("my_function", result)
        self.assertIn("my_module", result)
        self.assertIn("my_function_invoke", result)


class TestTemplateAgnosticGeneration(unittest.TestCase):
    """Verify the generator is driven by external templates and registries."""

    def setUp(self):
        from translator import ffi_codegen
        ffi_codegen._default_generator = None

    def test_registry_selects_domain_specific_legacy_template(self):
        known_names = ["apply_unitary", "compute_braiding_matrix",
                       "evolve_state_rk4", "topological_invariant"]
        fns = [_make_fn(n) for n in known_names]
        nodes = assign_nodes(fns)
        code = generate_legacy_dispatch(nodes)
        # The default registry maps these to their domain-specific legacy files,
        # so the generator no longer embeds physics-specific bodies in source.
        self.assertIn("let dim = input[0] as usize;", code)
        self.assertIn("pub fn apply_unitary_legacy", code)
        self.assertIn("pub fn compute_braiding_matrix_legacy", code)

    def test_blueprint_template_dir_override(self):
        """A project-level template directory can override the default wrapper."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp) / "ffi"
            tdir.mkdir()
            (tdir / "registry.json").write_text(
                '{"default": {"wrapper": "wrapper_custom.rs"}}',
                encoding="utf-8",
            )
            (tdir / "wrapper_custom.rs").write_text(
                "pub fn $name(input: &[f64]) -> Vec<f64> { /* custom */ }\n",
                encoding="utf-8",
            )
            fn = _make_fn("project_fn")
            node = assign_nodes([fn])[0]
            wrapper = generate_wrapper_fn(
                node,
                blueprint={"artifact_generation": {"template_dirs": [str(tdir)]}},
            )
            self.assertIn("/* custom */", wrapper)
            self.assertIn("pub fn project_fn", wrapper)


class TestCollisionAvoidanceAndIdempotency(unittest.TestCase):
    """Verify the generator refuses to emit duplicate definitions."""

    def test_detects_conflicts_with_existing_source(self):
        source = """
pub fn apply_unitary(input: &[f64]) -> Vec<f64> { vec![] }
"""
        fn = _make_fn("apply_unitary")
        node = assign_nodes([fn])[0]
        report = generate_ffi_artifacts(source, [node], "test_mod")
        self.assertIn("apply_unitary", report["conflicts"])
        self.assertEqual(report["files"], {})

    def test_idempotent_across_reprocessing(self):
        fn = _make_fn("steady_fn")
        node = assign_nodes([fn])[0]
        first = generate_wrapper_fn(node)
        second = generate_wrapper_fn(node)
        # With a fresh generator per call (no shared emission cache) the output is
        # identical; with the shared default generator the second call returns the
        # cached output for the same node name.  Either way no duplicate
        # definitions are produced.
        self.assertEqual(first, second)

    def test_duplicate_nodes_are_deduplicated(self):
        fn = _make_fn("dup")
        nodes = assign_nodes([fn, fn])  # same node twice
        # Should not crash / should emit a single definition.
        code = generate_legacy_dispatch(nodes)
        self.assertEqual(code.count("pub fn dup_legacy"), 1)


if __name__ == "__main__":
    unittest.main()
