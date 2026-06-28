# -*- coding: utf-8 -*-
"""Unit tests for translator.ffi_codegen."""

import unittest

from translator.ffi_codegen import (
    FfiNode,
    assign_nodes,
    generate_wrapper_fn,
    generate_aero_ffi_module,
    generate_legacy_dispatch,
    generate_single_handle,
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
    def test_apply_unitary_wrapper(self):
        fn = _make_fn("apply_unitary")
        node = assign_nodes([fn])[0]
        wrapper = generate_wrapper_fn(node)
        self.assertIn("aero_ffi::", wrapper)
        self.assertIn("pub fn apply_unitary", wrapper)

    def test_generic_fallback_wrapper(self):
        fn = _make_fn("unknown_func")
        node = assign_nodes([fn])[0]
        wrapper = generate_wrapper_fn(node)
        self.assertIn("aero_ffi::", wrapper)
        self.assertIn("unwrap_or(0.0)", wrapper)


class TestGenerateAeroFFIModule(unittest.TestCase):
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
    def test_generates_module(self):
        result = generate_single_handle("my_function", "my_module")
        self.assertIn("my_function", result)
        self.assertIn("my_module", result)
        self.assertIn("my_function_invoke", result)


if __name__ == "__main__":
    unittest.main()
