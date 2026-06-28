# -*- coding: utf-8 -*-
"""Unit tests for translator.rust_ast."""

import unittest

from translator.rust_ast import (
    safe_ident,
    map_ffi_type,
    RustFn,
    RustParam,
    Edit,
    apply_edits,
    deactivation_block,
    parse,
    extract_functions,
    last_use_end_byte,
    module_anchor_byte,
)


class TestSafeIdent(unittest.TestCase):
    def test_normal_name(self):
        self.assertEqual(safe_ident("foo_bar"), "foo_bar")

    def test_special_chars(self):
        self.assertEqual(safe_ident("hello-world!"), "hello_world_")

    def test_leading_digit(self):
        self.assertEqual(safe_ident("3abc"), "_3abc")

    def test_empty_string(self):
        self.assertEqual(safe_ident(""), "_anon")

    def test_all_special(self):
        self.assertEqual(safe_ident("!!!"), "___")


class TestMapFFIType(unittest.TestCase):
    def test_slice_f64(self):
        self.assertEqual(map_ffi_type("&[f64]"), ("*const f64", "slice_f64"))
        self.assertEqual(map_ffi_type("&mut [f64]"), ("*const f64", "slice_f64"))

    def test_scalar_f64(self):
        self.assertEqual(map_ffi_type("f64"), ("f64", "scalar_f64"))

    def test_scalar_int_types(self):
        for t in ("usize", "isize", "u32", "i32", "u64", "i64"):
            ext, kind = map_ffi_type(t)
            self.assertEqual(ext, t)
            self.assertEqual(kind, "scalar_int")

    def test_nested_vec(self):
        self.assertEqual(map_ffi_type("Vec<Vec<f64>>"), ("*const f64", "nested_f64"))

    def test_unknown_defaults_to_slice(self):
        self.assertEqual(map_ffi_type("SomeCustomType"), ("*const f64", "slice_f64"))


class TestParse(unittest.TestCase):
    def test_parses_simple_rust(self):
        src = "fn main() { println!(\"hello\"); }"
        tree = parse(src)
        self.assertIsNotNone(tree)
        self.assertIsNotNone(tree.root_node)


class TestExtractFunctions(unittest.TestCase):
    def test_extracts_single_function(self):
        src = "fn add(a: i32, b: i32) -> i32 { a + b }"
        fns = extract_functions(src)
        self.assertEqual(len(fns), 1)
        self.assertEqual(fns[0].name, "add")
        self.assertEqual(len(fns[0].params), 2)
        self.assertEqual(fns[0].params[0].name, "a")
        self.assertEqual(fns[0].params[0].rust_type, "i32")

    def test_extracts_multiple_functions(self):
        src = """
fn foo() {}
fn bar(x: f64) -> f64 { x * 2.0 }
"""
        fns = extract_functions(src)
        self.assertEqual(len(fns), 2)
        names = [f.name for f in fns]
        self.assertIn("foo", names)
        self.assertIn("bar", names)

    def test_unsafe_detection(self):
        src = "unsafe fn danger() {}"
        fns = extract_functions(src)
        self.assertEqual(len(fns), 1)
        self.assertTrue(fns[0].is_unsafe)

    def test_node_text(self):
        src = "fn hello() { 42 }"
        fns = extract_functions(src)
        self.assertEqual(fns[0].node_text(src), src)


class TestLastUseEndByte(unittest.TestCase):
    def test_with_use_declarations(self):
        src = "use std::io;\nuse std::fs;\n\nfn main() {}"
        end = last_use_end_byte(src)
        self.assertGreater(end, 0)

    def test_no_use_declarations(self):
        src = "fn main() {}"
        end = last_use_end_byte(src)
        self.assertEqual(end, 0)


class TestModuleAnchorByte(unittest.TestCase):
    def test_with_use(self):
        src = "use std::io;\n\nfn main() {}"
        anchor = module_anchor_byte(src)
        self.assertGreater(anchor, 0)

    def test_without_use_with_inner_attr(self):
        src = '#![allow(dead_code)]\n\nfn main() {}'
        anchor = module_anchor_byte(src)
        self.assertGreater(anchor, 0)

    def test_empty_file(self):
        src = "fn main() {}"
        anchor = module_anchor_byte(src)
        self.assertEqual(anchor, 0)


class TestApplyEdits(unittest.TestCase):
    def test_single_edit(self):
        src = "hello world"
        edits = [Edit(start=6, end=11, replacement="rust")]
        result = apply_edits(src, edits)
        self.assertEqual(result, "hello rust")

    def test_multiple_non_overlapping_edits(self):
        src = "aaa bbb ccc"
        edits = [
            Edit(start=0, end=3, replacement="xxx"),
            Edit(start=8, end=11, replacement="zzz"),
        ]
        result = apply_edits(src, edits)
        self.assertEqual(result, "xxx bbb zzz")

    def test_insertion(self):
        src = "fn foo() {}"
        # Insert at position 3
        edits = [Edit(start=3, end=3, replacement="bar_")]
        result = apply_edits(src, edits)
        self.assertEqual(result, "fn bar_foo() {}")

    def test_out_of_range_raises(self):
        src = "short"
        edits = [Edit(start=0, end=100, replacement="x")]
        with self.assertRaises(ValueError):
            apply_edits(src, edits)

    def test_overlapping_raises(self):
        src = "hello world"
        edits = [
            Edit(start=0, end=7, replacement="a"),
            Edit(start=5, end=11, replacement="b"),
        ]
        with self.assertRaises(ValueError):
            apply_edits(src, edits)


class TestDeactivationBlock(unittest.TestCase):
    def test_block_comment_form(self):
        text = "fn foo() { 1 + 1 }"
        result = deactivation_block(text, "node0", "aero_execute_node0")
        self.assertIn("/*", result)
        self.assertIn("*/", result)
        self.assertIn("AERO-DEACTIVATED node0", result)

    def test_line_comment_form_when_close_present(self):
        text = "let x = */ oops"
        result = deactivation_block(text, "node1", "hook1")
        self.assertIn("// ", result)
        self.assertNotIn("/*", result)


if __name__ == "__main__":
    unittest.main()
