# -*- coding: utf-8 -*-
"""Tests for the deterministic self-healing compilation wrapper."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from core.toolchain.self_healing import (
    Category,
    Diagnostic,
    SymbolIndex,
    _apply_edits,
    _crate_name_from_message,
    _parse_cargo_dependencies,
    _plan_edits,
    categorize,
    flag_healing_failure,
    heal_duplicate_definition,
    heal_missing_crate,
    heal_mismatched_slice_type,
    heal_module,
    heal_pyo3_glob_import,
    make_rust_build_fn,
    parse_rustc_json,
)


def _has(b):
    return shutil.which(b) is not None


_RUSTC_JSON = (
    '{"$message_type":"diagnostic","message":"cannot assign twice to immutable variable `x`",'
    '"code":{"code":"E0384"},"level":"error","spans":[{"file_name":"a.rs","byte_start":21,'
    '"byte_end":22,"line_start":1,"column_start":1,"is_primary":true}]}\n'
    '{"$message_type":"diagnostic","message":"aborting due to 1 previous error","level":"error","spans":[]}\n'
)


class TestDiagnosticParsing(unittest.TestCase):
    def test_parse_rustc_json(self):
        diags = parse_rustc_json(_RUSTC_JSON)
        self.assertEqual(len(diags), 1)
        d = diags[0]
        self.assertEqual(d.code, "E0384")
        self.assertEqual(d.start_byte, 21)
        self.assertEqual(d.source, "rustc")


class TestCategorize(unittest.TestCase):
    def test_codes(self):
        self.assertEqual(categorize(Diagnostic("x", "f", code="E0432")), Category.MISSING_IMPORT)
        self.assertEqual(categorize(Diagnostic("x", "f", code="E0384")), Category.IMMUTABLE_ASSIGNMENT)

    def test_messages(self):
        self.assertEqual(
            categorize(Diagnostic('"foo" is not defined', "f")), Category.MISSING_IMPORT)
        self.assertEqual(
            categorize(Diagnostic("cannot assign twice to immutable variable `x`", "f")),
            Category.IMMUTABLE_ASSIGNMENT)
        self.assertEqual(
            categorize(Diagnostic("expected one of `.`, `;`, `?` ... found keyword `let`", "f")),
            Category.MISSING_SEMICOLON)

    def test_uncategorized(self):
        self.assertIsNone(categorize(Diagnostic("some random error", "f", code="E9999")))


class TestHealRules(unittest.TestCase):
    def test_missing_semicolon_edit(self):
        src = b"let x = 1\nlet y = 2\n"
        edits = _plan_edits(src, [Diagnostic("expected `;`", "f", start_byte=9)], "rust", None)
        self.assertEqual(_apply_edits(src, edits), b"let x = 1;\nlet y = 2\n")

    def test_immutable_assignment_adds_mut(self):
        src = b"fn main(){ let x = 1; x = 2; }"
        d = Diagnostic("cannot assign twice to immutable variable `x`", "f", code="E0384")
        edits = _plan_edits(src, [d], "rust", None)
        self.assertEqual(_apply_edits(src, edits), b"fn main(){ let mut x = 1; x = 2; }")

    def test_immutable_only_targets_named_binding(self):
        src = b"fn main(){ let a = 1; let b = 2; b = 3; }"
        d = Diagnostic("cannot assign twice to immutable variable `b`", "f", code="E0384")
        edits = _plan_edits(src, [d], "rust", None)
        out = _apply_edits(src, edits).decode()
        self.assertIn("let mut b", out)
        self.assertNotIn("let mut a", out)

    def test_python_missing_import_uses_dag(self):
        with tempfile.TemporaryDirectory() as ws:
            os.makedirs(os.path.join(ws, "pkg"))
            Path(ws, "pkg", "util.py").write_text("def helper():\n    return 1\n")
            index = SymbolIndex.build(Path(ws))
            self.assertEqual(index.module_for("helper"), "pkg/util.py")
            src = b"x = helper()\n"
            d = Diagnostic('"helper" is not defined', "f")
            edits = _plan_edits(src, [d], "python", index)
            out = _apply_edits(src, edits).decode()
            self.assertTrue(out.startswith("from pkg.util import helper\n"))


class TestHealLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_heal_then_success(self):
        f = Path(self.ws, "a.rs")
        f.write_text("fn main(){ let x = 1; x = 2; }\n")

        def build(path):
            if "let mut x" in path.read_text():
                return []
            return [Diagnostic("cannot assign twice to immutable variable `x`",
                               str(path), code="E0384")]

        report = heal_module(f, build, workspace=self.ws)
        self.assertTrue(report.success)
        self.assertEqual(report.attempts, 2)
        self.assertIn("immutable_assignment", report.applied)
        self.assertIn("let mut x", f.read_text())

    def test_rollback_on_unhealable(self):
        f = Path(self.ws, "b.rs")
        original = "fn main(){ let y = 1; }\n"
        f.write_text(original)
        bp = Path(self.ws, "blueprint.aero")
        bp.write_text('[system]\nname = "t"\n')

        def build(path):
            return [Diagnostic("mysterious failure", str(path), code="E9999")]

        report = heal_module(f, build, workspace=self.ws, blueprint_path=bp)
        self.assertFalse(report.success)
        self.assertTrue(report.rolled_back)
        self.assertEqual(f.read_text(), original)  # restored
        text = bp.read_text()
        self.assertIn("[self_healing]", text)
        self.assertIn("E9999", text)

    def test_budget_capped_at_three_builds(self):
        f = Path(self.ws, "c.rs")
        f.write_text("fn main(){ let z = 1 }\n")
        calls = {"n": 0}

        def build(path):
            calls["n"] += 1
            # Healable category, but never actually fixed -> keeps editing.
            return [Diagnostic("expected `;`", str(path),
                               start_byte=len(path.read_text()) - 3)]

        report = heal_module(f, build, workspace=self.ws)
        self.assertFalse(report.success)
        self.assertTrue(report.rolled_back)
        self.assertLessEqual(calls["n"], 3)
        self.assertLessEqual(report.attempts, 3)

    def test_blueprint_flag_idempotent(self):
        bp = Path(self.ws, "blueprint.aero")
        bp.write_text('[scaling]\nhierarchy_depth = 4\n')
        flag_healing_failure(bp, "m1.rs", "reason a", [Diagnostic("e", "f", code="E1")])
        flag_healing_failure(bp, "m2.rs", "reason b", [Diagnostic("e", "f", code="E2")])
        text = bp.read_text()
        self.assertEqual(text.count("[self_healing]"), 1)  # single table
        self.assertIn("m1.rs", text)
        self.assertIn("m2.rs", text)
        self.assertIn("[scaling]", text)  # other sections preserved
        from src.blueprint.loader import _toml
        parsed = _toml.loads(text)
        self.assertIn("self_healing", parsed)


# ---------------------------------------------------------------------------
# Advanced Rust error handlers (E0433, E0428, E0432, E0308)
# ---------------------------------------------------------------------------


class TestCategorizeAdvanced(unittest.TestCase):
    """Categorisation of the four new rustc error codes."""

    def test_e0433_missing_crate(self):
        d = Diagnostic("cannot find crate `serde` in the list of imported crates", "f", code="E0433")
        self.assertEqual(categorize(d), Category.MISSING_CRATE)

    def test_e0405_missing_trait(self):
        d = Diagnostic("cannot find trait `Serialize` in crate `serde`", "f", code="E0405")
        self.assertEqual(categorize(d), Category.MISSING_CRATE)

    def test_e0428_duplicate(self):
        d = Diagnostic("the name `Foo` is defined multiple times", "f", code="E0428")
        self.assertEqual(categorize(d), Category.DUPLICATE_DEFINITION)

    def test_e0428_by_message(self):
        d = Diagnostic("the name `Bar` is defined multiple times", "f")
        self.assertEqual(categorize(d), Category.DUPLICATE_DEFINITION)

    def test_e0432_super_import(self):
        d = Diagnostic("unresolved imports `super::helper`", "f", code="E0432")
        self.assertEqual(categorize(d), Category.UNRESOLVED_PARENT_IMPORT)

    def test_e0432_without_super_stays_missing_import(self):
        d = Diagnostic("unresolved import `foo`", "f", code="E0432")
        self.assertEqual(categorize(d), Category.MISSING_IMPORT)

    def test_e0308_slice_mismatch(self):
        d = Diagnostic(
            "expected `&[Complex]`, found `&[[Complex; 4]; 4]`",
            "f", code="E0308",
        )
        self.assertEqual(categorize(d), Category.MISMATCHED_SLICE_TYPE)

    def test_e0308_non_slice_stays_none(self):
        d = Diagnostic("expected `i32`, found `f64`", "f", code="E0308")
        self.assertIsNone(categorize(d))


class TestCrateNameExtraction(unittest.TestCase):
    def test_crate_from_e0433(self):
        self.assertEqual(
            _crate_name_from_message("failed to resolve: use of undeclared crate or module `serde`"),
            "serde",
        )

    def test_crate_from_nested_path(self):
        self.assertEqual(
            _crate_name_from_message("cannot find crate `tokio::runtime` in..."),
            "tokio",
        )

    def test_trait_extraction(self):
        self.assertEqual(
            _crate_name_from_message("cannot find trait `Serialize` in crate `serde`"),
            "Serialize",
        )


class TestHealMissingCrate(unittest.TestCase):
    def test_injects_dependency_into_cargo_toml(self):
        with tempfile.TemporaryDirectory() as ws:
            cargo = Path(ws) / "Cargo.toml"
            cargo.write_text('[package]\nname = "test"\n\n[dependencies]\n')
            src = Path(ws) / "main.rs"
            src.write_text("use serde::Serialize;\n")
            d = Diagnostic(
                "failed to resolve: use of undeclared crate or module `serde`",
                str(src), code="E0433",
            )
            heal_missing_crate(b"", d, "rust", cargo_toml_path=cargo)
            text = cargo.read_text()
            self.assertIn('serde = "*"', text)

    def test_creates_dependencies_section_if_absent(self):
        with tempfile.TemporaryDirectory() as ws:
            cargo = Path(ws) / "Cargo.toml"
            cargo.write_text('[package]\nname = "test"\n')
            d = Diagnostic(
                "use of undeclared crate or module `rayon`",
                str(Path(ws) / "lib.rs"), code="E0433",
            )
            heal_missing_crate(b"", d, "rust", cargo_toml_path=cargo)
            text = cargo.read_text()
            self.assertIn("[dependencies]", text)
            self.assertIn('rayon = "*"', text)

    def test_idempotent_if_already_present(self):
        with tempfile.TemporaryDirectory() as ws:
            cargo = Path(ws) / "Cargo.toml"
            cargo.write_text('[package]\nname = "t"\n\n[dependencies]\nserde = "1"\n')
            d = Diagnostic(
                "crate or module `serde`", str(Path(ws) / "main.rs"), code="E0433"
            )
            heal_missing_crate(b"", d, "rust", cargo_toml_path=cargo)
            self.assertEqual(cargo.read_text().count("serde"), 1)

    def test_preserves_existing_features_and_version(self):
        with tempfile.TemporaryDirectory() as ws:
            cargo = Path(ws) / "Cargo.toml"
            original = (
                '[package]\nname = "pyo3-demo"\n\n[dependencies]\n'
                'pyo3 = { version = "0.20", features = ["extension-module"] }\n'
            )
            cargo.write_text(original)
            d = Diagnostic(
                "cannot find crate `pyo3`", str(Path(ws) / "lib.rs"), code="E0433",
            )
            heal_missing_crate(b"", d, "rust", cargo_toml_path=cargo)
            self.assertEqual(cargo.read_text(), original)

    def test_preserves_multiline_table_dependency(self):
        with tempfile.TemporaryDirectory() as ws:
            cargo = Path(ws) / "Cargo.toml"
            original = (
                '[package]\nname = "t"\n\n[dependencies]\n'
                'tokio = { version = "1",\n'
                '  features = ["full"] }\n'
            )
            cargo.write_text(original)
            d = Diagnostic(
                "use of undeclared crate or module `tokio`",
                str(Path(ws) / "main.rs"), code="E0433",
            )
            heal_missing_crate(b"", d, "rust", cargo_toml_path=cargo)
            self.assertEqual(cargo.read_text(), original)


class TestHealMismatchedSlice(unittest.TestCase):
    def test_flatten_appended(self):
        src = b"call(&matrix)"
        d = Diagnostic(
            "expected `&[Complex]`, found `&[[Complex; 4]; 4]`",
            "f", code="E0308", start_byte=5, end_byte=12,
        )
        edits = heal_mismatched_slice_type(src, d, "rust")
        out = _apply_edits(src, edits)
        self.assertIn(b".flatten()", out)

    def test_no_double_flatten(self):
        src = b"call(&matrix.flatten())"
        d = Diagnostic(
            "expected `&[Complex]`, found `&[[Complex; 4]; 4]`",
            "f", code="E0308", start_byte=5, end_byte=22,
        )
        edits = heal_mismatched_slice_type(src, d, "rust")
        self.assertEqual(edits, [])

    def test_non_rust_is_noop(self):
        d = Diagnostic(
            "expected `&[Complex]`, found `&[[Complex; 4]; 4]`",
            "f", code="E0308", start_byte=0, end_byte=5,
        )
        self.assertEqual(heal_mismatched_slice_type(b"hello", d, "python"), [])


class TestPlanEditsAdvanced(unittest.TestCase):
    def test_missing_crate_dispatched(self):
        d = Diagnostic(
            "use of undeclared crate or module `serde`", "f", code="E0433",
        )
        edits = _plan_edits(b"", [d], "rust", None)
        # heal_missing_crate modifies Cargo.toml as side-effect, returns []
        self.assertEqual(edits, [])

    def test_slice_mismatch_dispatched(self):
        src = b"func(&arr)"
        d = Diagnostic(
            "expected `&[i32]`, found `&[[i32; 3]; 3]`",
            "f", code="E0308", start_byte=5, end_byte=9,
        )
        edits = _plan_edits(src, [d], "rust", None)
        self.assertTrue(any(".flatten()" in e.text for e in edits))


@unittest.skipUnless(_has("rustc"), "rustc not available")
class TestRealRustc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_mutability_heal(self):
        f = Path(self.ws, "m.rs")
        f.write_text("fn main(){ let x = 1; x = 2; let _ = x; }\n")
        report = heal_module(f, make_rust_build_fn(), workspace=self.ws)
        self.assertTrue(report.success)
        self.assertIn("let mut x", f.read_text())

    def test_real_semicolon_heal(self):
        f = Path(self.ws, "s.rs")
        f.write_text("fn main(){ let x = 1 let _ = x; }\n")
        report = heal_module(f, make_rust_build_fn(), workspace=self.ws)
        self.assertTrue(report.success)


class TestParseCargoDepedencies(unittest.TestCase):
    def test_inline_dependency(self):
        text = '[dependencies]\nserde = "1"\n'
        deps = _parse_cargo_dependencies(text)
        self.assertIn("serde", deps)

    def test_table_dependency(self):
        text = '[dependencies]\npyo3 = { version = "0.20", features = ["extension-module"] }\n'
        deps = _parse_cargo_dependencies(text)
        self.assertIn("pyo3", deps)

    def test_multiline_table(self):
        text = '[dependencies]\ntokio = { version = "1",\n  features = ["full"] }\n'
        deps = _parse_cargo_dependencies(text)
        self.assertIn("tokio", deps)

    def test_stops_at_next_section(self):
        text = '[dependencies]\nserde = "1"\n\n[dev-dependencies]\ntest = "1"\n'
        deps = _parse_cargo_dependencies(text)
        self.assertIn("serde", deps)
        self.assertNotIn("test", deps)


class TestManifestRollback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_cargo_toml_restored_on_rollback(self):
        cargo = Path(self.ws) / "Cargo.toml"
        original_cargo = (
            '[package]\nname = "demo"\n\n'
            '[dependencies]\npyo3 = { version = "0.20", features = ["extension-module"] }\n'
        )
        cargo.write_text(original_cargo)
        f = Path(self.ws, "lib.rs")
        f.write_text("use unknown_crate::Thing;\n")

        def build(path):
            return [Diagnostic(
                "use of undeclared crate or module `unknown_crate`",
                str(path), code="E0433",
            )]

        report = heal_module(f, build, workspace=self.ws)
        self.assertFalse(report.success)
        self.assertTrue(report.rolled_back)
        # Cargo.toml must be restored to its original state (features intact).
        self.assertEqual(cargo.read_text(), original_cargo)


class TestPyo3GlobImport(unittest.TestCase):
    def test_categorize_glob_import(self):
        d = Diagnostic(
            "#[pymodule] cannot import glob statements", "lib.rs", severity="error"
        )
        self.assertEqual(categorize(d), Category.PYO3_GLOB_IMPORT)

    def test_expands_glob_to_named_imports(self):
        src = (
            b"use pyo3::prelude::*;\n"
            b"struct Alpha;\n"
            b"enum Beta { A, B }\n"
            b"fn gamma() {}\n"
            b"#[pymodule]\n"
            b"mod my_ext {\n"
            b"    use super::*;\n"
            b"}\n"
        )
        d = Diagnostic("#[pymodule] cannot import glob statements", "lib.rs")
        edits = heal_pyo3_glob_import(src, d, "rust")
        out = _apply_edits(src, edits).decode()
        self.assertIn("use super::{Alpha, Beta, gamma};", out)
        self.assertNotIn("use super::*;", out)

    def test_no_pymodule_is_noop(self):
        src = b"fn x() {}\nmod m { use super::*; }\n"
        d = Diagnostic("#[pymodule] cannot import glob statements", "lib.rs")
        self.assertEqual(heal_pyo3_glob_import(src, d, "rust"), [])

    def test_non_rust_is_noop(self):
        d = Diagnostic("#[pymodule] cannot import glob statements", "lib.rs")
        self.assertEqual(heal_pyo3_glob_import(b"x", d, "python"), [])


class TestTypeAwareDuplicate(unittest.TestCase):
    def test_renames_2d_matrix_variant(self):
        src = (
            b"fn build_unitary() -> Vec<Complex> { vec![] }\n"
            b"fn build_unitary() -> [[Complex; 4]; 4] { [[Complex; 4]; 4] }\n"
        )
        d = Diagnostic(
            "the name `build_unitary` is defined multiple times",
            "lib.rs", code="E0428",
        )
        edits = heal_duplicate_definition(src, d, "rust")
        out = _apply_edits(src, edits).decode()
        # The flat Vec variant is preserved; the 2D variant is deprecated.
        self.assertIn("fn build_unitary() -> Vec<Complex>", out)
        self.assertIn("fn build_unitary_legacy() -> [[Complex; 4]; 4]", out)

    def test_fallback_comments_out_when_not_type_discriminable(self):
        src = b"fn foo() {}\nfn foo() {}\n"
        d = Diagnostic(
            "the name `foo` is defined multiple times", "lib.rs", code="E0428",
        )
        edits = heal_duplicate_definition(src, d, "rust")
        out = _apply_edits(src, edits).decode()
        self.assertIn("auto-healed: duplicate removed", out)


if __name__ == "__main__":
    unittest.main()
