# -*- coding: utf-8 -*-
"""Tests for Rust build diagnostics: RUSTFLAGS policy, error root-cause analysis,
the ``--debug`` plumbing, and the suggestion-aware failure report.

Covers:
* ``src.build.rustflags``        -- optimization words, explicit override, disable.
* ``src.build.error_analysis``   -- method-not-found / unresolved-import diagnosis,
                                     Cargo.lock version resolution.
* ``error_interceptor``          -- failed Rust results get suggestions attached.
* ``aero_ui``                    -- debug + suggestion rendering.
* blueprint plumbing             -- ``optimization`` / ``rustflags`` target fields.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from src.build.compilers import CompileResult
from src.build.error_analysis import (
    RustErrorDiagnosis,
    analyze_rust_error,
    read_locked_versions,
)
from src.build.rustflags import resolve_rustflags


# ---------------------------------------------------------------------------
# RUSTFLAGS policy
# ---------------------------------------------------------------------------


class TestRustFlags(unittest.TestCase):
    def test_default_injects_nothing(self):
        d = resolve_rustflags()
        self.assertFalse(d.inject)
        self.assertEqual(d.flags, [])
        self.assertEqual(d.source, "default")

    def test_none_disables_injection(self):
        d = resolve_rustflags(optimization="none")
        self.assertFalse(d.inject)
        self.assertIn("portable", d.reason)

    def test_generic_is_portable_target_cpu(self):
        d = resolve_rustflags(optimization="generic")
        self.assertTrue(d.inject)
        self.assertEqual(d.flags, ["-C", "target-cpu=generic"])

    def test_native_and_maximum_hardware(self):
        for word in ("native", "maximum_hardware", "aggressive"):
            d = resolve_rustflags(optimization=word)
            self.assertEqual(d.flags, ["-C", "target-cpu=native"], word)
            self.assertTrue(d.inject)

    def test_size(self):
        self.assertEqual(resolve_rustflags(optimization="size").flags, ["-C", "opt-level=z"])

    def test_unknown_word_is_safe(self):
        d = resolve_rustflags(optimization="turbo")
        self.assertFalse(d.inject)
        self.assertIn("unrecognised", d.reason)

    def test_explicit_rustflags_override(self):
        d = resolve_rustflags(optimization="native", rustflags=["-C", "target-cpu=generic"])
        self.assertEqual(d.flags, ["-C", "target-cpu=generic"])  # explicit wins over optimization
        self.assertEqual(d.source, "explicit")

    def test_explicit_empty_disables(self):
        d = resolve_rustflags(rustflags=[])
        self.assertFalse(d.inject)

    def test_env_sets_rustflags_only_when_injecting(self):
        self.assertEqual(resolve_rustflags(optimization="generic").env(), {"RUSTFLAGS": "-C target-cpu=generic"})
        self.assertEqual(resolve_rustflags(optimization="none").env(), {})

    def test_env_preserves_base(self):
        env = resolve_rustflags(optimization="native").env({"PATH": "/x"})
        self.assertEqual(env["PATH"], "/x")
        self.assertIn("RUSTFLAGS", env)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_method_not_found_with_crate_path_and_lock(self):
        (self.root / "Cargo.lock").write_text(
            '[[package]]\nname = "rug"\nversion = "1.24.0"\n', encoding="utf-8"
        )
        stderr = "error[E0599]: no method named `neg_mut` found for struct `rug::Integer` in the current scope"
        d = analyze_rust_error(stderr, dependencies={"rug": "0.22"}, crate_root=self.root)
        self.assertIsNotNone(d)
        self.assertEqual(d.kind, "method_not_found")
        self.assertEqual(d.method, "neg_mut")
        self.assertEqual(d.crate, "rug")
        self.assertEqual(d.resolved_version, "1.24.0")
        self.assertEqual(d.declared_version, "0.22")
        rendered = "\n".join(d.render())
        self.assertIn("version mismatch", rendered)
        self.assertIn("1.24.0", rendered)
        self.assertIn("0.22", rendered)

    def test_method_not_found_bare_type_single_dependency(self):
        stderr = "error[E0599]: no method named `neg_mut` found for struct `Integer` in the current scope"
        d = analyze_rust_error(stderr, dependencies={"rug": "0.22"})
        self.assertEqual(d.crate, "rug")  # single declared dep -> attributed
        self.assertEqual(d.declared_version, "0.22")

    def test_method_not_found_reference_receiver(self):
        stderr = "error[E0599]: no method named `foo` found for reference `&rug::Integer` in the current scope"
        d = analyze_rust_error(stderr, dependencies={"rug": "0.22"})
        self.assertEqual(d.method, "foo")
        self.assertEqual(d.receiver_type, "rug::Integer")
        self.assertEqual(d.crate, "rug")

    def test_unresolved_import(self):
        (self.root / "Cargo.lock").write_text(
            '[[package]]\nname = "serde"\nversion = "1.0.200"\n', encoding="utf-8"
        )
        d = analyze_rust_error("error[E0432]: unresolved import `serde::Thing`", crate_root=self.root)
        self.assertEqual(d.kind, "unresolved_import")
        self.assertEqual(d.crate, "serde")
        self.assertEqual(d.resolved_version, "1.0.200")

    def test_unrelated_error_returns_none(self):
        self.assertIsNone(analyze_rust_error("error[E0425]: cannot find value `x` in this scope"))

    def test_empty_returns_none(self):
        self.assertIsNone(analyze_rust_error(""))

    def test_read_locked_versions(self):
        (self.root / "Cargo.lock").write_text(
            '[[package]]\nname = "a"\nversion = "1.0.0"\n\n[[package]]\nname = "b"\nversion = "2.3.4"\n',
            encoding="utf-8",
        )
        versions = read_locked_versions(self.root)
        self.assertEqual(versions, {"a": "1.0.0", "b": "2.3.4"})

    def test_read_locked_versions_missing(self):
        self.assertEqual(read_locked_versions(self.root), {})
        self.assertEqual(read_locked_versions(None), {})


# ---------------------------------------------------------------------------
# error_interceptor: suggestions attached to Rust failures
# ---------------------------------------------------------------------------


class TestInterceptorDiagnosis(unittest.TestCase):
    def test_rust_failure_gets_suggestions(self):
        from aero_ui import AeroUI
        from error_interceptor import handle_compile_results

        result = CompileResult(
            target_name="calc",
            success=False,
            command=["cargo", "build"],
            stderr="error[E0599]: no method named `neg_mut` found for struct `rug::Integer` in the current scope",
            return_code=101,
            details={"language": "rust", "declared_dependencies": {"rug": "0.22"}},
        )
        buf = io.StringIO()
        ui = AeroUI(stream=buf)
        code = handle_compile_results([result], ui)
        out = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("Possible cause", out)
        self.assertIn("neg_mut", out)
        self.assertIn("version mismatch", out)

    def test_non_rust_failure_has_no_suggestions(self):
        from aero_ui import AeroUI
        from error_interceptor import handle_compile_results

        result = CompileResult(
            target_name="lib",
            success=False,
            command=["g++"],
            stderr="undefined reference to `foo`",
            return_code=1,
            details={"language": "cpp"},
        )
        buf = io.StringIO()
        handle_compile_results([result], AeroUI(stream=buf))
        self.assertNotIn("Possible cause", buf.getvalue())


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


class TestUiRendering(unittest.TestCase):
    def test_debug_block(self):
        from aero_ui import AeroUI

        buf = io.StringIO()
        ui = AeroUI(stream=buf)
        ui.debug_block("Cargo.toml in use", ["[package]", 'name = "x"'])
        out = buf.getvalue()
        self.assertIn("Debug", out)
        self.assertIn('name = "x"', out)

    def test_failure_report_with_suggestions(self):
        from aero_ui import AeroUI

        buf = io.StringIO()
        ui = AeroUI(stream=buf)
        ui.build_failure_report("t", "error: boom", suggestions=["check the version", "pin it"])
        out = buf.getvalue()
        self.assertIn("Aero Build Failure", out)
        self.assertIn("Possible cause", out)
        self.assertIn("check the version", out)

    def test_failure_report_without_suggestions(self):
        from aero_ui import AeroUI

        buf = io.StringIO()
        ui = AeroUI(stream=buf)
        ui.build_failure_report("t", "error: boom")
        self.assertNotIn("Possible cause", buf.getvalue())


# ---------------------------------------------------------------------------
# Blueprint plumbing for optimization / rustflags
# ---------------------------------------------------------------------------


class TestBlueprintRustFlagsPlumbing(unittest.TestCase):
    def test_dsl_accepts_optimization_and_rustflags(self):
        import blueprint_lang

        source = (
            'project "p" { version = "1.0" }\n'
            'target "engine" {\n'
            '    language     = "rust"\n'
            '    sources      = ["src/lib.rs"]\n'
            '    optimization = "none"\n'
            '    rustflags    = ["-C", "target-cpu=generic"]\n'
            "}\n"
        )
        self.assertIsNone(blueprint_lang.check_source(source))

    def test_fields_flow_into_metadata(self):
        import blueprint_lang
        from build_graph import blueprint_to_dag

        source = (
            'project "p" { version = "1.0" }\n'
            'target "engine" {\n'
            '    language     = "rust"\n'
            '    sources      = ["src/lib.rs"]\n'
            '    optimization = "generic"\n'
            "}\n"
        )
        graph = blueprint_to_dag(blueprint_lang.load_source(source))
        meta = graph.targets["engine"].to_dict()
        self.assertEqual(meta["optimization"], "generic")

    def test_infer_records_language_reason(self):
        from src.invisible_config import DAGInferenceEngine, LeanBlueprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "core" / "x.cpp").write_text("int main(){}")
            bp = LeanBlueprint(project="p", targets=["cpp_core"])
            dag = DAGInferenceEngine(bp, root).infer()
            target = dag.targets[0]
            self.assertEqual(target.language, "cpp")
            self.assertIn("hint", target.language_reason)


if __name__ == "__main__":
    unittest.main()
