"""Tests for the Invisible Configuration Layer.

Organised by layer:

* ``TestLeanParser``          -- the ultra-lean dialect parser + detection.
* ``TestDAGInference``        -- DAG/language/FFI inference from the file tree.
* ``TestSelfHealing``         -- error-correction loops for glue-code mismatches.
* ``TestInvisibleEngine``     -- parse + infer + executable build_context.
* ``TestParseBlueprintRoute`` -- blueprint_parser routes lean files correctly.
* ``TestInferCli``            -- the ``main.py`` ``infer`` subcommand.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import blueprint_parser
import main
from src.invisible_config import (
    INVARIANTS_NODE,
    DAGInferenceEngine,
    FfiBoundary,
    GlueCodePatcher,
    InvisibleConfigEngine,
    LeanBlueprint,
    LeanBlueprintError,
    SelfHealingExecutor,
    looks_like_lean_blueprint,
    parse_lean_blueprint,
)

EXAMPLE = (
    'project "biophysical_trader"\n'
    "\n"
    'ingest = ["./research/genomics.md", "./research/market_liquidity.txt"]\n'
    'targets = ["cpp_core", "python_dashboard"]\n'
    'optimize = "maximum_hardware"\n'
)


def _build_project(root: Path) -> None:
    (root / "research").mkdir()
    (root / "research" / "genomics.md").write_text("DNA gene mutation rate.")
    (root / "research" / "market_liquidity.txt").write_text("Let spread be the bid-ask spread.")
    (root / "core").mkdir()
    (root / "core" / "engine.cpp").write_text("int main(){return 0;}")
    (root / "core" / "api.hpp").write_text("#pragma once")
    (root / "dashboard").mkdir()
    (root / "dashboard" / "app.py").write_text("print('dash')")


# ---------------------------------------------------------------------------
# Lean parser
# ---------------------------------------------------------------------------


class TestLeanParser(unittest.TestCase):
    def test_parses_the_example(self):
        bp = parse_lean_blueprint(EXAMPLE)
        self.assertEqual(bp.project, "biophysical_trader")
        self.assertEqual(bp.targets, ["cpp_core", "python_dashboard"])
        self.assertEqual(bp.ingest, ["./research/genomics.md", "./research/market_liquidity.txt"])
        self.assertEqual(bp.optimize, "maximum_hardware")

    def test_optimize_defaults_to_balanced(self):
        bp = parse_lean_blueprint('project "x"\ntargets = ["a"]\n')
        self.assertEqual(bp.optimize, "balanced")

    def test_extra_keys_collected(self):
        bp = parse_lean_blueprint('project "x"\ntargets = ["a"]\nthreads = 8\n')
        self.assertEqual(bp.extras["threads"], 8)

    def test_missing_project_raises(self):
        with self.assertRaises(LeanBlueprintError):
            parse_lean_blueprint('targets = ["a"]\n')

    def test_missing_targets_raises(self):
        with self.assertRaises(LeanBlueprintError):
            parse_lean_blueprint('project "x"\n')

    def test_duplicate_key_raises(self):
        with self.assertRaises(LeanBlueprintError):
            parse_lean_blueprint('project "x"\ntargets = ["a"]\ntargets = ["b"]\n')

    def test_malformed_line_raises(self):
        with self.assertRaises(LeanBlueprintError):
            parse_lean_blueprint('project "x"\nthis is not valid\n')

    def test_comments_and_blanks_ignored(self):
        bp = parse_lean_blueprint('# a comment\nproject "x"\n\n// another\ntargets = ["a"]\n')
        self.assertEqual(bp.project, "x")

    def test_detection_accepts_lean(self):
        self.assertTrue(looks_like_lean_blueprint(EXAMPLE))

    def test_detection_rejects_ini(self):
        self.assertFalse(looks_like_lean_blueprint("[meta]\nname = \"x\"\n"))

    def test_detection_rejects_json(self):
        self.assertFalse(looks_like_lean_blueprint('{"project": "x"}'))

    def test_detection_rejects_block_dsl(self):
        self.assertFalse(looks_like_lean_blueprint('project "x" {\n  version = "1.0"\n}\n'))


# ---------------------------------------------------------------------------
# DAG inference
# ---------------------------------------------------------------------------


class TestDAGInference(unittest.TestCase):
    def _infer(self, root: Path, blueprint: LeanBlueprint):
        return DAGInferenceEngine(blueprint, root).infer()

    def test_infers_languages_roles_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            dag = self._infer(root, parse_lean_blueprint(EXAMPLE))
            by_name = {t.name: t for t in dag.targets}
            self.assertEqual(by_name["cpp_core"].language, "cpp")
            self.assertEqual(by_name["cpp_core"].role, "core")
            self.assertIn("core/engine.cpp", by_name["cpp_core"].sources)
            self.assertEqual(by_name["python_dashboard"].language, "python")
            self.assertEqual(by_name["python_dashboard"].role, "binding")
            self.assertIn("dashboard/app.py", by_name["python_dashboard"].sources)

    def test_core_depends_on_text_invariants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            dag = self._infer(root, parse_lean_blueprint(EXAMPLE))
            cpp = next(t for t in dag.targets if t.name == "cpp_core")
            self.assertIn(INVARIANTS_NODE, cpp.depends_on)
            self.assertTrue(dag.has_invariants)

    def test_no_ingest_means_no_invariants_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            bp = parse_lean_blueprint('project "x"\ntargets = ["cpp_core"]\n')
            dag = self._infer(root, bp)
            cpp = next(t for t in dag.targets if t.name == "cpp_core")
            self.assertNotIn(INVARIANTS_NODE, cpp.depends_on)
            self.assertFalse(dag.has_invariants)

    def test_ffi_boundary_cpp_python_is_pybind11(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            dag = self._infer(root, parse_lean_blueprint(EXAMPLE))
            self.assertEqual(len(dag.ffi_boundaries), 1)
            boundary = dag.ffi_boundaries[0]
            self.assertEqual(boundary.provider, "cpp_core")
            self.assertEqual(boundary.consumer, "python_dashboard")
            self.assertEqual(boundary.mechanism, "pybind11")

    def test_ffi_boundary_rust_python_is_pyo3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kernels").mkdir()
            (root / "kernels" / "lib.rs").write_text("pub fn f() {}")
            (root / "ui").mkdir()
            (root / "ui" / "main.py").write_text("print(1)")
            bp = parse_lean_blueprint('project "x"\ntargets = ["rust_kernels", "python_ui"]\n')
            dag = self._infer(root, bp)
            boundary = dag.ffi_boundaries[0]
            self.assertEqual(boundary.provider, "rust_kernels")
            self.assertEqual(boundary.mechanism, "pyo3")

    def test_dependency_consumer_links_to_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            dag = self._infer(root, parse_lean_blueprint(EXAMPLE))
            dash = next(t for t in dag.targets if t.name == "python_dashboard")
            self.assertIn("cpp_core", dash.depends_on)

    def test_execution_order_is_topological(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            dag = self._infer(root, parse_lean_blueprint(EXAMPLE))
            order = dag.topological_order()
            self.assertLess(order.index("cpp_core"), order.index("python_dashboard"))
            self.assertNotIn(INVARIANTS_NODE, order)

    def test_language_inferred_from_sources_when_name_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "widgets").mkdir()
            (root / "widgets" / "thing.rs").write_text("fn f(){}")
            bp = parse_lean_blueprint('project "x"\ntargets = ["widgets"]\n')
            dag = self._infer(root, bp)
            self.assertEqual(dag.targets[0].language, "rust")

    def test_unknown_target_defaults_to_python_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bp = parse_lean_blueprint('project "x"\ntargets = ["mystery"]\n')
            dag = self._infer(root, bp)
            self.assertEqual(dag.targets[0].language, "python")
            self.assertEqual(dag.targets[0].role, "binding")


# ---------------------------------------------------------------------------
# Self-healing error-correction loops
# ---------------------------------------------------------------------------


class TestSelfHealing(unittest.TestCase):
    def _boundary(self, mechanism="pybind11", pl="cpp", cl="python") -> FfiBoundary:
        return FfiBoundary(provider="core", consumer="dash", mechanism=mechanism,
                           provider_language=pl, consumer_language=cl)

    def test_patcher_detects_type_mismatch(self):
        patcher = GlueCodePatcher()
        self.assertTrue(patcher.is_type_mismatch("error: cannot convert int to double"))
        self.assertTrue(patcher.is_type_mismatch("error: mismatched types"))
        self.assertFalse(patcher.is_type_mismatch("error: undefined reference to symbol foo"))

    def test_patcher_inserts_specific_cast_for_known_pair(self):
        patcher = GlueCodePatcher()
        diag = "error: expected double, found int in binding"
        patched = patcher.patch("glue();\n", diag, self._boundary())
        self.assertIsNotNone(patched)
        source, desc = patched
        self.assertIn("AERO_COERCE", source)
        self.assertIn("static_cast<double>", source)

    def test_patcher_returns_none_for_non_mismatch(self):
        patcher = GlueCodePatcher()
        self.assertIsNone(patcher.patch("glue();\n", "undefined reference", self._boundary()))

    def test_executor_heals_after_retry(self):
        boundary = self._boundary()

        class FakeCompiler:
            def __init__(self):
                self.calls = 0

            def __call__(self, glue):
                self.calls += 1
                if "AERO_COERCE" in glue or "AERO_AUTO_COERCE" in glue:
                    return True, "ok"
                return False, "error: cannot convert int to double across binding"

        executor = SelfHealingExecutor(max_attempts=3)
        result = executor.run(FakeCompiler(), "PYBIND11_MODULE(core, m){}\n", boundary)
        self.assertTrue(result.succeeded)
        self.assertTrue(result.healed)
        self.assertGreaterEqual(len(result.attempts), 2)

    def test_executor_succeeds_first_try_is_not_healed(self):
        result = SelfHealingExecutor().run(lambda g: (True, "ok"), "glue", self._boundary())
        self.assertTrue(result.succeeded)
        self.assertFalse(result.healed)
        self.assertEqual(len(result.attempts), 1)

    def test_executor_gives_up_on_non_healable_error(self):
        boundary = self._boundary()
        result = SelfHealingExecutor(max_attempts=5).run(
            lambda g: (False, "error: undefined reference to symbol foo"), "glue", boundary
        )
        self.assertFalse(result.succeeded)
        self.assertEqual(len(result.attempts), 1)  # stopped immediately

    def test_executor_bounded_by_max_attempts(self):
        boundary = self._boundary()
        # Always fails with a mismatch, but the patcher only patches once
        # (marker guard) -> the loop terminates rather than looping forever.
        result = SelfHealingExecutor(max_attempts=4).run(
            lambda g: (False, "error: cannot convert int to double"), "glue", boundary
        )
        self.assertFalse(result.succeeded)
        self.assertLessEqual(len(result.attempts), 4)


# ---------------------------------------------------------------------------
# Engine -> executable build_context
# ---------------------------------------------------------------------------


class TestInvisibleEngine(unittest.TestCase):
    def test_build_context_shape_matches_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            ctx = InvisibleConfigEngine(root).build_context_from_source(EXAMPLE)

        for key in ("compilation_targets", "dependency_matrix", "active_optimizer_flags", "graph"):
            self.assertIn(key, ctx)
        self.assertEqual(ctx["compilation_targets"], ["cpp_core", "python_dashboard"])
        self.assertEqual(ctx["graph"]["dependencies"]["python_dashboard"], ["cpp_core"])
        self.assertIn("inferred_dag", ctx)
        self.assertTrue(ctx["self_healing"]["enabled"])

    def test_maximum_hardware_enables_polymorph_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            ctx = InvisibleConfigEngine(root).build_context_from_source(EXAMPLE)
        self.assertTrue(ctx["polymorphization"]["enabled"])
        self.assertTrue(ctx["gpu"]["enabled"])
        self.assertEqual(ctx["active_optimizer_flags"]["optimization_level"], "O3")

    def test_size_profile_is_conservative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            src = 'project "x"\ntargets = ["cpp_core"]\noptimize = "size"\n'
            ctx = InvisibleConfigEngine(root).build_context_from_source(src)
        self.assertNotIn("polymorphization", ctx)
        self.assertFalse(ctx["gpu"]["enabled"])
        self.assertEqual(ctx["active_optimizer_flags"]["optimization_level"], "Os")

    def test_ingest_becomes_context_sources_and_semantic_fluidity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            ctx = InvisibleConfigEngine(root).build_context_from_source(EXAMPLE)
        self.assertTrue(ctx["semantic_fluidity"]["enabled"])
        self.assertEqual(len(ctx["context"]["sources"]), 2)
        self.assertEqual(ctx["context"]["sources"][0]["target_mapping"], INVARIANTS_NODE)

    def test_context_is_json_serialisable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            ctx = InvisibleConfigEngine(root).build_context_from_source(EXAMPLE)
        json.loads(json.dumps(ctx))  # must not raise


# ---------------------------------------------------------------------------
# blueprint_parser routing
# ---------------------------------------------------------------------------


class TestParseBlueprintRoute(unittest.TestCase):
    def test_parse_blueprint_routes_lean_to_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            bp = root / "blueprint.aero"
            bp.write_text(EXAMPLE)
            ctx = blueprint_parser.parse_blueprint(str(bp), manifest_path=str(root / "none.json"))
        self.assertEqual(ctx["workspace_status"], "inferred_active")
        self.assertEqual(ctx["config_layer"], "invisible")
        self.assertEqual(ctx["compilation_targets"], ["cpp_core", "python_dashboard"])

    def test_parse_blueprint_still_handles_ini(self):
        # A legacy INI blueprint must not be misrouted to lean inference.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bp = root / "blueprint.aero"
            bp.write_text(
                "[graph]\ntargets = [\"a\"]\ndependencies = {\"a\": []}\n"
                "[compiler]\noptimization_level = \"O3\"\n[cortex]\n"
            )
            ctx = blueprint_parser.parse_blueprint(str(bp), manifest_path=str(root / "none.json"))
        # INI path does not set the invisible-layer marker.
        self.assertNotEqual(ctx.get("config_layer"), "invisible")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestInferCli(unittest.TestCase):
    def test_parser_wires_infer_subcommand(self):
        args = main.create_parser().parse_args(["infer", "--workspace", "."])
        self.assertIs(args.handler, main.infer_command)

    def test_infer_prints_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            (root / "blueprint.aero").write_text(EXAMPLE)
            out = io.StringIO()
            with redirect_stdout(out):
                rc = main.main(["infer", "--workspace", str(root)])
            self.assertEqual(rc, 0)
            text = out.getvalue()
            self.assertIn("biophysical_trader", text)
            self.assertIn("pybind11", text)
            self.assertIn("cpp_core", text)
            self.assertIn("python_dashboard", text)
            # The richer output explains detection and execution order.
            self.assertIn("execution order", text)
            self.assertIn("zero-config", text)

    def test_infer_json_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            (root / "blueprint.aero").write_text(EXAMPLE)
            out = io.StringIO()
            with redirect_stdout(out):
                rc = main.main(["infer", "--workspace", str(root), "--json"])
            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["project"], "biophysical_trader")
            self.assertEqual(payload["execution_order"], ["cpp_core", "python_dashboard"])

    def test_infer_rejects_non_lean_blueprint(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blueprint.aero").write_text("[meta]\nname = \"x\"\n")
            err = io.StringIO()
            old = sys.stderr
            sys.stderr = err
            try:
                rc = main.main(["infer", "--workspace", str(root)])
            finally:
                sys.stderr = old
            self.assertEqual(rc, 1)
            self.assertIn("not an ultra-lean blueprint", err.getvalue())


if __name__ == "__main__":
    unittest.main()
