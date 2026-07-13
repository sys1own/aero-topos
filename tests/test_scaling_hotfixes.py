import json
import os
import pathlib
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import blueprint_parser
import main
import orchestrator
from builder_brains import decision_tree, parameter_tuner
from builder_brains.scanner import resolve_anomaly_ceiling


class TestScalingHotfixes(unittest.TestCase):
    def test_dynamic_anomaly_ceiling_scales_with_target_count(self):
        self.assertEqual(blueprint_parser.get_anomaly_ceiling([]), 50)
        self.assertEqual(blueprint_parser.get_anomaly_ceiling(["f"] * 5000), 250)
        self.assertEqual(blueprint_parser.get_anomaly_ceiling(["f"] * 20000), 1000)

    def test_toml_parse_attaches_parser_validation(self):
        minimal = """
[system]
name = "scale-test"
strategy = "DIRECT_COMPILE"

[context_registry.core_logic]
path = "./app.py"
language = "python"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(minimal)
            blueprint_path = handle.name

        try:
            context = blueprint_parser.parse_blueprint(blueprint_path)
        finally:
            os.remove(blueprint_path)

        parser_validation = context.get("parser_validation", {})
        self.assertEqual(parser_validation.get("scan_targets"), ["./app.py"])
        self.assertEqual(parser_validation.get("anomaly_ceiling"), 50)
        self.assertEqual(parser_validation.get("parameter_validation_failures"), 0)

    def test_write_gate_requires_real_output(self):
        self.assertTrue(orchestrator.should_write_aeroc("DIRECT_COMPILE", 1, 128))
        self.assertFalse(orchestrator.should_write_aeroc("DIRECT_COMPILE", 0, 128))
        self.assertFalse(orchestrator.should_write_aeroc("DIRECT_COMPILE", 1, 0))

    def test_scanner_prefers_dynamic_ceiling_over_default_manifest_floor(self):
        ceiling = resolve_anomaly_ceiling(
            ["file.py"] * 5000,
            parser_validation={},
            thresholds={"anomaly_alert_ceiling": 50},
        )
        self.assertEqual(ceiling, 250)

    def test_blueprint_strategy_is_preserved_under_high_anomaly_pressure(self):
        metadata = {
            "blueprint_strategy": "DIRECT_COMPILE",
            "resolved_strategy": "AGGRESSIVE_MUTATION",
            "strategy_mode": "aggressive_decomposition",
            "selected_action_label": "execute_polyglot_decomposition",
            "anomaly_count": 50,
            "anomaly_ceiling": 50,
        }
        orchestrator._honor_blueprint_strategy(metadata)
        self.assertEqual(metadata["resolved_strategy"], "DIRECT_COMPILE")
        self.assertEqual(metadata["strategy_mode"], "direct_compile")
        self.assertEqual(metadata["selected_action_label"], "honor_blueprint_strategy")

    def test_parameter_tuner_forwards_current_config_without_reset(self):
        current_config = {"learning_rate": 0.1254321, "mutation_sigma": 0.05}
        forwarded = parameter_tuner._forward_current_config(current_config)
        self.assertEqual(
            forwarded,
            {"learning_rate": 0.125432, "mutation_sigma": 0.05},
        )

    def test_build_command_bypasses_orchestrator_loop_for_direct_compile(self):
        context = {"system": {"strategy": "DIRECT_COMPILE"}}
        args = Namespace(
            source=None,
            workspace=".",
            blueprint="blueprint.aero",
            config=None,
            cycles=None,
            no_scaffold_build=False,
        )
        with mock.patch("blueprint_parser.parse_blueprint", return_value=context), mock.patch(
            "main.orchestrator.run_direct_compile",
            return_value={
                "compiled_target_count": 1,
                "bytes_written": 128,
                "aeroc_output": "/tmp/matrix.aeroc",
            },
        ) as direct_compile, mock.patch("main.orchestrator.run_build") as run_build:
            rc = main.build_command(args)

        self.assertEqual(rc, 0)
        direct_compile.assert_called_once()
        run_build.assert_not_called()

    def test_run_direct_compile_updates_compilation_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = tmp
            source_path = os.path.join(tmp, "matrix.py")
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write("def compute():\n    return 42\n")

            context = {
                "system": {"strategy": "DIRECT_COMPILE"},
                "scaffold": {"source_entry": source_path},
                "parser_validation": {"scan_targets": [source_path], "anomaly_ceiling": 50},
            }
            summary = orchestrator.run_direct_compile(workspace, build_context=context)

            self.assertGreater(summary["compiled_target_count"], 0)
            self.assertGreater(summary["bytes_written"], 0)
            self.assertTrue(os.path.isfile(summary["aeroc_output"]))

    def test_34_file_scale_boundary_preserves_direct_compile(self):
        """A 34-file workspace must not trigger a strategy reversion to BALANCED.

        The dynamic anomaly ceiling saturates at the small-workspace floor (50)
        and the decision tree must continue to honor the explicit DIRECT_COMPILE
        directive without entering the evolutionary FSM.
        """
        files = [f"src_{i}.py" for i in range(34)]
        self.assertEqual(blueprint_parser.get_anomaly_ceiling(files), 50)
        self.assertEqual(
            resolve_anomaly_ceiling(files, {}, {"anomaly_alert_ceiling": 10}),
            50,
        )

        metadata = {
            "blueprint_system_strategy": "DIRECT_COMPILE",
            "active_command": "build",
            "scan_targets": files,
            "current_score": 0.5,
            "current_cycle": 1,
        }
        result = decision_tree.evaluate(metadata, {})
        self.assertEqual(result["resolved_strategy"], "DIRECT_COMPILE")
        self.assertEqual(result["primary_strategy"], "DIRECT_COMPILE")
        self.assertEqual(result["strategy_mode"], "direct_compile")
        self.assertEqual(result["fallback_cascade_depth"], 0)

    def test_direct_compile_early_exit_gate_bypasses_fsm(self):
        """The dominance gate must short-circuit the FSM and fallback cascade.

        When the user strategy is DIRECT_COMPILE, the decision tree should exit
        immediately with a DIRECT_COMPILE routing decision and no heuristic state
        transitions.
        """
        metadata = {
            "blueprint_system_strategy": "DIRECT_COMPILE",
            "active_command": "build",
            "current_score": 0.5,
            "current_cycle": 1,
        }
        result = decision_tree.evaluate(metadata, {})
        self.assertEqual(result["resolved_strategy"], "DIRECT_COMPILE")
        self.assertEqual(result["primary_strategy"], "DIRECT_COMPILE")
        self.assertEqual(result["strategy_mode"], "direct_compile")
        self.assertEqual(result["selected_action_label"], "direct_compile")
        self.assertEqual(result["fallback_cascade_depth"], 0)
        self.assertFalse(result["kinetic_stagnation_anomaly"])
        self.assertEqual(result["decision_tree_status"], "complete")
        self.assertEqual(result["fsm_snapshot"]["current_state"], None)
        self.assertEqual(result["fsm_snapshot"]["transition_count"], 0)
        self.assertEqual(result["fsm_snapshot"]["states"], {})

    def test_run_build_clamp_locks_direct_compile_keys(self):
        """The orchestrator clamp must lock every structural strategy key to DIRECT_COMPILE.

        Even when the decision-tree stage returns AGGRESSIVE_MUTATION (simulating
        heuristic drift), a build pass with a DIRECT_COMPILE blueprint must end
        the cycle with strategy, primary_strategy, and resolved_strategy all set to
        DIRECT_COMPILE.
        """
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle)

            blueprint_path = os.path.join(tmp, "blueprint.aero")
            with open(blueprint_path, "w", encoding="utf-8") as handle:
                handle.write('[system]\nstrategy = "DIRECT_COMPILE"\n')

            original_manifest_path = orchestrator._MANIFEST_PATH
            orchestrator._MANIFEST_PATH = pathlib.Path(manifest_path)

            def mock_scanner(metadata, hyper_params):
                metadata["scan_coverage"] = 0.5
                metadata["scan_target_count"] = 34
                metadata["anomaly_count"] = 50
                metadata["anomaly_ceiling"] = 50
                metadata["scan_targets"] = [f"f{i}.py" for i in range(34)]
                metadata["file_fingerprints"] = {f"f{i}.py": "fp" for i in range(34)}
                metadata["scanner_wall_seconds"] = 0.1
                metadata["aggregate_token_profile"] = {"function_def": 34, "comment_line": 0}
                return metadata

            def mock_decision_tree(metadata, hyper_params):
                metadata["resolved_strategy"] = "AGGRESSIVE_MUTATION"
                metadata["primary_strategy"] = "AGGRESSIVE_MUTATION"
                metadata["strategy_mode"] = "aggressive_decomposition"
                metadata["selected_action_label"] = "execute_polyglot_decomposition"
                metadata["kinetic_stagnation_anomaly"] = False
                metadata["is_stagnant"] = False
                return metadata

            def mock_tuner(metadata, hyper_params):
                metadata["best_config"] = {}
                metadata["pareto_frontier"] = []
                metadata["survival_tracker_stats"] = {"hypervolume": 0.0}
                return metadata

            try:
                with mock.patch.object(
                    orchestrator,
                    "_load_brain_modules",
                    return_value=[
                        ("scanner", mock_scanner),
                        ("decision_tree", mock_decision_tree),
                        ("parameter_tuner", mock_tuner),
                    ],
                ):
                    with mock.patch.object(
                        orchestrator,
                        "_compile_targets",
                        return_value={
                            "compiled_targets": [],
                            "compiled_target_count": 0,
                            "bytes_written": 0,
                            "optimization_level": "aggressive",
                        },
                    ):
                        with mock.patch.object(
                            orchestrator,
                            "_freeze_uast_matrix",
                            return_value={
                                "matrix_output": os.path.join(tmp, "matrix.aeroc"),
                                "matrix_unit_count": 1,
                                "matrix_bytes_written": 64,
                            },
                        ):
                            with mock.patch.object(
                                orchestrator, "_render_telemetry"
                            ), mock.patch.object(
                                orchestrator,
                                "_persist_orchestrator_state",
                                side_effect=lambda manifest, metadata, *a, **k: manifest,
                            ):
                                with mock.patch.object(
                                    orchestrator,
                                    "_apply_manifest_to_assets",
                                    return_value=[],
                                ):
                                    result = orchestrator.run_build(tmp, cycles=1)
                                    self.assertEqual(result["strategy"], "DIRECT_COMPILE")
                                    self.assertEqual(result["primary_strategy"], "DIRECT_COMPILE")
                                    self.assertEqual(result["resolved_strategy"], "DIRECT_COMPILE")
                                    self.assertEqual(result["strategy_mode"], "direct_compile")
                                    self.assertEqual(result["selected_action_label"], "direct_compile")
            finally:
                orchestrator._MANIFEST_PATH = original_manifest_path


if __name__ == "__main__":
    unittest.main()
