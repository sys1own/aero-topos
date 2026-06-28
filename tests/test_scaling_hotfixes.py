import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import blueprint_parser
import main
import orchestrator
from builder_brains import parameter_tuner
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


if __name__ == "__main__":
    unittest.main()
