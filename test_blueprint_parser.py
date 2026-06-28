# -*- coding: utf-8 -*-
"""
Unit tests for blueprint_parser.py.
"""

import os
import tempfile
import unittest

from blueprint_parser import BlueprintParseError, detect_cycles, parse_blueprint, parse_blueprint_content


class TestBlueprintParser(unittest.TestCase):

    def test_cycle_detection(self):
        acyclic = {
            "taskA": ["taskB", "taskC"],
            "taskB": ["taskD"],
            "taskC": [],
            "taskD": [],
        }
        self.assertEqual(detect_cycles(acyclic), [])

        cyclic_direct = {
            "taskA": ["taskB"],
            "taskB": ["taskA"],
        }
        cycle = detect_cycles(cyclic_direct)
        self.assertIn("taskA", cycle)
        self.assertIn("taskB", cycle)

        cyclic_long = {
            "taskA": ["taskB"],
            "taskB": ["taskC"],
            "taskC": ["taskD"],
            "taskD": ["taskB"],
        }
        cycle2 = detect_cycles(cyclic_long)
        self.assertTrue(len(cycle2) > 0)
        self.assertNotIn("taskA", cycle2)

    def test_valid_monolithic_schema_parsing(self):
        content = """
        [graph]
        entrypoint = orchestrator
        targets = ["scanner", "decision_tree", "parameter_tuner"]
        dependencies = {"scanner": [], "decision_tree": ["scanner"], "parameter_tuner": ["decision_tree"]}
        workspace_mode = incremental
        allow_partial_graph = false

        [compiler]
        profile_guided_optimization = enabled_strict
        tier_shifting_hotness_threshold = 100
        hotspot_loop_unroll_depth = 32
        aot_boundary_check_elimination = true
        vector_intrinsics_auto_generation = true
        pipeline_budget_seconds = 120.0
        max_memory_mb = 2048

        [cortex]
        consensus_protocol = raft_driven_mutation_lock
        mutation_entropy_clamp_threshold = 0.05
        total_cooperating_agents = 8
        heuristic_exploration_depth = 3
        execution_mode = lock_free_polling_wheel_realtime
        core_affinity_mask = 0xFFFF
        numa_node_locality_binding = true
        inter_core_ring_buffer_capacity = 262144
        """
        sections, deps = parse_blueprint_content(content)

        self.assertEqual(sections["graph"]["entrypoint"], "orchestrator")
        self.assertEqual(sections["compiler"]["pipeline_budget_seconds"], 120.0)
        self.assertEqual(sections["cortex"]["total_cooperating_agents"], 8)
        self.assertEqual(deps["decision_tree"], ["scanner"])
        self.assertEqual(deps["parameter_tuner"], ["decision_tree"])

    def test_missing_required_section_raises(self):
        bad_content = """
        [graph]
        targets = ["scanner"]
        dependencies = {"scanner": []}

        [compiler]
        profile_guided_optimization = enabled_strict
        """
        with self.assertRaises(BlueprintParseError) as exc:
            parse_blueprint_content(bad_content)
        self.assertIn("Missing required section", str(exc.exception))

    def test_fallback_reversion_gate(self):
        bad_content = """
        [graph]
        entrypoint = orchestrator
        targets = ["scanner", "decision_tree"]
        dependencies = {"scanner": ["decision_tree"], "decision_tree": ["scanner"]}

        [compiler]
        profile_guided_optimization = enabled_strict

        [cortex]
        consensus_protocol = raft_driven_mutation_lock
        """

        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(bad_content)
            bad_blueprint_path = handle.name

        try:
            context = parse_blueprint(bad_blueprint_path)
            self.assertEqual(context["workspace_status"], "reverted_fallback")
            self.assertIn("Invalid instruction loop", context["fallback_reason"])
            self.assertIn("profile_guided_optimization", context["active_optimizer_flags"])
            self.assertEqual(context["graph"]["workspace_mode"], "fallback_manifest")
        finally:
            if os.path.exists(bad_blueprint_path):
                os.remove(bad_blueprint_path)

    def test_parse_blueprint_returns_build_context(self):
        # The repo's blueprint.aero migrated to the living-blueprint schema, so the
        # legacy parser is exercised against a preserved legacy fixture instead.
        blueprint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures", "legacy_blueprint.aero"
        )
        context = parse_blueprint(blueprint_path)

        self.assertEqual(context["workspace_status"], "stable_active")
        self.assertEqual(context["graph"]["entrypoint"], "orchestrator")
        self.assertEqual(context["compilation_targets"], ["scanner", "decision_tree", "parameter_tuner", "compactor", "aero_translator"])
        self.assertIn("max_memory_mb", context["resource_metrics"])

    def test_minimal_toml_blueprint_no_crash(self):
        """A minimal TOML blueprint with [system] + [context_registry] must
        parse without crashes, inferring targets and using default compiler/cortex."""
        minimal = """
[system]
name = "colab-test-system"
strategy = "universal-engine"
ephemeral_code = true

[context_registry.core_logic]
path = "./app.py"
language = "python"
preserve_original_logic = true
"""
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(minimal)
            bp_path = handle.name

        try:
            context = parse_blueprint(bp_path)
            self.assertEqual(context["workspace_status"], "stable_active")
            self.assertEqual(context["blueprint_format"], "toml_native")
            self.assertIn("core_logic", context["compilation_targets"])
            self.assertIn("max_memory_mb", context["resource_metrics"])
            self.assertIn("active_optimizer_flags", context)
            self.assertIn("environment_targets", context)
        finally:
            if os.path.exists(bp_path):
                os.remove(bp_path)

    def test_optional_sections_default_injection(self):
        """When [graph], [compiler], and [cortex] are missing but [system] is
        present, _validate_sections should inject defaults instead of raising."""
        from blueprint_parser import _validate_sections
        sections = {"system": {"name": "test"}}
        deps = _validate_sections(sections)
        self.assertIn("graph", sections)
        self.assertIn("compiler", sections)
        self.assertIn("cortex", sections)
        self.assertIsInstance(deps, dict)

    def test_coerce_to_list_native_list(self):
        """Native Python lists pass through _coerce_to_list unchanged."""
        from blueprint_parser import _coerce_to_list
        result = _coerce_to_list("graph", "targets", ["a", "b"])
        self.assertEqual(result, ["a", "b"])

    def test_coerce_to_list_json_string(self):
        """A JSON array string is parsed into a real list."""
        from blueprint_parser import _coerce_to_list
        result = _coerce_to_list("graph", "targets", '["core_logic", "utils"]')
        self.assertEqual(result, ["core_logic", "utils"])

    def test_coerce_to_list_comma_separated(self):
        """Comma-separated strings are split into a list."""
        from blueprint_parser import _coerce_to_list
        result = _coerce_to_list("graph", "targets", "core_logic, utils")
        self.assertEqual(result, ["core_logic", "utils"])

    def test_toml_blueprint_stores_blueprint_dir(self):
        """The build context should contain blueprint_dir so ingestion can
        resolve relative paths against the blueprint's parent directory."""
        minimal = """
[system]
name = "dir-test"
strategy = "microkernel"

[context_registry.core_logic]
path = "./app.py"
language = "python"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(minimal)
            bp_path = handle.name

        try:
            context = parse_blueprint(bp_path)
            self.assertIn("blueprint_dir", context)
            # blueprint_dir should be the parent directory of the blueprint file.
            self.assertEqual(
                os.path.normpath(context["blueprint_dir"]),
                os.path.normpath(os.path.dirname(os.path.abspath(bp_path))),
            )
        finally:
            if os.path.exists(bp_path):
                os.remove(bp_path)

    def test_toml_context_registry_in_build_context(self):
        """context_registry entries from TOML blueprints should be dicts with
        path, language, and preserve_original_logic keys."""
        minimal = """
[system]
name = "ctx-test"

[context_registry.utils]
path = "./utils.py"
language = "python"
preserve_original_logic = true
"""
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as handle:
            handle.write(minimal)
            bp_path = handle.name

        try:
            context = parse_blueprint(bp_path)
            registry = context.get("context_registry", {})
            self.assertIn("utils", registry)
            self.assertEqual(registry["utils"]["path"], "./utils.py")
            self.assertEqual(registry["utils"]["language"], "python")
            self.assertTrue(registry["utils"]["preserve_original_logic"])
        finally:
            if os.path.exists(bp_path):
                os.remove(bp_path)


if __name__ == "__main__":
    unittest.main()
