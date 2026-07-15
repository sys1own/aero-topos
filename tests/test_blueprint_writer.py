# -*- coding: utf-8 -*-
"""Tests for the AST-based ``self_host.aero`` TOML writer."""

import os
import tempfile
import unittest

import tomli as tomllib  # type: ignore

from evolve import write_params_to_blueprint


def _run_write(blueprint_text: str, genome: dict) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "self_host.aero")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(blueprint_text)
        write_params_to_blueprint(path, genome)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()


def _parse(text: str) -> dict:
    return tomllib.loads(text)


class TestBlueprintWriter(unittest.TestCase):
    def test_updates_inline_table_without_duplicate_keys(self):
        """Updating a value inside an inline table must not duplicate the table."""
        text = """[system]
name = "aero"

[cortex]
exploration_epsilon = 0.1
nsga2 = { population_size = 10, mutation_rate = 0.1, crossover_rate = 0.7 }
"""
        result = _run_write(
            text,
            {
                "target_accuracy_floor": 0.997,
                "cycles": 12,
                "population_size": 28,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
            },
        )
        doc = _parse(result)
        cortex = doc["cortex"]
        self.assertEqual(cortex["target_accuracy_floor"], 0.997)
        self.assertEqual(cortex["cycles"], 12)
        nsga2 = cortex["nsga2"]
        self.assertEqual(nsga2["population_size"], 28)
        self.assertEqual(nsga2["mutation_rate"], 0.15)
        self.assertEqual(nsga2["crossover_rate"], 0.8)
        # No duplicate nsga2 definitions -> a single table is emitted.
        self.assertEqual(result.count("nsga2"), 1)

    def test_updates_dotted_keys_in_section(self):
        """Dotted keys inside a section must be updated in-place."""
        text = """[cortex]
exploration_epsilon = 0.1
nsga2.mutation_rate = 0.1
nsga2.population_size = 10
"""
        result = _run_write(
            text,
            {"population_size": 28, "mutation_rate": 0.15},
        )
        doc = _parse(result)
        nsga2 = doc["cortex"]["nsga2"]
        self.assertEqual(nsga2["population_size"], 28)
        self.assertEqual(nsga2["mutation_rate"], 0.15)
        self.assertEqual(result.count("nsga2.mutation_rate"), 1)
        self.assertEqual(result.count("nsga2.population_size"), 1)

    def test_appends_missing_section_and_key(self):
        """Missing sections/keys are created without corrupting existing tables."""
        text = """[system]
name = "aero"
"""
        result = _run_write(
            text,
            {
                "population_size": 28,
                "mutation_rate": 0.15,
                "target_accuracy_floor": 0.997,
            },
        )
        doc = _parse(result)
        self.assertEqual(doc["system"]["name"], "aero")
        self.assertEqual(doc["cortex"]["nsga2"]["population_size"], 28)
        self.assertEqual(doc["cortex"]["nsga2"]["mutation_rate"], 0.15)
        self.assertEqual(doc["cortex"]["target_accuracy_floor"], 0.997)

    def test_preserves_comments_and_structure(self):
        """Comments and ordering should survive the update."""
        text = """[system]
name = "aero"

# Main tuning section
[cortex]
exploration_epsilon = 0.1
"""
        result = _run_write(
            text,
            {"cycles": 12, "population_size": 28},
        )
        self.assertIn("# Main tuning section", result)
        self.assertIn('[system]', result)
        self.assertIn('[cortex]', result)
        doc = _parse(result)
        self.assertEqual(doc["cortex"]["exploration_epsilon"], 0.1)
        self.assertEqual(doc["cortex"]["cycles"], 12)
        self.assertEqual(doc["cortex"]["nsga2"]["population_size"], 28)


if __name__ == "__main__":
    unittest.main()
