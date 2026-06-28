# -*- coding: utf-8 -*-
"""Tests for the living-blueprint loader (``src/blueprint/loader.py``).

The headline test loads the example schema straight out of ``README.md`` so the
documented DSL and the parser can never silently drift apart.
"""

import os
import re
import unittest

from src.blueprint import LivingBlueprint, load_blueprint
from src.blueprint.loader import Scaling

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _readme_living_blueprint_schema() -> str:
    """Return the first ```toml fenced block in README.md that defines [system]."""
    readme = os.path.join(_REPO_ROOT, "README.md")
    with open(readme, "r", encoding="utf-8") as handle:
        text = handle.read()
    for block in re.findall(r"```toml\n(.*?)```", text, re.DOTALL):
        if "[system]" in block:
            return block
    raise AssertionError("No [system] toml block found in README.md")


class TestReadmeExampleSchema(unittest.TestCase):
    """Every documented field must round-trip into the typed model."""

    @classmethod
    def setUpClass(cls):
        cls.bp = LivingBlueprint.from_str(_readme_living_blueprint_schema())

    def test_system_section(self):
        self.assertEqual(self.bp.system.name, "production-scale-polyglot-pipeline")
        self.assertEqual(self.bp.system.strategy, "universal-engine")
        self.assertTrue(self.bp.system.ephemeral_code)

    def test_context_registry(self):
        self.assertEqual(set(self.bp.context_registry), {"core_application"})

        app = self.bp.context_registry["core_application"]
        self.assertEqual(app.path, "./src/app_logic.py")
        self.assertEqual(app.language, "python")
        self.assertFalse(app.preserve_original_logic)

    def test_abstractions(self):
        self.assertEqual(len(self.bp.abstractions), 0)

    def test_scaling(self):
        self.assertEqual(self.bp.scaling.auto_split_threshold, 120)
        self.assertEqual(self.bp.scaling.max_module_complexity, 12)
        self.assertEqual(self.bp.scaling.hierarchy_depth, 4)

    def test_context_bridges(self):
        self.assertEqual(len(self.bp.context_bridges), 0)


class TestShippedBlueprint(unittest.TestCase):
    """The minimal TOML blueprint fixture must load cleanly."""

    def test_loads_fixture_blueprint(self):
        fixture = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures", "minimal_blueprint.aero"
        )
        bp = load_blueprint(fixture)
        self.assertEqual(bp.system.name, "aero-nova")
        self.assertEqual(bp.system.version, "0.1.0")
        self.assertEqual(bp.system.strategy, "microkernel")
        self.assertTrue(bp.system.ephemeral_code)
        # Empty [context_registry]/[abstractions] tables -> empty collections.
        self.assertEqual(bp.context_registry, {})
        self.assertEqual(bp.abstractions, [])
        self.assertEqual(bp.context_bridges, [])
        self.assertEqual(bp.scaling.auto_split_threshold, 1500)
        self.assertEqual(bp.scaling.max_module_complexity, 200)
        self.assertEqual(bp.scaling.hierarchy_depth, 4)

    def test_production_blueprint_passes_check(self):
        """The root blueprint.aero must pass strict DSL validation."""
        import blueprint_lang

        bp_path = os.path.join(_REPO_ROOT, "blueprint.aero")
        if not os.path.exists(bp_path):
            self.skipTest("blueprint.aero not present at repo root")
        with open(bp_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        error = blueprint_lang.check_source(source, filename=bp_path)
        self.assertIsNone(error, f"Production blueprint.aero is invalid:\n{error}")


class TestOptionalAndDefaults(unittest.TestCase):
    """Missing files and missing sections must degrade to defaults."""

    def test_missing_file_returns_defaults(self):
        bp = load_blueprint(os.path.join(_REPO_ROOT, "does-not-exist.aero"))
        self.assertIsInstance(bp, LivingBlueprint)
        self.assertEqual(bp.system.name, "")
        self.assertEqual(bp.context_registry, {})
        self.assertEqual(bp.abstractions, [])
        self.assertEqual(bp.context_bridges, [])
        self.assertEqual(bp.scaling, Scaling())

    def test_empty_document_uses_scaling_defaults(self):
        bp = LivingBlueprint.from_str("")
        self.assertEqual(bp.scaling.auto_split_threshold, 1500)
        self.assertEqual(bp.scaling.max_module_complexity, 200)
        self.assertEqual(bp.scaling.hierarchy_depth, 4)

    def test_partial_scaling_overrides_only_given_fields(self):
        bp = LivingBlueprint.from_str("[scaling]\nhierarchy_depth = 7\n")
        self.assertEqual(bp.scaling.hierarchy_depth, 7)
        self.assertEqual(bp.scaling.auto_split_threshold, 1500)
        self.assertEqual(bp.scaling.max_module_complexity, 200)


if __name__ == "__main__":
    unittest.main()
