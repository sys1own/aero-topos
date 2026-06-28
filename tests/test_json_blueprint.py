"""Tests for JSON blueprint parsing + INI backward compatibility."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from blueprint_parser import (
    BlueprintParseError,
    looks_like_json,
    parse_blueprint,
    parse_json_blueprint,
)

_REQUIRED = (
    "project", "analysis", "precision_shield", "hardware_profiling",
    "memoization", "context", "frameworks", "runtime", "validation", "physics",
)


def _full_blueprint() -> dict:
    bp = {section: {} for section in _REQUIRED}
    bp["context"] = {"sources": [{"path": "../shbt", "language": "python", "target_mapping": "src/shbt"}]}
    bp["precision_shield"] = {"shield_zones": [{"identifier": "z", "files": ["a.py"]}], "default_float": "quad"}
    return bp


class TestDetection(unittest.TestCase):
    def test_looks_like_json(self):
        self.assertTrue(looks_like_json('  \n {"a": 1}'))
        self.assertFalse(looks_like_json("[graph]\nkey = val"))


class TestJsonParsing(unittest.TestCase):
    def test_valid_blueprint(self):
        ctx = parse_json_blueprint(json.dumps(_full_blueprint()))
        self.assertEqual(ctx["workspace_status"], "stable_active")
        self.assertEqual(ctx["blueprint_format"], "json")
        # Unknown nested keys preserved.
        self.assertEqual(ctx["precision_shield"]["shield_zones"][0]["identifier"], "z")
        # Optional sections that were absent get defaults filled.
        self.assertIn("libraries", ctx)
        self.assertIn("gpu", ctx)

    def test_missing_required_sections(self):
        bp = _full_blueprint()
        del bp["context"]
        del bp["physics"]
        with self.assertRaises(BlueprintParseError) as exc:
            parse_json_blueprint(json.dumps(bp))
        msg = str(exc.exception)
        self.assertIn("context", msg)
        self.assertIn("physics", msg)

    def test_invalid_json(self):
        with self.assertRaises(BlueprintParseError):
            parse_json_blueprint("{not valid json")

    def test_non_object_json(self):
        with self.assertRaises(BlueprintParseError):
            parse_json_blueprint("[1, 2, 3]")

    def test_parse_blueprint_routes_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False) as fh:
            fh.write(json.dumps(_full_blueprint()))
            path = fh.name
        try:
            ctx = parse_blueprint(path)
            self.assertEqual(ctx["blueprint_format"], "json")
            self.assertEqual(len(ctx["context"]["sources"]), 1)
        finally:
            os.remove(path)


class TestIniBackwardCompat(unittest.TestCase):
    def test_ini_blueprint_still_parses(self):
        ini = """
[graph]
entrypoint = orchestrator
targets = ["scanner"]
dependencies = {"scanner": []}
[compiler]
optimization_level = "O3"
[cortex]
target_accuracy_floor = 0.99
"""
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False) as fh:
            fh.write(ini)
            path = fh.name
        try:
            ctx = parse_blueprint(path)
            self.assertEqual(ctx["workspace_status"], "stable_active")
            self.assertIsNone(ctx.get("blueprint_format"))  # INI path, not JSON
            self.assertEqual(ctx["compilation_targets"], ["scanner"])
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
