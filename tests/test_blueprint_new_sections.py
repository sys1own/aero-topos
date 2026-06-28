"""Tests for the new optional blueprint sections + backward compatibility."""

from __future__ import annotations

import os
import tempfile
import unittest

from blueprint_parser import (
    BlueprintParseError,
    normalize_optional_sections,
    parse_blueprint,
    parse_blueprint_content,
)

_LEGACY = """
[graph]
entrypoint = orchestrator
targets = ["scanner"]
dependencies = {"scanner": []}

[compiler]
optimization_level = "O3"

[cortex]
target_accuracy_floor = 0.99
"""

_FULL = _LEGACY + """
[libraries]
blas = "openblas"
lapack = "auto"
mpi = true
mpi_flavor = "mpich"
cuda = "12.3"

[distributed]
enabled = true
worker_nodes = ["build@n1", "ssh://build@n2"]
cache_sharing = "redis"

[gpu]
enabled = true
backend = "hip"
kernel_sources = ["src/kernels/*.cu"]

[physics]
dimensions = ["length", "time"]
symbolic_validation = true

[precision_shield]
floating_point_contract = "allow"
fast_math_override = true
ieee_compliance = "relaxed"
"""


class TestBackwardCompatibility(unittest.TestCase):
    def test_legacy_blueprint_gets_defaults(self):
        sections, _ = parse_blueprint_content(_LEGACY)
        opt = normalize_optional_sections(sections)
        self.assertEqual(opt["libraries"]["blas"], "none")
        self.assertFalse(opt["distributed"]["enabled"])
        self.assertFalse(opt["gpu"]["enabled"])
        self.assertFalse(opt["physics"]["symbolic_validation"])
        self.assertEqual(opt["precision_shield"]["ieee_compliance"], "strict")

    def test_legacy_parse_blueprint_still_stable(self):
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False) as fh:
            fh.write(_LEGACY)
            path = fh.name
        try:
            ctx = parse_blueprint(path)
            self.assertEqual(ctx["workspace_status"], "stable_active")
            # Optional sections present with safe defaults.
            self.assertIn("libraries", ctx)
            self.assertIn("gpu", ctx)
        finally:
            os.remove(path)


class TestNewSectionParsing(unittest.TestCase):
    def test_full_blueprint_values(self):
        sections, _ = parse_blueprint_content(_FULL)
        opt = normalize_optional_sections(sections)
        self.assertEqual(opt["libraries"]["blas"], "openblas")
        self.assertEqual(opt["libraries"]["mpi_flavor"], "mpich")
        self.assertEqual(opt["libraries"]["cuda"], "12.3")
        self.assertTrue(opt["distributed"]["enabled"])
        self.assertEqual(opt["distributed"]["cache_sharing"], "redis")
        self.assertEqual(opt["distributed"]["worker_nodes"], ["build@n1", "ssh://build@n2"])
        self.assertEqual(opt["gpu"]["backend"], "hip")
        self.assertEqual(opt["physics"]["dimensions"], ["length", "time"])
        self.assertTrue(opt["precision_shield"]["fast_math_override"])

    def test_parse_blueprint_surfaces_sections(self):
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False) as fh:
            fh.write(_FULL)
            path = fh.name
        try:
            ctx = parse_blueprint(path)
            self.assertEqual(ctx["gpu"]["backend"], "hip")
            self.assertTrue(ctx["distributed"]["enabled"])
        finally:
            os.remove(path)


class TestValidation(unittest.TestCase):
    def _expect_error(self, snippet: str):
        with self.assertRaises(BlueprintParseError):
            parse_blueprint_content(_LEGACY + snippet)

    def test_invalid_blas(self):
        self._expect_error('\n[libraries]\nblas = "badlib"\n')

    def test_invalid_gpu_backend(self):
        self._expect_error('\n[gpu]\nbackend = "metal"\n')

    def test_invalid_cache_sharing(self):
        self._expect_error('\n[distributed]\ncache_sharing = "ftp"\n')

    def test_invalid_ieee(self):
        self._expect_error('\n[precision_shield]\nieee_compliance = "loose"\n')

    def test_invalid_mpi_type(self):
        self._expect_error('\n[libraries]\nmpi = "yes-please"\n')


if __name__ == "__main__":
    unittest.main()
