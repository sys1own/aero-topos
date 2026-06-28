# -*- coding: utf-8 -*-
"""Tests for ``aero_ui`` -- the clean terminal UI layer.

Covers:
* All phase-tag methods produce correct bracketed output
* Counters (compiled, skipped, failed) track correctly
* ``build_failure_report`` formats the Aero Build Failure header
* ``stats`` property
"""

from __future__ import annotations

import io
import unittest

from aero_ui import AeroUI


def _ui() -> tuple[AeroUI, io.StringIO]:
    buf = io.StringIO()
    return AeroUI(stream=buf), buf


class TestPhaseTags(unittest.TestCase):
    def test_parsing(self):
        ui, buf = _ui()
        ui.parsing("blueprint.aero")
        self.assertIn("Parsing", buf.getvalue())
        self.assertIn("blueprint.aero", buf.getvalue())

    def test_validating(self):
        ui, buf = _ui()
        ui.validating(5)
        self.assertIn("Validating", buf.getvalue())
        self.assertIn("5 targets", buf.getvalue())

    def test_validating_with_errors(self):
        ui, buf = _ui()
        ui.validating(3, error_count=2)
        output = buf.getvalue()
        self.assertIn("Validating", output)
        self.assertIn("2 error(s)", output)

    def test_resolving(self):
        ui, buf = _ui()
        ui.resolving(4, 3)
        output = buf.getvalue()
        self.assertIn("Resolving", output)
        self.assertIn("4 targets", output)
        self.assertIn("3 stages", output)

    def test_compiling(self):
        ui, buf = _ui()
        ui.compiling("core", "cpp")
        self.assertIn("Compiling", buf.getvalue())
        self.assertIn("core", buf.getvalue())
        self.assertIn("cpp", buf.getvalue())

    def test_compiled(self):
        ui, buf = _ui()
        ui.compiled("core", "cpp", "0.5s")
        self.assertIn("Compiled", buf.getvalue())
        self.assertEqual(ui._compiled, 1)

    def test_skipped(self):
        ui, buf = _ui()
        ui.skipped("optional_thing", "(optional)")
        self.assertIn("Skipped", buf.getvalue())
        self.assertEqual(ui._skipped, 1)

    def test_compile_error(self):
        ui, buf = _ui()
        ui.compile_error("broken", "syntax error")
        self.assertIn("Error", buf.getvalue())
        self.assertEqual(ui._failed, 1)
        self.assertTrue(ui.has_errors)

    def test_success(self):
        ui, buf = _ui()
        ui.compiled("a", "c")
        ui.compiled("b", "c")
        ui.success()
        output = buf.getvalue()
        self.assertIn("Success", output)
        self.assertIn("2 compiled", output)

    def test_failure(self):
        ui, buf = _ui()
        ui.compile_error("t", "err")
        ui.failure()
        output = buf.getvalue()
        self.assertIn("build failed", output)
        self.assertIn("1 error(s)", output)


class TestBuildFailureReport(unittest.TestCase):
    def test_report_contains_header_and_target(self):
        ui, buf = _ui()
        ui.build_failure_report("core", "some/file.cpp:10: error: stuff")
        output = buf.getvalue()
        self.assertIn("Aero Build Failure", output)
        self.assertIn("core", output)
        self.assertIn("some/file.cpp:10: error: stuff", output)


class TestStats(unittest.TestCase):
    def test_stats_shape(self):
        ui, _ = _ui()
        ui.compiled("a", "c")
        ui.skipped("b")
        ui.compile_error("c", "err")
        s = ui.stats
        self.assertEqual(s["compiled"], 1)
        self.assertEqual(s["skipped"], 1)
        self.assertEqual(s["failed"], 1)
        self.assertIn("elapsed", s)


if __name__ == "__main__":
    unittest.main()
