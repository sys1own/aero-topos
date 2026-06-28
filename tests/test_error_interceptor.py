# -*- coding: utf-8 -*-
"""Tests for ``error_interceptor`` -- stderr normalisation, result handling, guarded main.

Covers:
* :func:`normalise_stderr` -- ANSI stripping, line limiting
* :func:`handle_compile_results` -- routing successes and failures
* :func:`guarded_main` -- exception guard, clean exit on crash
"""

from __future__ import annotations

import io
import unittest

from aero_ui import AeroUI
from error_interceptor import guarded_main, handle_compile_results, normalise_stderr
from src.build.compilers import CompileResult


def _ui() -> tuple[AeroUI, io.StringIO]:
    buf = io.StringIO()
    return AeroUI(stream=buf), buf


class TestNormaliseStderr(unittest.TestCase):
    def test_strips_ansi(self):
        raw = "\033[31merror:\033[0m something"
        self.assertEqual(normalise_stderr(raw), "error: something")

    def test_empty_returns_placeholder(self):
        self.assertEqual(normalise_stderr(""), "(no output from compiler)")
        self.assertEqual(normalise_stderr("   \n\n  "), "(no output from compiler)")

    def test_limits_long_output(self):
        stderr = "\n".join(f"line {i}" for i in range(100))
        result = normalise_stderr(stderr, max_lines=10)
        lines = result.splitlines()
        self.assertEqual(len(lines), 11)  # 10 kept + 1 "omitted" line
        self.assertIn("90 more lines omitted", lines[-1])

    def test_short_output_unchanged(self):
        stderr = "single error line"
        self.assertEqual(normalise_stderr(stderr), "single error line")


class TestHandleCompileResults(unittest.TestCase):
    def test_all_success(self):
        ui, buf = _ui()
        results = [
            CompileResult(target_name="a", success=True, command=["gcc", "a.c"]),
            CompileResult(target_name="b", success=True, command=["gcc", "b.c"]),
        ]
        self.assertEqual(handle_compile_results(results, ui), 0)

    def test_one_failure(self):
        ui, buf = _ui()
        results = [
            CompileResult(target_name="a", success=True, command=["gcc", "a.c"]),
            CompileResult(
                target_name="b",
                success=False,
                command=["gcc", "b.c"],
                stderr="b.c:1: error: undeclared",
                return_code=1,
            ),
        ]
        exit_code = handle_compile_results(results, ui)
        self.assertEqual(exit_code, 1)
        output = buf.getvalue()
        self.assertIn("Aero Build Failure", output)
        self.assertIn("b", output)

    def test_empty_results(self):
        ui, _ = _ui()
        self.assertEqual(handle_compile_results([], ui), 0)


class TestGuardedMain(unittest.TestCase):
    def test_clean_exit(self):
        self.assertEqual(guarded_main(lambda: 0), 0)
        self.assertEqual(guarded_main(lambda: 42), 42)

    def test_catches_runtime_error(self):
        buf = io.StringIO()

        def boom():
            raise RuntimeError("kaboom")

        code = guarded_main(boom, stream=buf)
        self.assertEqual(code, 1)
        output = buf.getvalue()
        self.assertIn("Aero Build Failure", output)
        self.assertIn("RuntimeError", output)
        self.assertIn("kaboom", output)
        # Must NOT contain a traceback
        self.assertNotIn("Traceback", output)

    def test_catches_keyboard_interrupt(self):
        buf = io.StringIO()

        def interrupted():
            raise KeyboardInterrupt

        code = guarded_main(interrupted, stream=buf)
        self.assertEqual(code, 130)

    def test_catches_system_exit(self):
        code = guarded_main(lambda: (_ for _ in ()).throw(SystemExit(3)))
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
