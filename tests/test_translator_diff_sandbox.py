# -*- coding: utf-8 -*-
"""Unit tests for translator.diff_sandbox."""

import json
import os
import tempfile
import unittest

from translator.diff_sandbox import (
    SandboxInput,
    ExecutionTrace,
    DiffResult,
    VerificationReport,
    _hash_value,
    execute_legacy,
    run_differential,
    apply_rollback,
)


class TestHashValue(unittest.TestCase):
    def test_deterministic(self):
        val = {"a": 1, "b": [2, 3]}
        self.assertEqual(_hash_value(val), _hash_value(val))

    def test_different_values(self):
        self.assertNotEqual(_hash_value(42), _hash_value(43))

    def test_handles_non_serializable(self):
        # default=str should handle non-serializable objects
        result = _hash_value(object())
        self.assertEqual(len(result), 64)  # sha256 hex


class TestExecuteLegacy(unittest.TestCase):
    def test_executes_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def add(a, b): return a + b\n")
            path = f.name
        try:
            ti = SandboxInput(args=(3, 4))
            trace = execute_legacy(path, "add", ti)
            self.assertTrue(trace.succeeded)
            self.assertEqual(trace.return_value, 7)
            self.assertNotEqual(trace.return_hash, "")
        finally:
            os.unlink(path)

    def test_function_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass\n")
            path = f.name
        try:
            ti = SandboxInput()
            trace = execute_legacy(path, "nonexistent", ti)
            self.assertFalse(trace.succeeded)
            self.assertIn("not found", trace.exception)
        finally:
            os.unlink(path)

    def test_exception_in_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken(): raise ValueError('oops')\n")
            path = f.name
        try:
            ti = SandboxInput()
            trace = execute_legacy(path, "broken", ti)
            self.assertFalse(trace.succeeded)
            self.assertIn("ValueError", trace.exception)
        finally:
            os.unlink(path)


class TestApplyRollback(unittest.TestCase):
    def test_no_rollback_needed(self):
        report = VerificationReport(
            function_name="good_fn",
            source_file="test.py",
            translatable=True,
            rolled_back=False,
        )
        log = apply_rollback(report)
        self.assertEqual(len(log), 1)
        self.assertIn("Not needed", log[0])

    def test_rollback_creates_flag(self):
        with tempfile.TemporaryDirectory() as td:
            report = VerificationReport(
                function_name="bad_fn",
                source_file="test.py",
                total_tests=2,
                passed=0,
                failed=2,
                translatable=False,
                reject_reason="2/2 failed",
                rolled_back=True,
                results=[
                    DiffResult(
                        input_label="case1",
                        match=False,
                        mismatch_detail="hash mismatch",
                    ),
                ],
            )
            log = apply_rollback(report, recipe_dir=td)
            # Should have written a flag file
            flag_dir = os.path.join(td, ".flags")
            self.assertTrue(os.path.isdir(flag_dir))
            flag_path = os.path.join(flag_dir, "bad_fn.non_translatable")
            self.assertTrue(os.path.exists(flag_path))
            with open(flag_path) as f:
                data = json.load(f)
            self.assertEqual(data["function"], "bad_fn")

    def test_rollback_deletes_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            recipe_path = os.path.join(td, "translated_test.txt")
            with open(recipe_path, "w") as f:
                f.write("recipe content")

            report = VerificationReport(
                function_name="bad_fn",
                source_file="test.py",
                rolled_back=True,
                reject_reason="mismatch",
                results=[],
            )
            log = apply_rollback(report, recipe_dir=td)
            self.assertFalse(os.path.exists(recipe_path))
            self.assertTrue(any("Deleted" in entry for entry in log))


if __name__ == "__main__":
    unittest.main()
