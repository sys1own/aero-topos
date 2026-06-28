# -*- coding: utf-8 -*-
"""Unit tests for src.validation.validator."""

import json
import unittest

from src.validation.validator import ValidationCaseResult, ValidationReport, Validator
from src.utils.json_parsing import extract_json


class TestValidationCaseResult(unittest.TestCase):
    def test_to_dict(self):
        case = ValidationCaseResult(name="case1", passed=True, error=1e-9)
        d = case.to_dict()
        self.assertEqual(d["name"], "case1")
        self.assertTrue(d["passed"])
        self.assertEqual(d["error"], 1e-9)


class TestValidationReport(unittest.TestCase):
    def test_num_passed_failed(self):
        report = ValidationReport(
            passed=True,
            cases=[
                ValidationCaseResult(name="a", passed=True),
                ValidationCaseResult(name="b", passed=False),
                ValidationCaseResult(name="c", passed=True),
            ],
        )
        self.assertEqual(report.num_passed, 2)
        self.assertEqual(report.num_failed, 1)

    def test_to_dict(self):
        report = ValidationReport(passed=True, returncode=0)
        d = report.to_dict()
        self.assertTrue(d["passed"])
        self.assertEqual(d["returncode"], 0)
        self.assertEqual(d["num_passed"], 0)
        self.assertEqual(d["num_failed"], 0)


class TestValidatorConfig(unittest.TestCase):
    def test_defaults(self):
        v = Validator()
        self.assertFalse(v.enabled)
        self.assertEqual(v.tolerance, 1e-8)
        self.assertTrue(v.validation_required)
        self.assertEqual(v.suite, "")

    def test_custom_config(self):
        config = {
            "validation": {
                "suite": "nbody",
                "tolerance": 1e-6,
                "execution_command": "python validate.py",
                "validation_required": False,
            }
        }
        v = Validator(config=config)
        self.assertTrue(v.enabled)
        self.assertEqual(v.tolerance, 1e-6)
        self.assertFalse(v.is_gatekeeper)
        self.assertEqual(v.suite, "nbody")


class TestValidatorRun(unittest.TestCase):
    def test_disabled_validator_passes(self):
        v = Validator()
        report = v.run()
        self.assertTrue(report.passed)
        self.assertIn("disabled", report.summary)

    def test_json_output_parsing(self):
        json_output = json.dumps({
            "passed": True,
            "cases": [
                {"name": "gravity", "passed": True, "error": 1e-10},
                {"name": "collision", "passed": True, "error": 5e-9},
            ],
        })

        def runner(cmd, workdir):
            return (0, json_output, "")

        config = {"validation": {"execution_command": "run_tests"}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertTrue(report.passed)
        self.assertEqual(len(report.cases), 2)
        self.assertEqual(report.cases[0].name, "gravity")
        self.assertTrue(report.cases[0].passed)

    def test_json_with_error_threshold(self):
        json_output = json.dumps({
            "cases": [
                {"name": "case1", "error": 1e-5},
                {"name": "case2", "error": 1e-10},
            ],
        })

        def runner(cmd, workdir):
            return (0, json_output, "")

        config = {"validation": {"execution_command": "run", "tolerance": 1e-8}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertFalse(report.passed)  # case1 error exceeds tolerance
        self.assertFalse(report.cases[0].passed)
        self.assertTrue(report.cases[1].passed)

    def test_line_format_parsing(self):
        stdout = "gravity: PASS\ncollision: FAIL\norbital: OK\n"

        def runner(cmd, workdir):
            return (0, stdout, "")

        config = {"validation": {"execution_command": "validate"}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertFalse(report.passed)  # collision failed
        self.assertEqual(len(report.cases), 3)
        self.assertTrue(report.cases[0].passed)
        self.assertFalse(report.cases[1].passed)
        self.assertTrue(report.cases[2].passed)

    def test_error_format_parsing(self):
        stdout = "nbody error=1e-12\nmetric error=5e-7\n"

        def runner(cmd, workdir):
            return (0, stdout, "")

        config = {"validation": {"execution_command": "check", "tolerance": 1e-8}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertFalse(report.passed)  # metric error exceeds tolerance
        self.assertTrue(report.cases[0].passed)
        self.assertFalse(report.cases[1].passed)

    def test_exit_code_fallback(self):
        def runner(cmd, workdir):
            return (1, "no structured output", "something failed")

        config = {"validation": {"execution_command": "broken"}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertFalse(report.passed)
        self.assertEqual(report.returncode, 1)
        self.assertIn("something failed", report.error)

    def test_successful_exit_code_no_output(self):
        def runner(cmd, workdir):
            return (0, "", "")

        config = {"validation": {"execution_command": "check"}}
        v = Validator(config=config, runner=runner)
        report = v.run()
        self.assertTrue(report.passed)

    def test_bad_execution_command(self):
        config = {"validation": {"execution_command": "unterminated 'quote"}}
        v = Validator(config=config)
        report = v.run()
        self.assertFalse(report.passed)
        self.assertIn("bad execution_command", report.error)


class TestExtractJson(unittest.TestCase):
    def test_valid_json(self):
        result = extract_json('{"key": 1}')
        self.assertEqual(result, {"key": 1})

    def test_embedded_json(self):
        result = extract_json('prefix {"a": "b"} suffix')
        self.assertEqual(result, {"a": "b"})

    def test_invalid_json(self):
        result = extract_json("not json")
        self.assertIsNone(result)


class TestParseLines(unittest.TestCase):
    def test_pass_fail_format(self):
        v = Validator()
        cases = v._parse_lines("test1: PASS\ntest2: FAIL\n")
        self.assertEqual(len(cases), 2)
        self.assertTrue(cases[0].passed)
        self.assertFalse(cases[1].passed)

    def test_error_value_format(self):
        config = {"validation": {"tolerance": 1e-6}}
        v = Validator(config=config)
        cases = v._parse_lines("case1 error=1e-8\ncase2 error=1e-3\n")
        self.assertEqual(len(cases), 2)
        self.assertTrue(cases[0].passed)
        self.assertFalse(cases[1].passed)

    def test_empty_output(self):
        v = Validator()
        cases = v._parse_lines("")
        self.assertEqual(cases, [])


if __name__ == "__main__":
    unittest.main()
