# -*- coding: utf-8 -*-
"""Unit tests for src.runtime.feedback."""

import json
import os
import tempfile
import unittest

from src.runtime.feedback import RuntimeMetrics, RuntimeFeedback
from src.utils.json_parsing import extract_json


class TestRuntimeMetrics(unittest.TestCase):
    def test_defaults(self):
        m = RuntimeMetrics()
        self.assertFalse(m.success)
        self.assertEqual(m.wall_time, 0.0)
        self.assertIsNone(m.energy)
        self.assertIsNone(m.accuracy_error)

    def test_to_dict(self):
        m = RuntimeMetrics(success=True, wall_time=1.5, energy=0.8)
        d = m.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["wall_time"], 1.5)
        self.assertEqual(d["energy"], 0.8)


class TestRuntimeFeedbackConfig(unittest.TestCase):
    def test_defaults(self):
        rf = RuntimeFeedback()
        self.assertFalse(rf.enabled)
        self.assertFalse(rf.enable_feedback)
        self.assertEqual(rf.benchmark_command, "")
        self.assertEqual(rf.feedback_weight, 0.3)

    def test_enabled_config(self):
        config = {
            "runtime": {
                "enable_feedback": True,
                "benchmark_command": "echo hello",
                "metrics_to_collect": ["wall_time", "energy"],
                "feedback_weight": 0.5,
            }
        }
        rf = RuntimeFeedback(config=config)
        self.assertTrue(rf.enabled)
        self.assertEqual(rf.feedback_weight, 0.5)
        self.assertEqual(rf.metrics_to_collect, ["wall_time", "energy"])

    def test_disabled_without_command(self):
        config = {"runtime": {"enable_feedback": True, "benchmark_command": ""}}
        rf = RuntimeFeedback(config=config)
        self.assertFalse(rf.enabled)


class TestRunBenchmark(unittest.TestCase):
    def test_disabled_returns_unsuccessful(self):
        rf = RuntimeFeedback()
        result = rf.run_benchmark()
        self.assertFalse(result.success)
        self.assertIn("disabled", result.error)

    def test_runs_benchmark_command(self):
        config = {
            "runtime": {
                "enable_feedback": True,
                "benchmark_command": "echo wall_time=0.42",
            }
        }
        rf = RuntimeFeedback(config=config)
        result = rf.run_benchmark()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.wall_time, 0.42, places=2)

    def test_json_metrics_parsing(self):
        config = {
            "runtime": {
                "enable_feedback": True,
                "benchmark_command": 'echo \'{"wall_time": 1.5, "energy": 0.9}\'',
            }
        }
        rf = RuntimeFeedback(config=config)
        result = rf.run_benchmark()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.wall_time, 1.5)
        self.assertAlmostEqual(result.energy, 0.9)

    def test_timeout_handling(self):
        config = {
            "runtime": {
                "enable_feedback": True,
                "benchmark_command": "sleep 100",
            }
        }
        rf = RuntimeFeedback(config=config)
        result = rf.run_benchmark(timeout=0.1)
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_bad_command(self):
        config = {
            "runtime": {
                "enable_feedback": True,
                "benchmark_command": "/nonexistent/binary",
            }
        }
        rf = RuntimeFeedback(config=config)
        result = rf.run_benchmark()
        self.assertFalse(result.success)


class TestParseMetrics(unittest.TestCase):
    def test_json_parsing(self):
        rf = RuntimeFeedback()
        stdout = '{"wall_time": 2.5, "energy": 1.2, "accuracy_error": 0.001}'
        parsed = rf.parse_metrics(stdout)
        self.assertEqual(parsed["wall_time"], 2.5)
        self.assertEqual(parsed["energy"], 1.2)
        self.assertEqual(parsed["accuracy"], 0.001)

    def test_regex_parsing(self):
        rf = RuntimeFeedback()
        stdout = "wall_time = 3.7\nenergy: 0.5\n"
        parsed = rf.parse_metrics(stdout)
        self.assertEqual(parsed["wall_time"], 3.7)
        self.assertEqual(parsed["energy"], 0.5)

    def test_empty_output(self):
        rf = RuntimeFeedback()
        parsed = rf.parse_metrics("")
        self.assertEqual(parsed, {})


class TestCompareAccuracy(unittest.TestCase):
    def test_matching_numbers(self):
        rf = RuntimeFeedback()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1.0 2.0 3.0\n")
            ref_path = f.name
        try:
            error = rf.compare_accuracy("1.0 2.0 3.0", os.path.dirname(ref_path), ref_path)
            self.assertIsNotNone(error)
            self.assertAlmostEqual(error, 0.0)
        finally:
            os.unlink(ref_path)

    def test_mismatching_numbers(self):
        rf = RuntimeFeedback()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1.0 2.0 3.0\n")
            ref_path = f.name
        try:
            error = rf.compare_accuracy("1.1 2.0 3.0", os.path.dirname(ref_path), ref_path)
            self.assertIsNotNone(error)
            self.assertGreater(error, 0.0)
        finally:
            os.unlink(ref_path)

    def test_nonexistent_reference(self):
        rf = RuntimeFeedback()
        error = rf.compare_accuracy("1.0", "/tmp", "/nonexistent/file.txt")
        self.assertIsNone(error)


class TestFitnessBlending(unittest.TestCase):
    def test_to_fitness_objectives(self):
        config = {"runtime": {"metrics_to_collect": ["wall_time", "energy", "accuracy"]}}
        rf = RuntimeFeedback(config=config)
        metrics = RuntimeMetrics(wall_time=2.0, energy=0.5, accuracy_error=0.01)
        objectives = rf.to_fitness_objectives(metrics)
        self.assertEqual(objectives["runtime_wall_time"], 2.0)
        self.assertEqual(objectives["runtime_energy"], 0.5)
        self.assertEqual(objectives["runtime_accuracy_error"], 0.01)

    def test_blend_into_fitness(self):
        config = {"runtime": {"metrics_to_collect": ["wall_time"],
                              "feedback_weight": 0.5}}
        rf = RuntimeFeedback(config=config)
        metrics = RuntimeMetrics(wall_time=4.0)
        build_fitness = {"compile_time": 1.0, "binary_size": 500.0}
        blended = rf.blend_into_fitness(build_fitness, metrics)
        self.assertIn("compile_time", blended)
        self.assertIn("runtime_wall_time", blended)
        self.assertEqual(blended["runtime_wall_time"], 4.0 * 0.5)


class TestExtractJson(unittest.TestCase):
    def test_full_json(self):
        result = extract_json('{"a": 1}')
        self.assertEqual(result, {"a": 1})

    def test_embedded_json(self):
        result = extract_json('some prefix {"key": "val"} suffix')
        self.assertEqual(result, {"key": "val"})

    def test_invalid(self):
        result = extract_json("not json at all")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
