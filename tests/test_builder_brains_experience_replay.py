# -*- coding: utf-8 -*-
"""Unit tests for builder_brains.experience_replay."""

import json
import math
import os
import tempfile
import unittest

from builder_brains.experience_replay import ExperienceReplayBuffer


class TestSanitizeState(unittest.TestCase):
    def test_valid_state(self):
        state = {"learning_rate": 0.01, "epochs": 10, "label": "run1", "flag": True}
        result = ExperienceReplayBuffer._sanitize_state(state)
        self.assertEqual(result["learning_rate"], 0.01)
        self.assertEqual(result["epochs"], 10)
        self.assertEqual(result["label"], "run1")
        self.assertTrue(result["flag"])

    def test_non_dict_returns_empty(self):
        self.assertEqual(ExperienceReplayBuffer._sanitize_state("not a dict"), {})
        self.assertEqual(ExperienceReplayBuffer._sanitize_state(None), {})
        self.assertEqual(ExperienceReplayBuffer._sanitize_state(42), {})

    def test_filters_non_finite(self):
        state = {"inf": float("inf"), "nan": float("nan"), "ok": 1.0}
        result = ExperienceReplayBuffer._sanitize_state(state)
        self.assertNotIn("inf", result)
        self.assertNotIn("nan", result)
        self.assertIn("ok", result)

    def test_truncates_long_strings(self):
        state = {"long": "x" * 1000}
        result = ExperienceReplayBuffer._sanitize_state(state)
        self.assertLessEqual(len(result["long"]), 256)


class TestSanitizeMatrix(unittest.TestCase):
    def test_valid_matrix(self):
        matrix = {"alpha": 0.1, "beta": 2.5, "gamma": 0}
        result = ExperienceReplayBuffer._sanitize_matrix(matrix)
        self.assertEqual(result["alpha"], 0.1)
        self.assertEqual(result["beta"], 2.5)
        self.assertEqual(result["gamma"], 0.0)

    def test_filters_non_numeric(self):
        matrix = {"ok": 1.0, "bad": "text", "also_bad": None}
        result = ExperienceReplayBuffer._sanitize_matrix(matrix)
        self.assertIn("ok", result)
        self.assertNotIn("bad", result)
        self.assertNotIn("also_bad", result)

    def test_skips_booleans(self):
        matrix = {"flag": True, "value": 3.0}
        result = ExperienceReplayBuffer._sanitize_matrix(matrix)
        self.assertNotIn("flag", result)
        self.assertIn("value", result)

    def test_non_dict_returns_empty(self):
        self.assertEqual(ExperienceReplayBuffer._sanitize_matrix(None), {})
        self.assertEqual(ExperienceReplayBuffer._sanitize_matrix([1, 2]), {})


class TestCoerceResult(unittest.TestCase):
    def test_numeric(self):
        self.assertEqual(ExperienceReplayBuffer._coerce_result(3.14), 3.14)
        self.assertEqual(ExperienceReplayBuffer._coerce_result(42), 42.0)

    def test_bool(self):
        self.assertEqual(ExperienceReplayBuffer._coerce_result(True), 1.0)
        self.assertEqual(ExperienceReplayBuffer._coerce_result(False), 0.0)

    def test_non_finite(self):
        self.assertIsNone(ExperienceReplayBuffer._coerce_result(float("inf")))
        self.assertIsNone(ExperienceReplayBuffer._coerce_result(float("nan")))

    def test_non_numeric(self):
        self.assertIsNone(ExperienceReplayBuffer._coerce_result("text"))
        self.assertIsNone(ExperienceReplayBuffer._coerce_result(None))


class TestResultSortKey(unittest.TestCase):
    def test_numeric_result(self):
        self.assertEqual(ExperienceReplayBuffer._result_sort_key({"result": 5.0}), 5.0)

    def test_bool_result(self):
        self.assertEqual(ExperienceReplayBuffer._result_sort_key({"result": True}), 1.0)
        self.assertEqual(ExperienceReplayBuffer._result_sort_key({"result": False}), 0.0)

    def test_missing_result(self):
        self.assertEqual(ExperienceReplayBuffer._result_sort_key({}), float("-inf"))

    def test_non_finite_result(self):
        self.assertEqual(
            ExperienceReplayBuffer._result_sort_key({"result": float("nan")}),
            float("-inf"),
        )


class TestParseLedger(unittest.TestCase):
    def test_valid_json_array(self):
        data = json.dumps([{"state": {}, "action": "a", "result": 1.0}]).encode()
        result = ExperienceReplayBuffer._parse_ledger(data)
        self.assertEqual(len(result), 1)

    def test_empty_bytes(self):
        self.assertEqual(ExperienceReplayBuffer._parse_ledger(b""), [])
        self.assertEqual(ExperienceReplayBuffer._parse_ledger(b"   "), [])

    def test_corrupted_json(self):
        result = ExperienceReplayBuffer._parse_ledger(b"not json {{{")
        self.assertEqual(result, [])

    def test_non_array_json(self):
        result = ExperienceReplayBuffer._parse_ledger(b'{"key": "value"}')
        self.assertEqual(result, [])

    def test_filters_non_dict_entries(self):
        data = json.dumps([{"ok": True}, "bad", 42, {"also_ok": True}]).encode()
        result = ExperienceReplayBuffer._parse_ledger(data)
        self.assertEqual(len(result), 2)


class TestExperienceReplayBufferIntegration(unittest.TestCase):
    def test_create_and_record(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "replay.json")
            buf = ExperienceReplayBuffer(file_path=path, max_size=5)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(buf.file_path, path)
            self.assertEqual(buf.max_size, 5)

            # Record a trajectory
            buf.record_trajectory(
                state={"lr": 0.01},
                action="train",
                result=0.95,
                parameter_matrix={"alpha": 1.0},
            )
            entries = buf.top_trajectories(count=10)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["action"], "train")
            self.assertAlmostEqual(entries[0]["result"], 0.95)

    def test_bounded_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "replay.json")
            buf = ExperienceReplayBuffer(file_path=path, max_size=3)
            for i in range(10):
                buf.record_trajectory(state={}, action=f"a{i}", result=float(i))
            entries = buf.top_trajectories(count=10)
            self.assertLessEqual(len(entries), 3)
            # Top performers should be kept
            results = [e["result"] for e in entries]
            self.assertIn(9.0, results)

    def test_query_best(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "replay.json")
            buf = ExperienceReplayBuffer(file_path=path, max_size=10)
            for i in range(5):
                buf.record_trajectory(state={"x": i}, action="run", result=float(i))
            best = buf.top_trajectories(count=2)
            self.assertEqual(len(best), 2)
            self.assertEqual(best[0]["result"], 4.0)
            self.assertEqual(best[1]["result"], 3.0)


if __name__ == "__main__":
    unittest.main()
