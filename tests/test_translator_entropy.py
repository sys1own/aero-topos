# -*- coding: utf-8 -*-
"""Unit tests for translator.entropy_filter."""

import unittest

from translator.entropy_filter import (
    token_entropy,
    line_entropy,
    param_fingerprint,
    structural_diversity,
    check_entropy,
    detect_param_recycling,
)


class TestTokenEntropy(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(token_entropy(""), 0.0)

    def test_single_token(self):
        self.assertEqual(token_entropy("hello"), 0.0)

    def test_all_unique_tokens(self):
        text = "a b c d e f g h"
        ent = token_entropy(text)
        self.assertGreater(ent, 0.0)
        self.assertAlmostEqual(ent, 3.0, places=5)

    def test_repeated_tokens_low_entropy(self):
        text = "x x x x x x x x"
        self.assertEqual(token_entropy(text), 0.0)

    def test_mixed_tokens(self):
        text = "a a b b c c"
        ent = token_entropy(text)
        self.assertGreater(ent, 0.0)


class TestLineEntropy(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(line_entropy(""), 0.0)

    def test_unique_lines(self):
        text = "line1\nline2\nline3\n"
        ent = line_entropy(text)
        self.assertGreater(ent, 0.0)

    def test_duplicate_lines_zero_entropy(self):
        text = "same\nsame\nsame\n"
        self.assertEqual(line_entropy(text), 0.0)


class TestParamFingerprint(unittest.TestCase):
    def test_deterministic(self):
        text = 'foo("bar", 42, "baz")'
        fp1 = param_fingerprint(text)
        fp2 = param_fingerprint(text)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 16)

    def test_different_params_different_fingerprint(self):
        self.assertNotEqual(
            param_fingerprint('x = "hello" + 123'),
            param_fingerprint('x = "world" + 456'),
        )

    def test_same_params_same_fingerprint(self):
        self.assertEqual(
            param_fingerprint('call("a", 1)'),
            param_fingerprint('other("a", 1)'),
        )


class TestStructuralDiversity(unittest.TestCase):
    def test_no_task_blocks(self):
        self.assertEqual(structural_diversity("no tasks here"), 1.0)

    def test_unique_tasks(self):
        text = (
            "[task:a]\nop = print\ntext = hello\n\n"
            "[task:b]\nop = call\nfn = something\n"
        )
        sd = structural_diversity(text)
        self.assertEqual(sd, 1.0)

    def test_duplicate_tasks(self):
        text = (
            "[task:a]\nop = print\ntext = hello\n\n"
            "[task:b]\nop = print\ntext = hello\n"
        )
        sd = structural_diversity(text)
        self.assertLess(sd, 1.0)


class TestCheckEntropy(unittest.TestCase):
    def test_passes_diverse_recipe(self):
        text = "\n".join([f"[task:t{i}]\nop = step{i}\narg = val{i}" for i in range(20)])
        result = check_entropy(text)
        self.assertIn("passed", result)
        self.assertIn("token_entropy", result)
        self.assertIn("line_entropy", result)
        self.assertIn("structural_diversity", result)
        self.assertIn("param_fingerprint", result)
        self.assertIn("reasons", result)

    def test_fails_low_entropy(self):
        text = "x\n" * 100
        result = check_entropy(text)
        self.assertFalse(result["passed"])
        self.assertTrue(len(result["reasons"]) > 0)

    def test_custom_thresholds(self):
        text = "a b"
        result = check_entropy(text, min_token_entropy=0.0, min_line_entropy=0.0, min_diversity=0.0)
        self.assertTrue(result["passed"])


class TestDetectParamRecycling(unittest.TestCase):
    def test_recycled_params(self):
        old = 'call("arg1", 42)'
        new = 'other("arg1", 42)'
        self.assertTrue(detect_param_recycling(old, new))

    def test_no_recycling(self):
        old = 'call("arg1", 42)'
        new = 'call("arg2", 99)'
        self.assertFalse(detect_param_recycling(old, new))


if __name__ == "__main__":
    unittest.main()
