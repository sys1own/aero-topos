# -*- coding: utf-8 -*-
"""Unit tests for builder_brains.scanner."""

import unittest

from builder_brains.scanner import (
    TokenProfiler,
    TokenNgramAnalyzer,
    _compile_pattern,
)


class TestCompilePattern(unittest.TestCase):
    def test_caches_pattern(self):
        pat1 = _compile_pattern(r"\bdef\b")
        pat2 = _compile_pattern(r"\bdef\b")
        self.assertIs(pat1, pat2)

    def test_returns_compiled_regex(self):
        pat = _compile_pattern(r"\d+")
        self.assertIsNotNone(pat.match("123"))


class TestTokenProfiler(unittest.TestCase):
    def test_profiles_source(self):
        source = """
def hello():
    return 42

class MyClass:
    pass

import os
"""
        profiler = TokenProfiler()
        counts = profiler.profile(source)
        self.assertIn("function_def", counts)
        self.assertIn("class_def", counts)
        self.assertIn("import_stmt", counts)
        self.assertEqual(counts["function_def"], 1)
        self.assertEqual(counts["class_def"], 1)
        self.assertGreaterEqual(counts["import_stmt"], 1)

    def test_counts_returns(self):
        source = "def foo():\n    return 1\ndef bar():\n    return 2\n"
        profiler = TokenProfiler()
        counts = profiler.profile(source)
        self.assertEqual(counts["return_stmt"], 2)

    def test_empty_source(self):
        profiler = TokenProfiler()
        counts = profiler.profile("")
        for v in counts.values():
            self.assertEqual(v, 0)

    def test_profile_with_positions(self):
        source = "def foo(): pass\ndef bar(): pass\n"
        profiler = TokenProfiler()
        positions = profiler.profile_with_positions(source)
        self.assertIn("function_def", positions)
        self.assertEqual(len(positions["function_def"]), 2)
        for start, end in positions["function_def"]:
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertLess(start, end)


class TestTokenNgramAnalyzer(unittest.TestCase):
    def test_extract_ngrams(self):
        analyzer = TokenNgramAnalyzer(window=3)
        tokens = ["def", "name", "return", "def", "name", "return"]
        ngrams = analyzer.extract_ngrams(tokens)
        self.assertIsInstance(ngrams, dict)
        self.assertGreater(len(ngrams), 0)

    def test_single_token(self):
        analyzer = TokenNgramAnalyzer(window=3)
        ngrams = analyzer.extract_ngrams(["single"])
        self.assertEqual(len(ngrams), 0)

    def test_window_size_1(self):
        analyzer = TokenNgramAnalyzer(window=1)
        tokens = ["a", "b", "c"]
        ngrams = analyzer.extract_ngrams(tokens)
        self.assertEqual(len(ngrams), 3)

    def test_empty_sequence(self):
        analyzer = TokenNgramAnalyzer(window=4)
        ngrams = analyzer.extract_ngrams([])
        self.assertEqual(len(ngrams), 0)


if __name__ == "__main__":
    unittest.main()
