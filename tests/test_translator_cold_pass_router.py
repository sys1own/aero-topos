# -*- coding: utf-8 -*-
"""Unit tests for translator.cold_pass_router."""

import os
import tempfile
import unittest

from translator.cold_pass_router import (
    ColdPathReason,
    FunctionRouting,
    RoutingAnalysis,
    _THREADING_MODULES,
    _HARDWARE_MODULES,
    _GPU_MODULES,
    _UNSAFE_PATTERNS,
    _RS_MEMORY_PATTERNS,
    _RS_THREADING_PATTERNS,
    _RS_LIFETIME_PATTERNS,
    analyze_routing,
)


class TestPatternConstants(unittest.TestCase):
    def test_threading_modules(self):
        self.assertIn("threading", _THREADING_MODULES)
        self.assertIn("asyncio", _THREADING_MODULES)
        self.assertIn("multiprocessing", _THREADING_MODULES)

    def test_hardware_modules(self):
        self.assertIn("ctypes", _HARDWARE_MODULES)
        self.assertIn("cffi", _HARDWARE_MODULES)

    def test_gpu_modules(self):
        self.assertIn("cuda", _GPU_MODULES)
        self.assertIn("torch", _GPU_MODULES)
        self.assertIn("numba", _GPU_MODULES)


class TestUnsafePatterns(unittest.TestCase):
    def test_detects_ctypes(self):
        self.assertIsNotNone(_UNSAFE_PATTERNS.search("ctypes.CDLL"))
        self.assertIsNotNone(_UNSAFE_PATTERNS.search("ctypes.cdll"))

    def test_detects_cuda(self):
        self.assertIsNotNone(_UNSAFE_PATTERNS.search("tensor.cuda()"))
        self.assertIsNotNone(_UNSAFE_PATTERNS.search("numba.jit"))

    def test_detects_mmap(self):
        self.assertIsNotNone(_UNSAFE_PATTERNS.search("mmap.mmap"))


class TestRustPatterns(unittest.TestCase):
    def test_memory_patterns(self):
        self.assertIsNotNone(_RS_MEMORY_PATTERNS.search("unsafe fn foo()"))
        self.assertIsNotNone(_RS_MEMORY_PATTERNS.search("*mut u8"))
        self.assertIsNotNone(_RS_MEMORY_PATTERNS.search("std::mem::transmute"))
        self.assertIsNotNone(_RS_MEMORY_PATTERNS.search("ManuallyDrop"))

    def test_threading_patterns(self):
        self.assertIsNotNone(_RS_THREADING_PATTERNS.search("Mutex<T>"))
        self.assertIsNotNone(_RS_THREADING_PATTERNS.search("thread::spawn"))
        self.assertIsNotNone(_RS_THREADING_PATTERNS.search("async fn handler()"))
        self.assertIsNotNone(_RS_THREADING_PATTERNS.search("rayon::"))

    def test_lifetime_patterns(self):
        self.assertIsNotNone(_RS_LIFETIME_PATTERNS.search("<'a>"))
        self.assertIsNotNone(_RS_LIFETIME_PATTERNS.search("&'a str"))


class TestAnalyzeRouting(unittest.TestCase):
    def test_pure_function_is_hot(self):
        source = """
def compute(x, y):
    return x + y

def add_one(n):
    return n + 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            analysis = analyze_routing(path)
            self.assertIsInstance(analysis, RoutingAnalysis)
            self.assertEqual(analysis.cold_count, 0)
            self.assertEqual(analysis.hot_count, 2)
            for fr in analysis.functions:
                self.assertFalse(fr.is_cold_passthrough)
        finally:
            os.unlink(path)

    def test_threading_function_is_cold(self):
        source = """
import threading

def parallel_work():
    t = threading.Thread(target=lambda: None)
    t.start()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            analysis = analyze_routing(path)
            cold_fns = [fr for fr in analysis.functions if fr.is_cold_passthrough]
            self.assertGreater(len(cold_fns), 0)
        finally:
            os.unlink(path)

    def test_gpu_function_is_cold(self):
        source = """
import torch

def train_model(data):
    tensor = torch.tensor(data)
    return tensor.cuda()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            analysis = analyze_routing(path)
            cold_fns = [fr for fr in analysis.functions if fr.is_cold_passthrough]
            self.assertGreater(len(cold_fns), 0)
        finally:
            os.unlink(path)

    def test_syntax_error_returns_empty(self):
        source = "def broken(\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            analysis = analyze_routing(path)
            self.assertEqual(analysis.functions, [])
        finally:
            os.unlink(path)


class TestDataclasses(unittest.TestCase):
    def test_cold_path_reason(self):
        r = ColdPathReason(category="threading", detail="Thread creation", lineno=5)
        self.assertEqual(r.category, "threading")

    def test_function_routing(self):
        fr = FunctionRouting(name="foo", lineno=1, is_cold_passthrough=True)
        self.assertTrue(fr.is_cold_passthrough)


if __name__ == "__main__":
    unittest.main()
