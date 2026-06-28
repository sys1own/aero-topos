"""Tests for the Hardware Profiling Engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.hardware_profiling.profiler import (
    CacheLevel,
    HardwareProfile,
    HardwareProfiler,
    InstructionLatency,
    MemoryBandwidth,
    SIMDCapability,
)

_CONFIG = {
    "hardware_profiling": {
        "probe_at_compile_time": True,
        "profiling_strategy": "active-micro-benchmark",
        "benchmarks": {
            "cache_hierarchy": {"stride_sizes": [64, 128], "max_buffer_mb": 16, "iterations": 50},
            "simd_throughput": {"test_vectors": ["sse4.2", "avx2"], "test_sizes": [8, 16]},
            "memory_bandwidth": {"allocation_modes": ["sequential"], "block_sizes": [256]},
            "instruction_latency": {"operations": ["add", "mul"], "precision": ["f64"]},
        },
        "dynamic_recipe_generator": {
            "polyhedral_tiling_adjustment": True,
            "unroll_factor_limits": {"min": 2, "max": 16},
            "vectorization_analysis": True,
            "cache_blocking_strategy": "multi_level",
        },
        "profile_storage": "",
        "profile_retention_days": 7,
    }
}


class TestHardwareProfile(unittest.TestCase):
    def test_to_dict_and_from_dict(self):
        profile = HardwareProfile(
            cpu_model="TestCPU",
            cpu_count=4,
            arch="x86_64",
            total_memory_bytes=8 * 1024 ** 3,
            cache_hierarchy=[CacheLevel(level=1, size_bytes=32768, line_size_bytes=64, latency_ns=1.5)],
            simd_capabilities=[SIMDCapability(instruction_set="avx2", available=True, throughput_gflops=10.0)],
            memory_bandwidths=[MemoryBandwidth(mode="sequential", bandwidth_gbps=20.0, block_size=4096)],
            instruction_latencies=[InstructionLatency(operation="add", precision="f64", latency_ns=0.3)],
        )
        d = profile.to_dict()
        restored = HardwareProfile.from_dict(d)
        self.assertEqual(restored.cpu_model, "TestCPU")
        self.assertEqual(restored.cpu_count, 4)
        self.assertEqual(len(restored.cache_hierarchy), 1)
        self.assertEqual(restored.cache_hierarchy[0].level, 1)
        self.assertEqual(len(restored.simd_capabilities), 1)
        self.assertTrue(restored.simd_capabilities[0].available)


class TestHardwareProfiler(unittest.TestCase):
    def _profiler(self, storage_dir: str) -> HardwareProfiler:
        cfg = dict(_CONFIG)
        cfg["hardware_profiling"] = dict(cfg["hardware_profiling"])
        cfg["hardware_profiling"]["profile_storage"] = storage_dir
        return HardwareProfiler(cfg)

    def test_probe_returns_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self._profiler(tmp)
            profile = profiler.probe()
            self.assertIsInstance(profile, HardwareProfile)
            self.assertGreater(profile.cpu_count, 0)
            self.assertGreater(len(profile.cache_hierarchy), 0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self._profiler(tmp)
            profile = profiler.probe()
            profiler.save_profile(profile)
            loaded = profiler.load_profile()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.cpu_model, profile.cpu_model)

    def test_generate_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiler = self._profiler(tmp)
            profile = profiler.probe()
            recipe = profiler.generate_recipe(profile)
            self.assertIn("unroll_factor", recipe)
            self.assertIn("parallelism", recipe)
            self.assertIn("polyhedral_tiling", recipe)
            self.assertGreaterEqual(recipe["unroll_factor"], 2)


if __name__ == "__main__":
    unittest.main()
