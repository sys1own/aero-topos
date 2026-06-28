"""Tests for the GPU offloading pipeline (feature #5)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.build.gpu_pipeline import GPUPipeline


def _make_kernels(root: Path) -> None:
    (root / "src" / "kernels").mkdir(parents=True)
    (root / "src" / "kernels" / "a.cu").write_text("__global__ void a(){}\n")
    (root / "src" / "kernels" / "b.cu").write_text("__global__ void b(){}\n")


class TestGPUPipeline(unittest.TestCase):
    def _cuda_cfg(self, enabled=True):
        return {"gpu": {"enabled": enabled, "backend": "cuda", "kernel_sources": ["src/kernels/*.cu"]}}

    def test_discovery_and_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_kernels(root)
            pipe = GPUPipeline(self._cuda_cfg())
            plan = pipe.plan(root)
            self.assertEqual(plan["kernel_count"], 2)
            self.assertEqual(len(plan["compile_steps"]), 2)
            self.assertIn("-lcudart", plan["link_flags"])

    def test_extra_flags_threaded_into_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_kernels(root)
            pipe = GPUPipeline(self._cuda_cfg())
            plan = pipe.plan(root, extra_flags=["--fmad=false"])
            self.assertIn("--fmad=false", plan["compile_steps"][0]["command"])
            self.assertEqual(plan["compile_steps"][0]["command"][0], "nvcc")

    def test_compile_skips_without_toolchain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_kernels(root)
            pipe = GPUPipeline(self._cuda_cfg())
            results = pipe.compile_kernels(root)
            # nvcc not installed in CI -> graceful skip, never raise.
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.status in ("skipped", "compiled") for r in results))

    def test_disabled_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_kernels(root)
            pipe = GPUPipeline(self._cuda_cfg(enabled=False))
            self.assertFalse(pipe.available())
            results = pipe.compile_kernels(root)
            self.assertTrue(all(r.status == "skipped" for r in results))

    def test_backend_link_flags(self):
        self.assertEqual(GPUPipeline({"gpu": {"backend": "cuda"}}).link_flags(), ["-lcudart"])
        self.assertEqual(GPUPipeline({"gpu": {"backend": "hip"}}).link_flags(), ["-lamdhip64"])
        self.assertEqual(GPUPipeline({"gpu": {"backend": "opencl"}}).link_flags(), ["-lOpenCL"])

    def test_hip_compiler_name(self):
        pipe = GPUPipeline({"gpu": {"enabled": True, "backend": "hip", "kernel_sources": ["*.cu"]}})
        self.assertEqual(pipe.compiler_name, "hipcc")


if __name__ == "__main__":
    unittest.main()
