"""Tests for the numerical library auto-tuner."""

from __future__ import annotations

import os
import tempfile
import unittest

from src.build.library_tuner import DetectedLibrary, LibraryTuner


class TestLibraryTuner(unittest.TestCase):
    def test_none_disables_detection(self):
        tuner = LibraryTuner(
            {"libraries": {"blas": "none", "lapack": "none", "mpi": False, "cuda": "none"}}
        )
        detected = tuner.detect_all()
        self.assertFalse(detected["blas"].found)
        self.assertFalse(detected["cuda"].found)
        self.assertEqual(tuner.linker_flags(detected), [])

    def test_graceful_when_nothing_installed(self):
        # auto everywhere -> on a host without the stack, all "not found".
        tuner = LibraryTuner(
            {"libraries": {"blas": "auto", "lapack": "auto", "mpi": True, "cuda": "auto"}}
        )
        detected = tuner.detect_all()
        self.assertIsInstance(detected["blas"], DetectedLibrary)
        # Must never raise; flags aggregate cleanly even when empty.
        self.assertIsInstance(tuner.linker_flags(detected), list)

    def test_env_var_blas_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lib"))
            os.makedirs(os.path.join(tmp, "include"))
            old = os.environ.get("OPENBLAS_HOME")
            os.environ["OPENBLAS_HOME"] = tmp
            try:
                tuner = LibraryTuner({"libraries": {"blas": "openblas"}})
                blas = tuner.detect_blas()
                self.assertTrue(blas.found)
                self.assertEqual(blas.flavor, "openblas")
                self.assertTrue(any(f.startswith("-L") for f in blas.linker_flags))
                self.assertIn("-lopenblas", blas.linker_flags)
                self.assertTrue(any(f.startswith("-I") for f in blas.cflags))
            finally:
                if old is None:
                    os.environ.pop("OPENBLAS_HOME", None)
                else:
                    os.environ["OPENBLAS_HOME"] = old

    def test_cuda_env_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lib64"))
            os.makedirs(os.path.join(tmp, "include"))
            old = os.environ.get("CUDA_HOME")
            os.environ["CUDA_HOME"] = tmp
            try:
                tuner = LibraryTuner({"libraries": {"cuda": "auto"}})
                cuda = tuner.detect_cuda()
                self.assertTrue(cuda.found)
                self.assertIn("-lcudart", cuda.linker_flags)
            finally:
                if old is None:
                    os.environ.pop("CUDA_HOME", None)
                else:
                    os.environ["CUDA_HOME"] = old

    def test_genome_space_offers_only_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "lib"))
            old = os.environ.get("OPENBLAS_HOME")
            os.environ["OPENBLAS_HOME"] = tmp
            try:
                tuner = LibraryTuner({"libraries": {"blas": "openblas", "lapack": "none"}})
                space = tuner.genome_space()
                self.assertIn("blas", space)
                self.assertIn("openblas", space["blas"])
                self.assertIn("none", space["blas"])
                # lapack was "none" -> single trivial value -> dropped from space.
                self.assertNotIn("lapack", space)
            finally:
                if old is None:
                    os.environ.pop("OPENBLAS_HOME", None)
                else:
                    os.environ["OPENBLAS_HOME"] = old


if __name__ == "__main__":
    unittest.main()
