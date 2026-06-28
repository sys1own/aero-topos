"""End-to-end acceptance tests mapped to the task's acceptance criteria."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from blueprint_parser import parse_blueprint
from src.hardware_profiling.profiler import HardwareProfiler
from src.precision_shield.shield import PrecisionShield

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE = _REPO_ROOT / "examples" / "physics_simulator"


class TestMockSimulatorComputesPi(unittest.TestCase):
    """The mock simulator builds/runs and computes pi to double precision."""

    def test_orchestrator_validates_pi(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.python.orchestrator"],
            cwd=str(_EXAMPLE),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"matches f64 : True", result.stdout)


class TestPrecisionShieldProtectsConstant(unittest.TestCase):
    """The shield prevents over-optimisation (constant folding) of a constant."""

    def _shield(self):
        return PrecisionShield(
            {
                "precision_shield": {
                    "fallback_on_smt_failure": "conservative",
                    "shield_zones": [
                        {
                            "identifier": "fundamental_constants",
                            "files": ["constants.py"],
                            "protection_level": "absolute_immutable",
                            "tolerated_precision_loss": 0.0,
                            "validation_rules": ["no_constant_folding", "no_associative_reordering"],
                        }
                    ],
                }
            }
        )

    def test_constant_folding_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A compiler folding 3.14 + 0.0015926 would mutate the constant.
            (root / "constants.py").write_text("PI = 3.14 + 0.0015926535\n")
            results = self._shield().validate_all(root)
            zone = [r for r in results if r.zone_id == "fundamental_constants"][0]
            self.assertFalse(zone.passed)
            self.assertTrue(any("constant-folding" in v for v in zone.violations))

    def test_clean_constant_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "constants.py").write_text("PI = 3.141592653589793\n")
            results = self._shield().validate_all(root)
            zone = [r for r in results if r.zone_id == "fundamental_constants"][0]
            self.assertTrue(zone.passed)


class TestHardwareProfilingDetects(unittest.TestCase):
    """Hardware profiling detects SIMD and numerical-library paths."""

    def test_simd_and_library_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "blas", "lib"))
            os.makedirs(os.path.join(tmp, "blas", "include"))
            old = os.environ.get("OPENBLAS_HOME")
            os.environ["OPENBLAS_HOME"] = os.path.join(tmp, "blas")
            try:
                config = {
                    "hardware_profiling": {
                        "benchmarks": {
                            "cache_hierarchy": {"stride_sizes": [64, 128], "iterations": 20},
                            "simd_throughput": {"test_vectors": ["sse4.2", "avx2"]},
                            "memory_bandwidth": {"allocation_modes": ["sequential"], "block_sizes": [256]},
                            "instruction_latency": {"operations": ["add"], "precision": ["f64"]},
                        },
                        "dynamic_recipe_generator": {"unroll_factor_limits": {"min": 2, "max": 16}},
                        "profile_storage": tmp,
                    },
                    "libraries": {"blas": "openblas", "lapack": "none", "mpi": False, "cuda": "none"},
                }
                profiler = HardwareProfiler(config)
                profile = profiler.probe()
                # SIMD probes ran.
                self.assertGreater(len(profile.simd_capabilities), 0)
                # Library path detected via the fake OPENBLAS_HOME.
                self.assertTrue(profile.libraries["detected"]["blas"]["found"])
                recipe = profiler.generate_recipe(profile)
                self.assertTrue(any("openblas" in f for f in recipe.get("linker_flags", [])))
            finally:
                if old is None:
                    os.environ.pop("OPENBLAS_HOME", None)
                else:
                    os.environ["OPENBLAS_HOME"] = old


class TestExampleBlueprint(unittest.TestCase):
    """The example physics blueprint parses with every new section populated."""

    def test_example_blueprint_parses(self):
        ctx = parse_blueprint(str(_EXAMPLE / "blueprint.aero"))
        self.assertEqual(ctx["workspace_status"], "stable_active")
        self.assertEqual(ctx["libraries"]["blas"], "openblas")
        self.assertTrue(ctx["distributed"]["enabled"])
        self.assertTrue(ctx["gpu"]["enabled"])
        self.assertEqual(ctx["gpu"]["backend"], "cuda")
        self.assertTrue(ctx["physics"]["symbolic_validation"])
        self.assertEqual(ctx["precision_shield"]["ieee_compliance"], "strict")


if __name__ == "__main__":
    unittest.main()
