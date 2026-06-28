"""Tests for strict floating-point control (Precision Shield feature #3)."""

from __future__ import annotations

import unittest

from src.precision_shield.shield import PrecisionShield

_STRICT = {
    "precision_shield": {
        "floating_point_contract": "disallow",
        "fast_math_override": False,
        "ieee_compliance": "strict",
        "shield_zones": [
            {
                "identifier": "relaxed_zone",
                "files": ["src/fast.c"],
                "protection_level": "invariant_preservation",
                "tolerated_precision_loss": 1e-6,
                "fast_math_override": True,
                "floating_point_contract": "allow",
                "ieee_compliance": "relaxed",
            }
        ],
    }
}


class TestGlobalFloatingPointFlags(unittest.TestCase):
    def setUp(self):
        self.shield = PrecisionShield(_STRICT)

    def test_gcc_strict_flags(self):
        flags = self.shield.compiler_flags("gcc")
        self.assertIn("-fno-fast-math", flags)
        self.assertIn("-ffp-contract=off", flags)
        self.assertIn("-frounding-math", flags)

    def test_clang_path_via_compiler_path(self):
        flags = self.shield.compiler_flags("/usr/bin/clang++-17")
        self.assertIn("-fno-fast-math", flags)

    def test_rustc_flags(self):
        flags = self.shield.compiler_flags("rustc")
        self.assertIn("-Cllvm-args=-fp-contract=off", flags)

    def test_nvcc_flags(self):
        flags = self.shield.compiler_flags("nvcc")
        self.assertIn("--fmad=false", flags)
        self.assertIn("--prec-div=true", flags)
        self.assertNotIn("--use_fast_math", flags)

    def test_intel_flags(self):
        flags = self.shield.compiler_flags("icx")
        self.assertIn("-fp-model=strict", flags)

    def test_all_compiler_flags_keys(self):
        allflags = self.shield.all_compiler_flags()
        self.assertEqual(set(allflags), {"gcc", "clang", "intel", "rustc", "nvcc"})


class TestPerZoneOverride(unittest.TestCase):
    def setUp(self):
        self.shield = PrecisionShield(_STRICT)
        self.relaxed = self.shield.zones[0]

    def test_zone_relaxes_fast_math(self):
        flags = self.shield.compiler_flags("gcc", self.relaxed)
        self.assertIn("-ffast-math", flags)
        self.assertIn("-ffp-contract=fast", flags)
        self.assertNotIn("-fno-fast-math", flags)

    def test_zone_relaxes_nvcc(self):
        flags = self.shield.compiler_flags("nvcc", self.relaxed)
        self.assertIn("--use_fast_math", flags)
        self.assertIn("--fmad=true", flags)

    def test_resolve_policy(self):
        glob = self.shield.resolve_fp_policy()
        self.assertFalse(glob["fast_math"])
        self.assertEqual(glob["ieee_compliance"], "strict")
        zone = self.shield.resolve_fp_policy(self.relaxed)
        self.assertTrue(zone["fast_math"])
        self.assertEqual(zone["ieee_compliance"], "relaxed")


class TestDefaultsAreConservative(unittest.TestCase):
    def test_defaults_when_unspecified(self):
        shield = PrecisionShield({"precision_shield": {"shield_zones": []}})
        self.assertFalse(shield.fast_math_override)
        self.assertEqual(shield.floating_point_contract, "disallow")
        self.assertEqual(shield.ieee_compliance, "strict")
        self.assertIn("-fno-fast-math", shield.compiler_flags("gcc"))


if __name__ == "__main__":
    unittest.main()
