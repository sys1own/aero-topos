"""Tests for the Precision Shield."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.precision_shield.shield import PrecisionShield, ShieldZone, ValidationResult

_CONFIG = {
    "precision_shield": {
        "enforce_strict_invariants": True,
        "smt_validation_backend": "z3",
        "smt_timeout_ms": 3000,
        "fallback_on_smt_failure": "conservative",
        "shield_zones": [
            {
                "identifier": "crypto_core",
                "files": ["src/crypto/core.py"],
                "protection_level": "absolute_immutable",
                "tolerated_precision_loss": 0.0,
                "validation_rules": [
                    "no_constant_folding",
                    "preserve_original_order",
                    "no_associative_reordering",
                ],
            },
            {
                "identifier": "ml_layers",
                "files": ["src/ml/layers.py"],
                "protection_level": "invariant_preservation",
                "tolerated_precision_loss": 1e-9,
                "floating_point_precision": "64-bit",
                "validation_rules": [
                    "preserve_identity_operations",
                    "no_fused_operations",
                ],
            },
        ],
    }
}


class TestShieldZoneParsing(unittest.TestCase):
    def test_zones_parsed(self):
        shield = PrecisionShield(_CONFIG)
        self.assertEqual(len(shield.zones), 2)
        self.assertEqual(shield.zones[0].identifier, "crypto_core")
        self.assertEqual(shield.zones[1].protection_level, "invariant_preservation")


class TestProtectedFiles(unittest.TestCase):
    def test_get_protected_files(self):
        shield = PrecisionShield(_CONFIG)
        protected = shield.get_protected_files()
        self.assertIn("src/crypto/core.py", protected)
        self.assertIn("src/ml/layers.py", protected)

    def test_is_file_protected(self):
        shield = PrecisionShield(_CONFIG)
        self.assertTrue(shield.is_file_protected("src/crypto/core.py"))
        self.assertFalse(shield.is_file_protected("src/other.py"))

    def test_get_zone_for_file(self):
        shield = PrecisionShield(_CONFIG)
        zone = shield.get_zone_for_file("src/ml/layers.py")
        self.assertIsNotNone(zone)
        self.assertEqual(zone.identifier, "ml_layers")


class TestPythonValidation(unittest.TestCase):
    def test_constant_folding_violation(self):
        shield = PrecisionShield(_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crypto_dir = root / "src" / "crypto"
            crypto_dir.mkdir(parents=True)
            (crypto_dir / "core.py").write_text("x = 3.14 + 2.71\n")
            results = shield.validate_all(root)
            crypto_result = [r for r in results if r.zone_id == "crypto_core"][0]
            self.assertFalse(crypto_result.passed)
            self.assertTrue(any("constant-folding" in v for v in crypto_result.violations))

    def test_fused_op_violation(self):
        shield = PrecisionShield(_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ml_dir = root / "src" / "ml"
            ml_dir.mkdir(parents=True)
            (ml_dir / "layers.py").write_text("result = fma(a, b, c)\n")
            results = shield.validate_all(root)
            ml_result = [r for r in results if r.zone_id == "ml_layers"][0]
            self.assertFalse(ml_result.passed)
            self.assertTrue(any("fused-op" in v for v in ml_result.violations))

    def test_clean_code_passes(self):
        shield = PrecisionShield(_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ml_dir = root / "src" / "ml"
            ml_dir.mkdir(parents=True)
            (ml_dir / "layers.py").write_text("x = a + b\n")
            results = shield.validate_all(root)
            ml_result = [r for r in results if r.zone_id == "ml_layers"][0]
            self.assertTrue(ml_result.passed)


class TestSMTValidation(unittest.TestCase):
    def test_absolute_immutable_smt(self):
        shield = PrecisionShield(_CONFIG)
        zone = shield.zones[0]
        result = shield._run_smt_validation(zone)
        self.assertTrue(result["passed"])

    def test_invariant_preservation_smt(self):
        shield = PrecisionShield(_CONFIG)
        zone = shield.zones[1]
        result = shield._run_smt_validation(zone)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
