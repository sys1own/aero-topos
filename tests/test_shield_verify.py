"""Tests for the Precision Shield mutation-equivalence guard (shield.verify)."""

from __future__ import annotations

import unittest

from src.precision_shield.shield import PrecisionShield, ShieldZone

_SHIELD = PrecisionShield({"precision_shield": {"shield_zones": []}})


class TestVerifyEquivalence(unittest.TestCase):
    def test_commutativity_and_distributivity(self):
        self.assertTrue(_SHIELD.verify_equivalence("a + b", "b + a"))
        self.assertTrue(_SHIELD.verify_equivalence("x*(y + z)", "x*y + x*z"))
        self.assertTrue(_SHIELD.verify_equivalence("2*x", "x + x"))

    def test_non_equivalent_rejected(self):
        self.assertFalse(_SHIELD.verify_equivalence("a + b", "a - b"))
        self.assertFalse(_SHIELD.verify_equivalence("a*b", "a + b"))

    def test_unsupported_expression_is_conservative(self):
        # Cannot model a function call -> conservatively reject.
        self.assertFalse(_SHIELD.verify_equivalence("foo(x)", "foo(x)"))

    def test_power_expansion(self):
        self.assertTrue(_SHIELD.verify_equivalence("x**2", "x*x"))


class TestVerifyZones(unittest.TestCase):
    def setUp(self):
        self.invariant = ShieldZone(
            identifier="solver", files=["s.py"],
            protection_level="invariant_preservation", tolerated_precision_loss=1e-9,
        )
        self.immutable = ShieldZone(
            identifier="const", files=["c.py"],
            protection_level="absolute_immutable", tolerated_precision_loss=0.0,
        )

    def test_invariant_allows_equivalent_rewrite(self):
        self.assertTrue(_SHIELD.verify("a + b", "b + a", self.invariant))
        self.assertFalse(_SHIELD.verify("a + b", "a * b", self.invariant))

    def test_immutable_blocks_any_change(self):
        self.assertTrue(_SHIELD.verify("3.141592653589793", "3.141592653589793", self.immutable))
        # Constant folding must be blocked even though the result is "equal".
        self.assertFalse(_SHIELD.verify("3.14 + 0.001592653", "3.141592653", self.immutable))
        # Reordering blocked for immutable zones even though algebraically equal.
        self.assertFalse(_SHIELD.verify("x + y", "y + x", self.immutable))

    def test_whitespace_insensitive_identity(self):
        self.assertTrue(_SHIELD.verify("a  +   b", "a+b", self.immutable))

    def test_default_zone_is_invariant(self):
        # No zone -> invariant_preservation semantics.
        self.assertTrue(_SHIELD.verify("a + b", "b + a"))


if __name__ == "__main__":
    unittest.main()
