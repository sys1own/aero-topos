"""Tests for the Aero-Calculus spacetime ledger (``core/spacetime_ledger.py``)."""

from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal

from core.spacetime_ledger import (
    AnomalyClosureError,
    BlockUniverseLedger,
    CoordinateVector,
    FrozenCoordinateError,
    RigidityVerifier,
)


# ---------------------------------------------------------------------------
# Coordinate vector + precision
# ---------------------------------------------------------------------------
class TestCoordinateVector(unittest.TestCase):
    def test_distance_is_exact_at_high_precision(self):
        # 3-4-5 triple -> exact distance 5, no float drift.
        a = CoordinateVector("0", "0", "0", 0)
        b = CoordinateVector("3", "4", "0", 0)
        self.assertEqual(a.distance_to(b), Decimal(5))

    def test_no_rounding_degradation_at_cosmological_scale(self):
        # A 10^-140 displacement survives alongside a unit term: 150-digit
        # precision means the tiny component is NOT rounded away.
        tiny = "1e-140"
        a = CoordinateVector("1", "0", "0", 0)
        b = CoordinateVector(str(Decimal("1") + Decimal(tiny)), "0", "0", 0)
        d = a.distance_to(b)
        self.assertEqual(d, Decimal(tiny))
        self.assertNotEqual(d, Decimal(0))

    def test_translation_preserves_relative_geometry(self):
        a = CoordinateVector("1", "2", "3", 0)
        b = CoordinateVector("4", "6", "3", 0)
        before = a.distance_to(b)
        am = a.translate(Decimal("1e-120"), Decimal("1e-120"), Decimal("1e-120"))
        bm = b.translate(Decimal("1e-120"), Decimal("1e-120"), Decimal("1e-120"))
        self.assertEqual(am.distance_to(bm), before)


# ---------------------------------------------------------------------------
# Frozen coordinate enforcement
# ---------------------------------------------------------------------------
class TestFrozenCoordinate(unittest.TestCase):
    def test_mutation_after_freeze_raises(self):
        c = CoordinateVector("1", "2", "3", 7)
        c.freeze()
        with self.assertRaises(FrozenCoordinateError):
            c.x = Decimal("9")
        with self.assertRaises(FrozenCoordinateError):
            c.t_causal = 99

    def test_unfrozen_coordinate_is_mutable(self):
        c = CoordinateVector("1", "2", "3", 7)
        c.x = Decimal("9")
        self.assertEqual(c.x, Decimal("9"))


# ---------------------------------------------------------------------------
# Block universe ledger
# ---------------------------------------------------------------------------
class TestBlockUniverseLedger(unittest.TestCase):
    def _ledger(self) -> BlockUniverseLedger:
        tmp = tempfile.mkdtemp()
        return BlockUniverseLedger(os.path.join(tmp, "context.aero"))

    def test_append_returns_sequential_t_causal(self):
        ledger = self._ledger()
        self.assertEqual(ledger.append_transaction({"a": 1}), 0)
        self.assertEqual(ledger.append_transaction({"b": 2}), 1)
        self.assertEqual(ledger.append_transaction({"c": 3}), 2)

    def test_hash_chain_integrity(self):
        ledger = self._ledger()
        ledger.append_transaction({"a": 1})
        ledger.append_transaction({"b": 2})
        self.assertTrue(ledger.verify_integrity())

    def test_tampering_breaks_integrity(self):
        ledger = self._ledger()
        ledger.append_transaction({"a": 1})
        ledger.append_transaction({"b": 2})
        # Mutate a past payload directly -> hash no longer matches.
        ledger._chain[0]["payload"]["a"] = 999
        self.assertFalse(ledger.verify_integrity())

    def test_committed_block_is_immutable(self):
        ledger = self._ledger()
        ledger.append_transaction({"a": 1})
        with self.assertRaises(FrozenCoordinateError):
            ledger.overwrite_transaction(0, {"a": 2})

    def test_get_transaction_is_read_only(self):
        ledger = self._ledger()
        ledger.append_transaction({"a": 1})
        view = ledger.get_transaction(0)
        with self.assertRaises(TypeError):
            view["payload"] = {"a": 2}

    def test_annotate_node_freezes_coordinate(self):
        ledger = self._ledger()

        class _N:
            node_id = "n1"

        node = _N()
        coord = CoordinateVector("1", "2", "3", -1)
        t = ledger.annotate_node(node, coord)
        self.assertEqual(t, 0)
        self.assertEqual(coord.t_causal, 0)
        self.assertTrue(coord.frozen)
        self.assertIs(node.coordinate, coord)
        # Frozen past coordinate index cannot be modified.
        with self.assertRaises(FrozenCoordinateError):
            coord.t_causal = 5


# ---------------------------------------------------------------------------
# Rigidity verifier
# ---------------------------------------------------------------------------
class TestRigidityVerifier(unittest.TestCase):
    def test_kernel_constants(self):
        v = RigidityVerifier()
        self.assertEqual(v.kernel_dimension, 26)
        self.assertEqual(v.kernel_model, (26, 8, 312))
        self.assertEqual(v.rigidity_eigenvalue, Decimal("8.312"))

    def test_rigid_boundary_passes(self):
        v = RigidityVerifier()
        boundary = [
            CoordinateVector("0", "0", "0", 0),
            CoordinateVector("3", "4", "0", 1),
            CoordinateVector("0", "0", "12", 2),
        ]
        self.assertTrue(v.verify_boundary(boundary))

    def test_eigenvalues_persist_under_rigid_transport(self):
        # Direct check: translating the whole boundary leaves the transport
        # matrix (and thus its eigenvalues) exactly invariant.
        v = RigidityVerifier()
        coords = [
            CoordinateVector("1", "0", "0", 0),
            CoordinateVector("0", "2", "0", 1),
            CoordinateVector("0", "0", "3", 2),
        ]
        base = v._eigenvalues(v._transport_matrix(coords))
        moved = [c.translate(v.delta, v.delta, v.delta) for c in coords]
        after = v._eigenvalues(v._transport_matrix(moved))
        self.assertEqual(v._spectral_drift(base, after), Decimal(0))

    def test_collapsed_boundary_raises_anomaly(self):
        v = RigidityVerifier()
        # All ports at the same coordinate -> zero spatial extent -> anomaly.
        boundary = [
            CoordinateVector("5", "5", "5", 0),
            CoordinateVector("5", "5", "5", 1),
            CoordinateVector("5", "5", "5", 2),
        ]
        with self.assertRaises(AnomalyClosureError):
            v.verify_boundary(boundary)

    def test_non_finite_boundary_raises_anomaly(self):
        v = RigidityVerifier()
        bad = CoordinateVector("0", "0", "0", 0)
        bad.x = Decimal("NaN")
        boundary = [bad, CoordinateVector("1", "1", "1", 1)]
        with self.assertRaises(AnomalyClosureError):
            v.verify_boundary(boundary)

    def test_two_point_eigenvalues_are_plus_minus_distance(self):
        v = RigidityVerifier()
        coords = [
            CoordinateVector("0", "0", "0", 0),
            CoordinateVector("3", "4", "0", 1),
        ]
        eigs = sorted(v._eigenvalues(v._transport_matrix(coords)))
        self.assertEqual(eigs[0], Decimal(-5))
        self.assertEqual(eigs[1], Decimal(5))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
