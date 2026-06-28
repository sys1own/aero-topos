"""Tests for Causal Horizon Synthesis, Predictive Mitosis, and the Quantum
Phase Space Registry, plus the evolve.py synthesis gateway."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

import evolve
from core.causal_gradient import CausalHorizonSynthesizer
from core.mitosis_predictor import PredictiveMitosisEngine
from core.quantum_registry import AnomalyClosureError, QuantumPhaseRegistry


def _uast(n: int) -> dict:
    return {"nodes": [{"id": f"n{i}", "type": "gamma"} for i in range(n)]}


# ---------------------------------------------------------------------------
# Causal Horizon Synthesizer
# ---------------------------------------------------------------------------
class TestCausalSynthesizer(unittest.TestCase):
    def test_empty_uast_returns_none(self):
        synth = CausalHorizonSynthesizer(ledger_path="")
        self.assertIsNone(synth.synthesize_pre_compacted_topology({"nodes": []}))

    def test_gradient_is_identity_without_history(self):
        synth = CausalHorizonSynthesizer(ledger_path="/no/such/ledger.aero")
        grad = synth.compute_causal_gradient(5, 6.0)
        self.assertEqual(grad.shape, (synth.kernel_dimension, synth.kernel_dimension))
        np.testing.assert_array_equal(grad, np.identity(synth.kernel_dimension))

    def test_synthesis_emits_kernel_locked_topology(self):
        synth = CausalHorizonSynthesizer(ledger_path="")
        out = synth.synthesize_pre_compacted_topology(_uast(4))
        self.assertIsInstance(out, bytes)
        data = json.loads(out.decode("utf-8"))
        self.assertEqual(data["format"], "aeroc")
        self.assertTrue(data["kernel_lock"])
        self.assertEqual(data["dimension"], 312)
        self.assertEqual(len(data["port_bindings"]), 4)
        for binding in data["port_bindings"]:
            self.assertEqual(len(binding["spatial_coordinates"]), 3)
            self.assertTrue(all(np.isfinite(binding["spatial_coordinates"])))

    def test_gradient_uses_history(self):
        # Build a ledger whose entries report high speed gains + coordinates.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "context.aero")
            chain = [
                {
                    "metrics": {"speed_gain": 0.9},
                    "coordinate_vector": [float(i + k) for k in range(312)],
                }
                for i in range(5)
            ]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"ledger": chain, "version": "aero_calculus_v1"}, fh)
            synth = CausalHorizonSynthesizer(ledger_path=path)
            grad = synth.compute_causal_gradient(5, 6.0)
            self.assertEqual(grad.shape, (312, 312))
            # A real projection differs from the identity fallback.
            self.assertFalse(np.array_equal(grad, np.identity(312)))
            # Projection operators are symmetric.
            np.testing.assert_allclose(grad, grad.T, atol=1e-9)

    def test_loads_mutation_history_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "context.aero")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"mutation_history": [{"metrics": {"speed_gain": 0.1}}]}, fh)
            synth = CausalHorizonSynthesizer(ledger_path=path)
            self.assertEqual(len(synth._load_ledger_history()), 1)


# ---------------------------------------------------------------------------
# Predictive Mitosis Engine
# ---------------------------------------------------------------------------
class TestPredictiveMitosis(unittest.TestCase):
    def test_small_graph_not_split(self):
        engine = PredictiveMitosisEngine(max_complexity=12)
        payload = json.dumps(
            {"format": "aeroc", "port_bindings": [
                {"node_id": "a", "spatial_coordinates": [0, 0, 0]}]}
        ).encode("utf-8")
        root, splits = engine.anticipate_and_slice(payload)
        self.assertEqual(root, payload)
        self.assertEqual(splits, [])

    def test_dense_graph_is_bisected(self):
        engine = PredictiveMitosisEngine(max_complexity=4)
        # Two spatial clusters so the Fiedler cut is well-defined.
        bindings = []
        for i in range(5):
            bindings.append({"node_id": f"l{i}", "spatial_coordinates": [float(i), 0.0, 0.0]})
        for i in range(5):
            bindings.append({"node_id": f"r{i}", "spatial_coordinates": [100.0 + i, 0.0, 0.0]})
        payload = json.dumps({"format": "aeroc", "dimension": 312,
                              "port_bindings": bindings}).encode("utf-8")
        root, splits = engine.anticipate_and_slice(payload)
        self.assertEqual(len(splits), 1)
        left = json.loads(root.decode("utf-8"))
        right = json.loads(splits[0].decode("utf-8"))
        self.assertEqual(left["boundary_port"], "contract_alpha")
        self.assertEqual(right["boundary_port"], "contract_beta")
        # Every node is preserved across the cut (no loss, no duplication).
        total = len(left["port_bindings"]) + len(right["port_bindings"])
        self.assertEqual(total, len(bindings))
        self.assertGreater(len(left["port_bindings"]), 0)
        self.assertGreater(len(right["port_bindings"]), 0)

    def test_malformed_bytes_passthrough(self):
        engine = PredictiveMitosisEngine()
        root, splits = engine.anticipate_and_slice(b"not json")
        self.assertEqual(root, b"not json")
        self.assertEqual(splits, [])


# ---------------------------------------------------------------------------
# Quantum Phase Registry
# ---------------------------------------------------------------------------
class TestQuantumRegistry(unittest.TestCase):
    def test_state_hash_is_deterministic_and_temporal(self):
        reg = QuantumPhaseRegistry()
        h1 = reg.compute_state_hash(b"graph", 5)
        h2 = reg.compute_state_hash(b"graph", 5)
        h3 = reg.compute_state_hash(b"graph", 6)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)  # T_causal participates in the hash
        self.assertEqual(len(h1), 64)

    def test_rigidity_floor_passes_for_finite_coords(self):
        reg = QuantumPhaseRegistry()
        self.assertTrue(reg.verify_rigidity_floor([1.0, 2.0, 3.0]))

    def test_rigidity_floor_empty_is_safe(self):
        self.assertTrue(QuantumPhaseRegistry().verify_rigidity_floor([]))

    def test_non_finite_raises_anomaly(self):
        reg = QuantumPhaseRegistry()
        with self.assertRaises(AnomalyClosureError):
            reg.verify_rigidity_floor([1.0, float("nan"), 3.0])

    def test_encrypt_and_verify_roundtrip(self):
        reg = QuantumPhaseRegistry()
        payload = json.dumps({"port_bindings": [
            {"node_id": "a", "spatial_coordinates": [1.0, 2.0, 3.0]}]}).encode("utf-8")
        state_hash = reg.encrypt_and_verify(payload, 7)
        self.assertEqual(state_hash, reg.compute_state_hash(payload, 7))


# ---------------------------------------------------------------------------
# evolve.py synthesis gateway
# ---------------------------------------------------------------------------
class TestSynthesisGateway(unittest.TestCase):
    def test_intercept_returns_payloads(self):
        payloads = evolve.intercept_and_synthesize_workload(_uast(6), current_t_causal=3)
        self.assertGreaterEqual(len(payloads), 1)
        for payload in payloads:
            self.assertIsInstance(payload, bytes)
            json.loads(payload.decode("utf-8"))  # valid serialized topology

    def test_intercept_empty_workload_falls_back(self):
        self.assertEqual(
            evolve.intercept_and_synthesize_workload({"nodes": []}, current_t_causal=0), []
        )

    def test_intercept_dense_workload_triggers_mitosis(self):
        # 30 nodes well above the default max_complexity (12) -> a split occurs.
        payloads = evolve.intercept_and_synthesize_workload(_uast(30), current_t_causal=1)
        self.assertGreaterEqual(len(payloads), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
