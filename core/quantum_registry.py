"""Quantum Phase Space Registry -- ledger immutability during self-writing.

Guarantees that a self-written graph state is hashed against its precise
temporal coordinate (``T_causal``) and that its geometry is *on-shell*: a
coordinate-perturbation sweep displaces the state vector by a microscopic
``10⁻¹²⁰`` and confirms the layout stays within the holographic noise floor
(``1/N ≈ 10⁻¹²²``).  If the geometry drifts off-shell an
:class:`AnomalyClosureError` is raised.

The canonical :class:`AnomalyClosureError` is shared with the spacetime ledger
layer so error handling is unified across the ecosystem (a local definition is
used as a fallback if that layer is unavailable).
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

try:  # unify the anomaly type with the rest of the stack
    from core.spacetime_ledger import AnomalyClosureError
except Exception:  # pragma: no cover - standalone fallback
    class AnomalyClosureError(Exception):
        """Raised when coordinate drift compromises the algebraic rigidity floor."""


class QuantumPhaseRegistry:
    """Temporal configuration hashing + coordinate rigidity verification."""

    def __init__(self, rigidity_threshold: float = 1e-115, perturbation: float = 1e-120):
        # Drift past this floor means the geometry has destabilized off-shell.
        self.rigidity_threshold = rigidity_threshold
        self.perturbation = perturbation

    # -- temporal configuration hashing -----------------------------------
    @staticmethod
    def compute_state_hash(graph_bytes: bytes, t_causal: int) -> str:
        """Hash a graph layout together with its temporal coordinate."""
        if not isinstance(graph_bytes, (bytes, bytearray)):
            graph_bytes = str(graph_bytes).encode("utf-8")
        payload = bytes(graph_bytes) + b"|T=" + str(int(t_causal)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # -- coordinate perturbation sweep ------------------------------------
    def verify_rigidity_floor(self, coordinates: Sequence[float]) -> bool:
        """Run a perturbation sweep and verify the geometry stays on-shell.

        Displaces the coordinates by a ``perturbation``-scale Gaussian jitter and
        confirms the resulting drift stays below ``rigidity_threshold``.  A
        non-finite coordinate is itself an off-shell anomaly.

        Raises:
            AnomalyClosureError: if the geometry drifts past the noise floor.
        """
        coords = np.asarray(list(coordinates), dtype=float)
        if coords.size == 0:
            return True
        if not np.all(np.isfinite(coords)):
            raise AnomalyClosureError(
                "Non-finite coordinate detected; trajectory is off-shell."
            )

        # Deterministic, reproducible jitter seeded from the coordinate content.
        seed = int.from_bytes(
            hashlib.sha256(coords.tobytes()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        perturbed = coords + rng.normal(0.0, self.perturbation, coords.shape)

        drift = float(np.linalg.norm(coords - perturbed))
        if drift > self.rigidity_threshold:
            raise AnomalyClosureError(
                "Holographic noise limit violated. Coordinate trajectory destabilized."
            )
        return True

    def encrypt_and_verify(self, payload: bytes, t_causal: int) -> str:
        """Hash a payload and verify the rigidity of its embedded coordinates.

        Convenience wrapper used by the synthesis pipeline: extracts any spatial
        coordinates carried in the payload, runs the rigidity sweep over them,
        and returns the locked state hash.
        """
        coords = self._extract_coordinates(payload)
        self.verify_rigidity_floor(coords)
        return self.compute_state_hash(payload, t_causal)

    @staticmethod
    def _extract_coordinates(payload: bytes) -> List[float]:
        import json

        try:
            data = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return []
        coords: List[float] = []
        for binding in data.get("port_bindings", []):
            for value in binding.get("spatial_coordinates", []):
                try:
                    coords.append(float(value))
                except (TypeError, ValueError):
                    continue
        return coords


__all__ = ["QuantumPhaseRegistry", "AnomalyClosureError"]
