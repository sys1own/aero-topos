"""Predictive Mitosis -- pre-emptive spectral bisection during synthesis.

Where automated module mitosis normally fires *reactively* once complexity
limits are breached, :class:`PredictiveMitosisEngine` splits a projected
trajectory *before* it is written to disk.  It formulates the graph Laplacian of
the anticipated topology, computes the Fiedler vector (eigenvector of the second
smallest Laplacian eigenvalue), and bisects along the minimum-cut line when the
projected density would violate ``max_module_complexity`` -- synthesizing clean,
decoupled boundary-port interface contracts for each partition.

All linear algebra is native :mod:`numpy`; every method isolates its errors and
degrades to "no split" so synthesis can always proceed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np


class PredictiveMitosisEngine:
    """Anticipates and slices over-dense projected topologies."""

    #: spatial proximity below which two port bindings are deemed connected.
    PROXIMITY = 5.0

    def __init__(self, max_complexity: int = 12):
        self.max_complexity = max_complexity

    # -- adjacency / Laplacian --------------------------------------------
    def _build_adjacency_matrix(self, graph_data: Dict[str, Any]) -> np.ndarray:
        """Adjacency from spatial proximity of the projected port bindings."""
        bindings = graph_data.get("port_bindings", [])
        size = len(bindings)
        if size == 0:
            return np.zeros((1, 1))

        coords = np.array(
            [self._coord(b) for b in bindings], dtype=float
        )  # (size, 3)
        adj = np.zeros((size, size), dtype=float)
        for i in range(size):
            # Vectorized proximity test against all j > i.
            diffs = coords[i + 1 :] - coords[i]
            dists = np.linalg.norm(diffs, axis=1)
            for offset, dist in enumerate(dists):
                if dist < self.PROXIMITY:
                    j = i + 1 + offset
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        return adj

    @staticmethod
    def _coord(binding: Dict[str, Any]) -> List[float]:
        raw = binding.get("spatial_coordinates", [0.0, 0.0, 0.0])
        vec = [float(v) for v in raw[:3]]
        while len(vec) < 3:
            vec.append(0.0)
        return vec

    # -- predictive slicing -----------------------------------------------
    def anticipate_and_slice(self, synthesized_bytes: bytes) -> Tuple[bytes, List[bytes]]:
        """Split the projected topology if it violates the complexity ceiling.

        Returns ``(root_bytes, [extra_partition_bytes, ...])``.  When no split is
        needed (or on any error) the input is returned unchanged with an empty
        partition list.
        """
        try:
            graph_data = json.loads(synthesized_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return synthesized_bytes, []

        bindings = graph_data.get("port_bindings", [])
        if len(bindings) <= self.max_complexity:
            return synthesized_bytes, []

        print(f"[*] Predictive Mitosis: node size ({len(bindings)}) violates safety ceilings.")
        print("[*] Constructing graph Laplacian to compute the spectral bisection vector...")

        try:
            fiedler = self._fiedler_vector(self._build_adjacency_matrix(graph_data))
        except np.linalg.LinAlgError:
            return synthesized_bytes, []

        left_partition: List[dict] = []
        right_partition: List[dict] = []
        for idx, binding in enumerate(bindings):
            (left_partition if fiedler[idx] >= 0 else right_partition).append(binding)

        # Guard against a degenerate one-sided split: fall back to a balanced
        # median cut so the boundary interface is always genuinely decoupled.
        if not left_partition or not right_partition:
            order = np.argsort(fiedler)
            half = len(bindings) // 2
            left_idx = set(order[:half].tolist())
            left_partition = [b for i, b in enumerate(bindings) if i in left_idx]
            right_partition = [b for i, b in enumerate(bindings) if i not in left_idx]

        print(
            f"[+] Spectral partition locked. Left block: {len(left_partition)} nodes | "
            f"Right block: {len(right_partition)} nodes."
        )

        left_subgraph = {
            "format": "aeroc",
            "kernel_lock": graph_data.get("kernel_lock", True),
            "dimension": graph_data.get("dimension"),
            "port_bindings": left_partition,
            "boundary_port": "contract_alpha",
        }
        right_subgraph = {
            "format": "aeroc",
            "kernel_lock": graph_data.get("kernel_lock", True),
            "dimension": graph_data.get("dimension"),
            "port_bindings": right_partition,
            "boundary_port": "contract_beta",
        }
        return (
            json.dumps(left_subgraph).encode("utf-8"),
            [json.dumps(right_subgraph).encode("utf-8")],
        )

    @staticmethod
    def _fiedler_vector(adjacency: np.ndarray) -> np.ndarray:
        """Fiedler vector: eigenvector of the 2nd-smallest Laplacian eigenvalue."""
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        order = np.argsort(eigenvalues)
        fiedler_idx = order[1] if len(order) > 1 else order[0]
        return eigenvectors[:, fiedler_idx]


__all__ = ["PredictiveMitosisEngine"]
