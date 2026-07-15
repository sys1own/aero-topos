"""Structural invariant verification for the Phase 5 execution engine.

``InvariantVerifier`` enforces three boundary-aware properties on a
:class:`~core.hin_graph.HINGraph`:

* **Edge conservation** -- every node's total in/out degree equals its expected
  arity and the types on either end of every edge agree.
* **Interface signatures** -- the boundary ports crossing each execution wave
  match the stored / expected signature.
* **Spectral stability** -- the graph Laplacian's Fiedler value does not drop
  catastrophically, which would signal a disconnected or nearly-disconnected
  topology.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.hin_graph import HINGraph, InterfaceSignature


class InvariantError(Exception):
    """Base class for structural invariant violations."""


class ArityMismatchError(InvariantError):
    """Raised when a node's degree does not match its expected arity."""

    def __init__(self, node: str, expected: int, actual: int) -> None:
        self.node = node
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Arity mismatch at node {node!r}: expected {expected}, actual {actual}"
        )


class PortTypeMismatchError(InvariantError):
    """Raised when the types on either side of an edge are not compatible."""

    def __init__(self, edge: str, expected: str, actual: str) -> None:
        self.edge = edge
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Port type mismatch on edge {edge!r}: expected {expected}, actual {actual}"
        )


class InterfaceChangedError(InvariantError):
    """Raised when a wave's boundary interface signature changes."""

    def __init__(
        self, wave: int, expected: InterfaceSignature, actual: InterfaceSignature
    ) -> None:
        self.wave = wave
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Interface changed for wave {wave}: expected {expected!r}, actual {actual!r}"
        )


class SpectralDegradedError(InvariantError):
    """Raised when the graph's algebraic connectivity drops significantly."""

    def __init__(self, original: float, actual: float, threshold: float) -> None:
        self.original = original
        self.actual = actual
        self.threshold = threshold
        super().__init__(
            f"Spectral stability degraded: original={original}, actual={actual}, "
            f"threshold={threshold}"
        )


class InvariantVerifier:
    """Verify edge conservation, interface signatures and spectral stability."""

    def verify_edge_conservation(self, graph: HINGraph) -> None:
        """Check that node degrees match expected arities and edge types agree.

        Raises:
            ArityMismatchError: if a node's actual degree differs from its arity.
            PortTypeMismatchError: if connected ports carry incompatible types.
        """
        for node in graph.nodes:
            expected = graph.expected_arity.get(node, 0)
            actual = graph.in_degrees.get(node, 0) + graph.out_degrees.get(node, 0)
            if expected != actual:
                raise ArityMismatchError(node, expected, actual)

        for edge_label, expected_type, actual_type in graph.edge_types:
            if not self._types_compatible(expected_type, actual_type):
                raise PortTypeMismatchError(
                    edge_label,
                    self._type_name(expected_type),
                    self._type_name(actual_type),
                )

    def verify_interface_signatures(
        self, graph: HINGraph, waves: List[List[str]]
    ) -> None:
        """Compute each wave's boundary signature and compare to expectations.

        Raises:
            InterfaceChangedError: when an actual signature does not match the
            stored expected signature.
        """
        node_to_wave: Dict[str, int] = {}
        for level, wave in enumerate(waves):
            for node in wave:
                node_to_wave[node] = level

        actual: Dict[int, InterfaceSignature] = {}
        for src, dst, label in graph.edges:
            w_src = node_to_wave.get(src)
            w_dst = node_to_wave.get(dst)
            if w_src is None or w_dst is None or w_src == w_dst:
                continue
            if w_src < w_dst:
                actual.setdefault(w_src, InterfaceSignature()).outputs.add(label)
                actual.setdefault(w_dst, InterfaceSignature()).inputs.add(label)
            else:
                actual.setdefault(w_src, InterfaceSignature()).inputs.add(label)
                actual.setdefault(w_dst, InterfaceSignature()).outputs.add(label)

        expected = graph.interface_signatures
        all_waves = sorted(set(expected.keys()) | set(actual.keys()))
        for wave_id in all_waves:
            exp = expected.get(wave_id)
            act = actual.get(wave_id, InterfaceSignature())
            if exp is None:
                expected[wave_id] = act
                continue
            if not exp.matches(act):
                raise InterfaceChangedError(wave_id, exp, act)

    def verify_spectral_stability(
        self,
        graph: HINGraph,
        original_lambda_2: float,
        threshold: float = 0.15,
    ) -> None:
        """Check that the graph's algebraic connectivity has not collapsed.

        Computes the second-smallest eigenvalue :math:`\\lambda_2` of the
        undirected graph Laplacian and raises if it drops below the allowed
        threshold relative to ``original_lambda_2``.

        Raises:
            SpectralDegradedError: if algebraic connectivity degraded.
        """
        n = len(graph.nodes)
        if n <= 1:
            current = 0.0
        else:
            nodes = sorted(graph.nodes)
            index = {node: i for i, node in enumerate(nodes)}
            adj = np.zeros((n, n), dtype=float)
            for u, neighbors in graph.undirected_adj.items():
                i = index[u]
                for v in neighbors:
                    j = index[v]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
            degree = np.diag(np.sum(adj, axis=1))
            laplacian = degree - adj
            eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
            current = float(eigenvalues[1])

        if original_lambda_2 > 0.0:
            if current < (1.0 - threshold) * original_lambda_2:
                raise SpectralDegradedError(original_lambda_2, current, threshold)
        else:
            if current < threshold:
                raise SpectralDegradedError(original_lambda_2, current, threshold)

    @staticmethod
    def _type_name(value: Any) -> str:
        return str(value) if value is not None else ""

    @staticmethod
    def _types_compatible(expected: Any, actual: Any) -> bool:
        """Return ``True`` if two port types can be connected.

        Delegates to a ``unifiable`` method when available (e.g.,
        :class:`~core.hin_vm.MELLType`), otherwise falls back to string equality.
        """
        if expected is None or actual is None:
            return True
        if hasattr(expected, "unifiable") and callable(expected.unifiable):
            return bool(expected.unifiable(actual))
        if hasattr(actual, "unifiable") and callable(actual.unifiable):
            return bool(actual.unifiable(expected))
        return str(expected) == str(actual)
