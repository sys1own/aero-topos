# -*- coding: utf-8 -*-
"""Unit tests for ``core.invariants``."""

import unittest

from core.hin_graph import HINGraph, InterfaceSignature
from core.invariants import (
    ArityMismatchError,
    InvariantVerifier,
    InterfaceChangedError,
    PortTypeMismatchError,
    SpectralDegradedError,
)


def _make_graph() -> HINGraph:
    nodes = {"a", "b", "c", "d"}
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    adj = {n: [] for n in nodes}
    rev = {n: [] for n in nodes}
    undirected = {n: [] for n in nodes}
    in_deg = {n: 0 for n in nodes}
    out_deg = {n: 0 for n in nodes}
    edge_types = []
    for s, d in edges:
        adj[s].append(d)
        rev[d].append(s)
        undirected[s].append(d)
        undirected[d].append(s)
        out_deg[s] += 1
        in_deg[d] += 1
        edge_types.append((f"{s}->{d}", "I", "I"))
    return HINGraph(
        nodes=nodes,
        edges=[(s, d, f"{s}->{d}") for s, d in edges],
        adj_list=adj,
        in_degrees=in_deg,
        out_degrees=out_deg,
        expected_arity={"a": 2, "b": 2, "c": 2, "d": 2},
        node_types={},
        reverse_adj=rev,
        undirected_adj=undirected,
        edge_types=edge_types,
    )


class TestInvariantVerifier(unittest.TestCase):
    def test_verify_edge_conservation_passes(self):
        graph = _make_graph()
        InvariantVerifier().verify_edge_conservation(graph)

    def test_verify_edge_conservation_arity_mismatch(self):
        graph = _make_graph()
        graph.expected_arity["a"] = 99
        with self.assertRaises(ArityMismatchError):
            InvariantVerifier().verify_edge_conservation(graph)

    def test_verify_edge_conservation_port_type_mismatch(self):
        graph = _make_graph()
        graph.edge_types = [("a->b", "Tensor", "I")]
        with self.assertRaises(PortTypeMismatchError):
            InvariantVerifier().verify_edge_conservation(graph)

    def test_verify_interface_signatures(self):
        graph = _make_graph()
        verifier = InvariantVerifier()
        waves = [["a"], ["b", "c"], ["d"]]
        verifier.verify_interface_signatures(graph, waves)
        self.assertIn(0, graph.interface_signatures)
        self.assertIn(1, graph.interface_signatures)
        self.assertIn(2, graph.interface_signatures)

    def test_verify_interface_signatures_changed(self):
        graph = _make_graph()
        graph.interface_signatures[1] = InterfaceSignature(inputs=set(), outputs=set())
        with self.assertRaises(InterfaceChangedError):
            InvariantVerifier().verify_interface_signatures(
                graph, [["a"], ["b", "c"], ["d"]]
            )

    def test_verify_spectral_stability_passes(self):
        graph = _make_graph()
        InvariantVerifier().verify_spectral_stability(graph, 0.5, threshold=0.15)

    def test_verify_spectral_stability_fails_when_disconnected(self):
        graph = _make_graph()
        # Remove edges crossing the middle, leaving two components.
        for s, d in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]:
            if d in graph.adj_list[s]:
                graph.adj_list[s].remove(d)
            if s in graph.reverse_adj[d]:
                graph.reverse_adj[d].remove(s)
        graph.undirected_adj = {"a": [], "b": [], "c": [], "d": []}
        with self.assertRaises(SpectralDegradedError):
            InvariantVerifier().verify_spectral_stability(graph, 1.0, threshold=0.15)


if __name__ == "__main__":
    unittest.main()
