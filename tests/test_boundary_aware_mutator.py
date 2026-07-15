# -*- coding: utf-8 -*-
"""Tests for ``BoundaryAwareMutator`` in ``evolve.py``."""

import unittest

from core.hin_graph import HINGraph
from core.invariants import InvariantVerifier
from evolve import BoundaryAwareMutator


def _diamond_graph() -> HINGraph:
    """a -> b -> d and a -> c -> d."""
    nodes = {"a", "b", "c", "d"}
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    adj = {n: [] for n in nodes}
    rev = {n: [] for n in nodes}
    undirected = {n: [] for n in nodes}
    in_deg = {n: 0 for n in nodes}
    out_deg = {n: 0 for n in nodes}
    records = []
    types = []
    for s, d in edges:
        adj[s].append(d)
        rev[d].append(s)
        undirected[s].append(d)
        undirected[d].append(s)
        out_deg[s] += 1
        in_deg[d] += 1
        records.append(
            {"src": s, "dst": d, "label": f"{s}->{d}", "src_type": "I", "dst_type": "I"}
        )
        types.append((f"{s}->{d}", "I", "I"))
    return HINGraph(
        nodes=nodes,
        edges=[(s, d, f"{s}->{d}") for s, d in edges],
        adj_list=adj,
        in_degrees=in_deg,
        out_degrees=out_deg,
        expected_arity={"a": 2, "b": 2, "c": 2, "d": 2},
        node_types={n: "Node" for n in nodes},
        reverse_adj=rev,
        undirected_adj=undirected,
        edge_types=types,
        edge_records=records,
    )


class TestBoundaryAwareMutator(unittest.TestCase):
    def test_extract_interface(self):
        graph = _diamond_graph()
        mutator = BoundaryAwareMutator()
        iface = mutator.extract_interface(graph, {"b", "c"})
        self.assertEqual(set(iface["inputs"].keys()), {"a->b", "a->c"})
        self.assertEqual(set(iface["outputs"].keys()), {"b->d", "c->d"})

    def test_crossover_with_matching_interface(self):
        # Parent 1: a -> b -> c -> d
        g1_edges = [("a", "b"), ("b", "c"), ("c", "d")]
        # Parent 2: a -> x -> d (external input "a" feeds x)
        g2_edges = [("a", "x"), ("x", "d")]
        g1 = _graph_from_edges(g1_edges, {"a": 1, "b": 2, "c": 2, "d": 1})
        g2 = _graph_from_edges(g2_edges, {"a": 1, "x": 2, "d": 1})

        mutator = BoundaryAwareMutator()
        child = mutator.crossover(g1, g2, "b")
        self.assertIn("a", child.nodes)
        self.assertTrue(any("donor" in n for n in child.nodes))
        InvariantVerifier().verify_all(child)

    def test_crossover_with_type_mismatch_inserts_adaptor(self):
        g1 = _graph_from_edges([("a", "b"), ("b", "c")], {"a": 1, "b": 2, "c": 1})
        g2 = _graph_from_edges([("a", "x"), ("x", "c")], {"a": 1, "x": 2, "c": 1})
        # Make the donor input require a different type than the parent source.
        g2.edge_records[0]["dst_type"] = "Tensor"
        g2.edge_types[0] = (g2.edge_types[0][0], "I", "Tensor")

        mutator = BoundaryAwareMutator()
        child = mutator.crossover(g1, g2, "b")
        self.assertTrue(any(n.startswith("adapt_") for n in child.nodes))
        InvariantVerifier().verify_all(child)

    def test_crossover_rolls_back_on_unrecoverable_failure(self):
        g1 = _graph_from_edges([("a", "b"), ("b", "c")], {"a": 1, "b": 2, "c": 1})
        # Parent 2 is a disconnected cycle that cannot be spliced in.
        g2 = _graph_from_edges([("x", "y"), ("y", "x")], {"x": 1, "y": 1})

        mutator = BoundaryAwareMutator()
        original_nodes = set(g1.nodes)
        child = mutator.crossover(g1, g2, "b")
        # Should abort and report failure without mutating parent1.
        self.assertIs(child, False)
        self.assertEqual(g1.nodes, original_nodes)


def _graph_from_edges(edge_list, arity):
    nodes = set(e[0] for e in edge_list) | set(e[1] for e in edge_list)
    adj = {n: [] for n in nodes}
    rev = {n: [] for n in nodes}
    undirected = {n: [] for n in nodes}
    in_deg = {n: 0 for n in nodes}
    out_deg = {n: 0 for n in nodes}
    records = []
    types = []
    for s, d in edge_list:
        adj[s].append(d)
        rev[d].append(s)
        undirected[s].append(d)
        undirected[d].append(s)
        out_deg[s] += 1
        in_deg[d] += 1
        records.append(
            {"src": s, "dst": d, "label": f"{s}->{d}", "src_type": "I", "dst_type": "I"}
        )
        types.append((f"{s}->{d}", "I", "I"))
    return HINGraph(
        nodes=nodes,
        edges=[(s, d, f"{s}->{d}") for s, d in edge_list],
        adj_list=adj,
        in_degrees=in_deg,
        out_degrees=out_deg,
        expected_arity=arity,
        node_types={n: "Node" for n in nodes},
        reverse_adj=rev,
        undirected_adj=undirected,
        edge_types=types,
        edge_records=records,
    )


if __name__ == "__main__":
    unittest.main()
