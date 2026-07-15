# -*- coding: utf-8 -*-
"""Unit tests for ``core.wavefront_scheduler``."""

import unittest

from core.hin_graph import GraphMutation, HINGraph
from core.wavefront_scheduler import CycleError, WavefrontScheduler


def _make_diamond() -> HINGraph:
    nodes = {"a", "b", "c", "d"}
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    adj = {n: [] for n in nodes}
    rev = {n: [] for n in nodes}
    undirected = {n: [] for n in nodes}
    in_deg = {n: 0 for n in nodes}
    out_deg = {n: 0 for n in nodes}
    for s, d in edges:
        adj[s].append(d)
        rev[d].append(s)
        undirected[s].append(d)
        undirected[d].append(s)
        out_deg[s] += 1
        in_deg[d] += 1
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
        edge_types=[],
    )


class TestWavefrontScheduler(unittest.TestCase):
    def test_compute_wavefronts_diamond(self):
        graph = _make_diamond()
        sched = WavefrontScheduler()
        waves = sched.compute_wavefronts(graph)
        self.assertEqual(waves, [["a"], ["b", "c"], ["d"]])

    def test_update_schedule_adds_new_leaf(self):
        graph = _make_diamond()
        sched = WavefrontScheduler()
        waves = sched.compute_wavefronts(graph)

        # Add a new edge d -> e.  d's out-degree grows and e becomes a new wave.
        graph.nodes.add("e")
        graph.adj_list["d"].append("e")
        graph.reverse_adj["e"] = ["d"]
        graph.undirected_adj["d"].append("e")
        graph.undirected_adj["e"] = ["d"]
        graph.in_degrees["e"] = 1
        graph.out_degrees["d"] += 1
        graph.expected_arity["d"] += 1
        graph.expected_arity["e"] = 1
        graph.edges.append(("d", "e", "d->e"))

        new = sched.update_schedule(graph, waves, [GraphMutation.add_edge("d", "e")])
        self.assertEqual(new, [["a"], ["b", "c"], ["d"], ["e"]])

    def test_cycle_raises(self):
        graph = _make_diamond()
        graph.adj_list["d"].append("a")
        graph.reverse_adj["a"].append("d")
        graph.in_degrees["a"] += 1
        graph.out_degrees["d"] += 1
        graph.edges.append(("d", "a", "d->a"))
        sched = WavefrontScheduler()
        with self.assertRaises(CycleError):
            sched.compute_wavefronts(graph)


if __name__ == "__main__":
    unittest.main()
