# -*- coding: utf-8 -*-
"""Phase 5 integration and edge-case tests for ``BoundaryAwareMutator``.

These tests verify that the DPO-style mutator:

1. Rolls back cleanly when an asymmetric boundary interface cannot be repaired.
2. Inserts adaptor nodes for repairable arity mismatches and produces a
   schedulable, invariant-safe graph.
3. Aborts and rolls back when a mutation would introduce a cyclic dependency.
"""

import unittest
from typing import Any, Dict, List, Set, Tuple

from core.hin_graph import HINGraph
from core.invariants import InvariantVerifier
from core.wavefront_scheduler import WavefrontScheduler
from evolve import BoundaryAwareMutator


def _build_graph(
    edges: List[Tuple[str, str]],
    expected_arity: Dict[str, int],
    types: Dict[Tuple[str, str], Tuple[Any, Any]] = None,
) -> HINGraph:
    """Build a deterministic ``HINGraph`` from a list of directed edges."""
    nodes: Set[str] = set()
    for s, d in edges:
        nodes.add(s)
        nodes.add(d)

    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    rev: Dict[str, List[str]] = {n: [] for n in nodes}
    undirected: Dict[str, List[str]] = {n: [] for n in nodes}
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    out_deg: Dict[str, int] = {n: 0 for n in nodes}

    records: List[dict] = []
    edge_types: List[Tuple[str, Any, Any]] = []
    type_map = types or {}

    for s, d in edges:
        adj[s].append(d)
        rev[d].append(s)
        undirected[s].append(d)
        undirected[d].append(s)
        out_deg[s] += 1
        in_deg[d] += 1

        src_type, dst_type = type_map.get((s, d), ("I", "I"))
        label = f"{s}->{d}"
        records.append(
            {"src": s, "dst": d, "label": label, "src_type": src_type, "dst_type": dst_type}
        )
        edge_types.append((label, src_type, dst_type))

    return HINGraph(
        nodes=nodes,
        edges=[(s, d, f"{s}->{d}") for s, d in edges],
        adj_list=adj,
        in_degrees=in_deg,
        out_degrees=out_deg,
        expected_arity=expected_arity,
        node_types={n: "Node" for n in nodes},
        reverse_adj=rev,
        undirected_adj=undirected,
        edge_types=edge_types,
        edge_records=records,
    )


class TestPhase5Invariants(unittest.TestCase):
    def _make_parent1(self, sink_arity: int) -> HINGraph:
        """Parent with a 2-input/1-output target subgraph rooted at ``t``."""
        return _build_graph(
            edges=[
                ("in1", "t"),
                ("in2", "t"),
                ("t", "u"),
                ("u", "sink"),
                ("sink", "out_ctx"),
            ],
            expected_arity={
                "in1": 1,
                "in2": 1,
                "t": 3,
                "u": 2,
                "sink": sink_arity,
                "out_ctx": 1,
            },
        )

    def _make_parent2(self) -> HINGraph:
        """Donor with a 1-input/3-output node ``m`` and three context sinks."""
        return _build_graph(
            edges=[
                ("src", "m"),
                ("m", "o1"),
                ("m", "o2"),
                ("m", "o3"),
            ],
            expected_arity={
                "src": 1,
                "m": 4,
                "o1": 1,
                "o2": 1,
                "o3": 1,
            },
        )

    def test_asymmetric_interface_rollback(self):
        """A 2-input/1-output target crossed with a 1-input/3-output donor
        that cannot be repaired must roll back and return ``False``."""
        parent1 = self._make_parent1(sink_arity=0)  # sink cannot accept output
        parent2 = self._make_parent2()

        mutator = BoundaryAwareMutator()
        original_nodes = set(parent1.nodes)
        original_edges = list(parent1.edges)

        with self.assertLogs("aero.evolve", level="ERROR") as log_ctx:
            result = mutator.crossover(parent1, parent2, "t", max_size=2)

        self.assertIs(result, False)
        self.assertEqual(parent1.nodes, original_nodes)
        self.assertEqual(parent1.edges, original_edges)
        self.assertIn(
            "error: evolution failed: Type-safe mutation broke edge conservation:",
            "\n".join(log_ctx.output),
        )

    def test_adaptor_insertion_and_execution(self):
        """A repairable input-arity mismatch must produce a valid graph whose
        wavefront schedule can be computed and whose invariants hold."""
        parent1 = self._make_parent1(sink_arity=2)
        parent2 = self._make_parent2()

        mutator = BoundaryAwareMutator()
        child = mutator.crossover(parent1, parent2, "t", max_size=2)

        self.assertIsNot(child, False)
        self.assertIsInstance(child, HINGraph)
        self.assertTrue(
            any(node.startswith("adapt_") for node in child.nodes),
            "Expected an identity/coercion adaptor to be inserted for the arity mismatch",
        )

        # Verify structural invariants and a valid topological wavefront schedule.
        InvariantVerifier().verify_all(child)
        waves = WavefrontScheduler().compute_wavefronts(child)
        self.assertGreater(len(waves), 0)

        # No wave should depend on itself (a basic acyclicity/smoke check).
        node_to_wave = {}
        for level, wave in enumerate(waves):
            for node in wave:
                node_to_wave[node] = level
        for s, d, _ in child.edges:
            if s in node_to_wave and d in node_to_wave:
                self.assertLess(
                    node_to_wave[s],
                    node_to_wave[d],
                    f"Edge {s!r} -> {d!r} violates wave ordering",
                )

    def test_topological_cycle_prevention(self):
        """A donor that introduces a feedback loop must trigger rollback."""
        parent1 = _build_graph(
            edges=[
                ("in1", "t"),
                ("in2", "t"),
                ("t", "u"),
                ("u", "sink"),
            ],
            expected_arity={
                "in1": 1,
                "in2": 1,
                "t": 3,
                "u": 2,
                "sink": 1,
            },
        )

        # Cyclic donor: x <-> y.  No topological schedule is possible.
        parent2 = _build_graph(
            edges=[("x", "y"), ("y", "x")],
            expected_arity={"x": 2, "y": 2},
        )

        mutator = BoundaryAwareMutator()
        original_nodes = set(parent1.nodes)

        with self.assertLogs("aero.evolve", level="ERROR") as log_ctx:
            result = mutator.crossover(parent1, parent2, "t", max_size=2)

        self.assertIs(result, False)
        self.assertEqual(parent1.nodes, original_nodes)
        self.assertIn(
            "error: evolution failed: Type-safe mutation broke edge conservation:",
            "\n".join(log_ctx.output),
        )


if __name__ == "__main__":
    unittest.main()
