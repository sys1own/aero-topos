"""Tests for the Aero-Calculus compiler frontend (``core/translator.py``)."""

from __future__ import annotations

import unittest

from core.hin_vm import (
    ConstructorNode,
    DuplicatorNode,
    HINNetwork,
    SwitchNode,
    ValueNode,
)
from core.translator import BoundaryPortNode, UASTToHINTranslator


def _count(net: HINNetwork, cls) -> int:
    return sum(1 for n in net.nodes.values() if isinstance(n, cls))


# ---------------------------------------------------------------------------
# Homomorphic translation
# ---------------------------------------------------------------------------
class TestTranslation(unittest.TestCase):
    def test_literal_becomes_value_node(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {"type": "literal", "value": 42},
        ]}
        net = tr.translate(uast)
        values = [n for n in net.nodes.values() if isinstance(n, ValueNode)]
        self.assertTrue(any(v.value == 42 for v in values))
        net.validate_conservation()

    def test_if_becomes_switch(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {
                "type": "if",
                "condition": {"type": "literal", "value": True},
                "then": {"type": "literal", "value": 1},
                "else": {"type": "literal", "value": 0},
            }
        ]}
        net = tr.translate(uast)
        self.assertEqual(_count(net, SwitchNode), 1)
        net.validate_conservation()

    def test_function_becomes_constructor(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {
                "type": "function_declaration",
                "name": "f",
                "param": "x",
                "body": [{"type": "reference", "name": "x"}],
            }
        ]}
        net = tr.translate(uast)
        self.assertEqual(_count(net, ConstructorNode), 1)
        net.validate_conservation()

    def test_single_reference_uses_no_duplicator(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {"type": "binding", "name": "y",
             "value": {"type": "literal", "value": 5}},
            {"type": "reference", "name": "y"},
        ]}
        net = tr.translate(uast)
        self.assertEqual(_count(net, DuplicatorNode), 0)
        net.validate_conservation()

    def test_multi_reference_injects_duplicator(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {"type": "binding", "name": "y",
             "value": {"type": "literal", "value": 5}},
            {
                "type": "if",
                "condition": {"type": "reference", "name": "y"},
                "then": {"type": "reference", "name": "y"},
                "else": {"type": "literal", "value": 0},
            },
        ]}
        net = tr.translate(uast)
        # y is referenced twice -> exactly one duplicator fork.
        self.assertEqual(_count(net, DuplicatorNode), 1)
        net.validate_conservation()

    def test_no_unterminated_ports(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {"type": "binding", "name": "a",
             "value": {"type": "literal", "value": 1}},
            {"type": "binding", "name": "b",
             "value": {"type": "literal", "value": 2}},
            {"type": "call",
             "function": {"type": "reference", "name": "a"},
             "argument": {"type": "reference", "name": "b"}},
        ]}
        net = tr.translate(uast)
        # The whole net is closed: every auxiliary port is bound.
        net.validate_conservation()


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------
class TestComplexity(unittest.TestCase):
    def test_metrics_keys(self):
        tr = UASTToHINTranslator()
        uast = {"type": "module", "children": [
            {"type": "literal", "value": 1},
        ]}
        net = tr.translate(uast)
        metrics = tr.evaluate_complexity(net)
        for key in ("node_count", "edge_count", "density", "avg_degree"):
            self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["node_count"], 1.0)


# ---------------------------------------------------------------------------
# Spectral mitosis
# ---------------------------------------------------------------------------
def _barbell() -> HINNetwork:
    """Two triangles joined by a single bridge edge (min cut = 1)."""
    net = HINNetwork()
    a = [ConstructorNode(f"a{i}") for i in range(3)]
    b = [ConstructorNode(f"b{i}") for i in range(3)]
    for node in a + b:
        net.register_node(node)

    def triangle(g):
        net._link(g[0].a_1, g[1].a_1)
        net._link(g[1].a_2, g[2].a_1)
        net._link(g[2].a_2, g[0].a_2)

    triangle(a)
    triangle(b)
    # single bridge between the clusters
    net._link(a[0].p, b[0].p)
    return net


class TestMitosis(unittest.TestCase):
    def test_fiedler_flags_single_crossing_edge(self):
        tr = UASTToHINTranslator()
        net = _barbell()
        part_1, part_2 = tr.split_module(net)

        # Both partitions are non-empty.
        self.assertGreater(len(part_1.nodes), 0)
        self.assertGreater(len(part_2.nodes), 0)

        # The minimum cut flags exactly the one bridge wire.
        contracts = part_1.boundary_contracts
        self.assertEqual(len(contracts), 1)

        # The two triangles ended up on opposite sides of the cut.
        self.assertEqual(contracts[0]["side_a"] != contracts[0]["side_b"], True)

    def test_split_preserves_termination(self):
        tr = UASTToHINTranslator()
        net = _barbell()
        part_1, part_2 = tr.split_module(net)
        # No auxiliary port is left dangling on either side: crossing edges are
        # reified as boundary caps.
        part_1.validate_conservation()
        part_2.validate_conservation()

    def test_boundary_caps_created(self):
        tr = UASTToHINTranslator()
        net = _barbell()
        part_1, part_2 = tr.split_module(net)
        caps = _count(part_1, BoundaryPortNode) + _count(part_2, BoundaryPortNode)
        # One severed edge -> one cap per side -> two caps total.
        self.assertEqual(caps, 2)

    def test_execute_mitosis_below_threshold_is_noop(self):
        tr = UASTToHINTranslator(auto_split_threshold=120)
        net = _barbell()  # 6 nodes, well under threshold
        primary, secondary = tr.execute_mitosis(net)
        self.assertIs(primary, net)
        self.assertEqual(len(secondary.nodes), 0)

    def test_execute_mitosis_above_threshold_splits(self):
        tr = UASTToHINTranslator(auto_split_threshold=4)
        net = _barbell()  # 6 nodes > threshold 4
        primary, secondary = tr.execute_mitosis(net)
        self.assertGreater(len(primary.nodes), 0)
        self.assertGreater(len(secondary.nodes), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
