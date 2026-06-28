"""Tests for the Aero-Calculus HIN-VM execution model (``core/hin_vm.py``)."""

from __future__ import annotations

import unittest

from core.hin_vm import (
    CausalProjectionNode,
    ConstructorNode,
    DestructorNode,
    DuplicatorNode,
    EraserNode,
    HINNetwork,
    MELLType,
    Node,
    Port,
    SwitchNode,
    TypeKind,
    ValueNode,
)


def _leaf(net: HINNetwork, node_id: str, mtype: MELLType) -> Port:
    """A single-aux marker node used to observe where a wire ends up."""

    class _Leaf(Node):
        symbol = "leaf"

        def __init__(self, nid: str, t: MELLType):
            super().__init__(nid)
            self._set_principal(t)
            self.a_1 = self._add_aux("a_1", t)

    leaf = _Leaf(node_id, mtype)
    net.register_node(leaf)
    return leaf.a_1


# ---------------------------------------------------------------------------
# MELL type system
# ---------------------------------------------------------------------------
class TestMELLTypes(unittest.TestCase):
    def test_repr(self):
        self.assertEqual(repr(MELLType.unit()), "I")
        self.assertEqual(
            repr(MELLType.tensor(MELLType.unit(), MELLType.unit())),
            "(I ⊗ I)",
        )
        self.assertEqual(
            repr(MELLType.implication(MELLType.unit(), MELLType.unit())),
            "(I ⊸ I)",
        )
        self.assertEqual(repr(MELLType.bang(MELLType.unit())), "!I")

    def test_structural_equality_and_unification(self):
        a = MELLType.tensor(MELLType.unit(), MELLType.unit())
        b = MELLType.tensor(MELLType.unit(), MELLType.unit())
        self.assertEqual(a, b)
        self.assertTrue(a.unifiable(b))

    def test_non_unifiable_kinds(self):
        self.assertFalse(MELLType.unit().unifiable(MELLType.tensor(
            MELLType.unit(), MELLType.unit())))
        self.assertFalse(
            MELLType.tensor(MELLType.unit(), MELLType.unit()).unifiable(
                MELLType.implication(MELLType.unit(), MELLType.unit())
            )
        )

    def test_bang_dereliction(self):
        # !A unifies with the bare A it promotes.
        self.assertTrue(MELLType.bang(MELLType.unit()).unifiable(MELLType.unit()))
        self.assertTrue(MELLType.unit().unifiable(MELLType.bang(MELLType.unit())))


# ---------------------------------------------------------------------------
# Typing guardrail on binding
# ---------------------------------------------------------------------------
class TestTypingGuardrail(unittest.TestCase):
    def test_compatible_bind_succeeds(self):
        net = HINNetwork()
        v1 = ValueNode("v1", 1, MELLType.unit())
        v2 = ValueNode("v2", 2, MELLType.unit())
        net.register_node(v1)
        net.register_node(v2)
        net.bind(v1.p, v2.p)
        self.assertIs(v1.p.target, v2.p)
        self.assertIs(v2.p.target, v1.p)

    def test_incompatible_bind_raises_type_error(self):
        net = HINNetwork()
        v1 = ValueNode("v1", 1, MELLType.unit())
        v2 = ValueNode(
            "v2", 2, MELLType.tensor(MELLType.unit(), MELLType.unit())
        )
        net.register_node(v1)
        net.register_node(v2)
        with self.assertRaises(TypeError):
            net.bind(v1.p, v2.p)


# ---------------------------------------------------------------------------
# Active pairs
# ---------------------------------------------------------------------------
class TestActivePairs(unittest.TestCase):
    def test_principal_binding_registers_active_pair(self):
        net = HINNetwork()
        c = ConstructorNode("c")
        d = DestructorNode("d")
        net.register_node(c)
        net.register_node(d)
        net.bind(c.p, d.p)
        self.assertEqual(len(net.active_pairs), 1)
        self.assertIs(c.active_pair, d)
        self.assertIs(d.active_pair, c)

    def test_auxiliary_binding_is_not_active(self):
        net = HINNetwork()
        c = ConstructorNode("c")
        d = DestructorNode("d")
        net.register_node(c)
        net.register_node(d)
        net.bind(c.a_1, d.a_1)
        self.assertEqual(net.active_pairs, [])
        self.assertIsNone(c.active_pair)


# ---------------------------------------------------------------------------
# Rule 1: Beta-reduction (Constructor ⋈ Destructor)
# ---------------------------------------------------------------------------
class TestBetaReduction(unittest.TestCase):
    def test_aux_ports_splice_and_nodes_are_collected(self):
        net = HINNetwork()
        c = ConstructorNode("ctor")
        d = DestructorNode("dtor")
        net.register_node(c)
        net.register_node(d)

        # Attach observable leaves to every auxiliary wire.
        c1 = _leaf(net, "c1", c.a_1.type)
        c2 = _leaf(net, "c2", c.a_2.type)
        d1 = _leaf(net, "d1", d.a_1.type)
        d2 = _leaf(net, "d2", d.a_2.type)
        net.bind(c.a_1, c1)
        net.bind(c.a_2, c2)
        net.bind(d.a_1, d1)
        net.bind(d.a_2, d2)

        # Form the active pair through the principal ports.
        net.bind(c.p, d.p)
        self.assertEqual(len(net.active_pairs), 1)

        self.assertTrue(net.reduce_step())

        # Constructor & destructor are gone from the live node dictionary.
        self.assertNotIn("ctor", net.nodes)
        self.assertNotIn("dtor", net.nodes)

        # Auxiliary wires are spliced directly: c.a_i external <-> d.a_i external.
        self.assertIs(c1.target, d1)
        self.assertIs(d1.target, c1)
        self.assertIs(c2.target, d2)
        self.assertIs(d2.target, c2)

        # Net is fully reduced.
        self.assertFalse(net.reduce_step())
        net.validate_conservation()


# ---------------------------------------------------------------------------
# Rule 2: Duplication (Duplicator ⋈ Constructor)
# ---------------------------------------------------------------------------
class TestDuplication(unittest.TestCase):
    def test_constructor_is_duplicated(self):
        net = HINNetwork()
        dup = DuplicatorNode("dup")
        ctor = ConstructorNode("ctor")
        net.register_node(dup)
        net.register_node(ctor)

        # Observable leaves on every free wire.
        dup_l1 = _leaf(net, "dl1", dup.a_1.type)
        dup_l2 = _leaf(net, "dl2", dup.a_2.type)
        ctor_l1 = _leaf(net, "cl1", ctor.a_1.type)
        ctor_l2 = _leaf(net, "cl2", ctor.a_2.type)
        net.bind(dup.a_1, dup_l1)
        net.bind(dup.a_2, dup_l2)
        net.bind(ctor.a_1, ctor_l1)
        net.bind(ctor.a_2, ctor_l2)

        net.bind(dup.p, ctor.p)
        self.assertTrue(net.reduce_step())

        # Originals collected.
        self.assertNotIn("dup", net.nodes)
        self.assertNotIn("ctor", net.nodes)

        # Two constructor clones now drive the duplicator's two outputs.
        clones = [n for n in net.nodes.values() if isinstance(n, ConstructorNode)]
        self.assertEqual(len(clones), 2)
        self.assertIs(dup_l1.target.owner, clones[0])
        self.assertIs(dup_l2.target.owner, clones[1])

        # Two duplicators now feed the constructor's former argument wires.
        sub_dups = [n for n in net.nodes.values() if isinstance(n, DuplicatorNode)]
        self.assertEqual(len(sub_dups), 2)

        net.validate_conservation()


# ---------------------------------------------------------------------------
# Rule 3: Erasure (Eraser ⋈ Constructor)
# ---------------------------------------------------------------------------
class TestErasure(unittest.TestCase):
    def test_eraser_propagates_onto_free_wires(self):
        net = HINNetwork()
        eraser = EraserNode("eps")
        ctor = ConstructorNode("ctor")
        net.register_node(eraser)
        net.register_node(ctor)

        l1 = _leaf(net, "l1", ctor.a_1.type)
        l2 = _leaf(net, "l2", ctor.a_2.type)
        net.bind(ctor.a_1, l1)
        net.bind(ctor.a_2, l2)

        net.bind(eraser.p, ctor.p)
        self.assertTrue(net.reduce_step())

        self.assertNotIn("eps", net.nodes)
        self.assertNotIn("ctor", net.nodes)

        # Each former constructor wire now terminates in a fresh eraser.
        self.assertIsInstance(l1.target.owner, EraserNode)
        self.assertIsInstance(l2.target.owner, EraserNode)
        net.validate_conservation()


# ---------------------------------------------------------------------------
# Rule 4: Conditional Switch (Switch ⋈ Value)
# ---------------------------------------------------------------------------
class TestSwitch(unittest.TestCase):
    def _run_switch(self, flag: bool):
        net = HINNetwork()
        switch = SwitchNode("sw")
        val = ValueNode("flag", flag, MELLType.unit())
        net.register_node(switch)
        net.register_node(val)

        true_leaf = _leaf(net, "tb", switch.a_1.type)
        false_leaf = _leaf(net, "fb", switch.a_2.type)
        out_leaf = _leaf(net, "out", switch.a_3.type)
        net.bind(switch.a_1, true_leaf)
        net.bind(switch.a_2, false_leaf)
        net.bind(switch.a_3, out_leaf)

        net.bind(switch.p, val.p)
        self.assertTrue(net.reduce_step())

        self.assertNotIn("sw", net.nodes)
        self.assertNotIn("flag", net.nodes)
        return out_leaf, true_leaf, false_leaf

    def test_true_branch_selected(self):
        out_leaf, true_leaf, false_leaf = self._run_switch(True)
        # Output wire is spliced to the true branch.
        self.assertIs(out_leaf.target, true_leaf)
        self.assertIs(true_leaf.target, out_leaf)
        # False branch terminates in an eraser.
        self.assertIsInstance(false_leaf.target.owner, EraserNode)

    def test_false_branch_selected(self):
        out_leaf, true_leaf, false_leaf = self._run_switch(False)
        self.assertIs(out_leaf.target, false_leaf)
        self.assertIsInstance(true_leaf.target.owner, EraserNode)


# ---------------------------------------------------------------------------
# Rule 5: Causal Projection (Projection ⋈ context coordinate)
# ---------------------------------------------------------------------------
class TestCausalProjection(unittest.TestCase):
    def test_projection_emits_value_on_output(self):
        net = HINNetwork()
        proj = CausalProjectionNode("proj")
        coord = ValueNode("coord", "ledger/path/42", MELLType.unit())
        net.register_node(proj)
        net.register_node(coord)

        out_leaf = _leaf(net, "out", proj.a_1.type)
        net.bind(proj.a_1, out_leaf)

        net.bind(proj.p, coord.p)
        self.assertTrue(net.reduce_step())

        self.assertNotIn("proj", net.nodes)
        self.assertNotIn("coord", net.nodes)

        emitted = out_leaf.target.owner
        self.assertIsInstance(emitted, ValueNode)
        self.assertEqual(emitted.value, "ledger/path/42")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class TestRunToCompletion(unittest.TestCase):
    def test_chained_reductions_terminate(self):
        net = HINNetwork()
        # eraser ⋈ constructor whose aux wires meet a second eraser pair.
        eraser = EraserNode("eps")
        ctor = ConstructorNode("ctor")
        net.register_node(eraser)
        net.register_node(ctor)
        # Dangle the constructor's aux wires onto erasers so erasure cascades
        # produce eraser ⋈ eraser annihilations.
        e1 = EraserNode("e1", ctor.a_1.type)
        e2 = EraserNode("e2", ctor.a_2.type)
        net.register_node(e1)
        net.register_node(e2)
        net.bind(ctor.a_1, e1.p)
        net.bind(ctor.a_2, e2.p)
        net.bind(eraser.p, ctor.p)

        steps = net.run_to_completion()
        self.assertGreaterEqual(steps, 1)
        self.assertEqual(net.active_pairs, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
