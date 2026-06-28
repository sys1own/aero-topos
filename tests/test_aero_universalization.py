"""Tests for the universal interlinking of the Aero-Calculus engine.

Covers UniversalHINNetwork (ledger coalescence + rigidity sweeps),
SHXTopologicalEvolution (crossover, type-safe mutation, O(1) compaction),
the spacetime path-integral cache + VP-Tree, and topological self-healing.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from decimal import Decimal

import evolve
import main
from core.aero_frontend import python_source_to_uast
from core.aeroc import load_aeroc, save_aeroc
from core.hin_vm import (
    AnomalyClosureError,
    ConstructorNode,
    DestructorNode,
    EraserNode,
    UniversalHINNetwork,
)
from core.spacetime_ledger import (
    CoordinateVector,
    PathIntegralCache,
    VantagePointTree,
    active_pair_signature,
)
from core.translator import UASTToHINTranslator

_SRC = """
def f(x):
    return x

def g(a):
    y = a
    return y
"""


def _net():
    return UASTToHINTranslator().translate_uast(python_source_to_uast(_SRC))


# ---------------------------------------------------------------------------
# UniversalHINNetwork
# ---------------------------------------------------------------------------
class TestUniversalHINNetwork(unittest.TestCase):
    def test_reduces_with_rigidity_sweeps(self):
        uni = UniversalHINNetwork.adopt(_net(), ledger_path="")
        steps = uni.run_to_completion()
        self.assertGreater(steps, 0)
        self.assertGreater(uni.rigidity_sweeps, 0)
        self.assertEqual(len(uni.nodes), 0)  # fully minimized
        uni.validate_conservation()

    def test_coalescence_hit_on_homomorphic_pairs(self):
        # Build two identical γ⋈ε active pairs -> same signature -> the second
        # reduction is served from the path-integral cache.
        uni = UniversalHINNetwork.adopt(
            UASTToHINTranslator().translate_uast(python_source_to_uast(_SRC)),
            ledger_path="",
        )
        uni.run_to_completion()
        # Two functions compile to two identical γ⋈ε pairs -> one coalescence.
        self.assertGreaterEqual(uni.coalescence_hits, 1)

    def test_rigidity_anomaly_aborts(self):
        # Force a collapsed boundary: give both nodes the same coordinate.
        uni = UniversalHINNetwork(ledger_path="")
        a = ConstructorNode("a")
        b = DestructorNode("b")
        uni.register_node(a)
        uni.register_node(b)
        collapsed = CoordinateVector("7", "7", "7", -1)
        a.coordinate = collapsed
        b.coordinate = CoordinateVector("7", "7", "7", -1)
        uni.bind(a.a_1, b.a_1)
        uni.bind(a.a_2, b.a_2)
        uni.bind(a.p, b.p)
        with self.assertRaises(AnomalyClosureError):
            uni.reduce_step()


# ---------------------------------------------------------------------------
# Path-integral cache + VP-Tree
# ---------------------------------------------------------------------------
class TestSpacetimeCaches(unittest.TestCase):
    def test_path_integral_cache_reinforces(self):
        cache = PathIntegralCache()
        self.assertIsNone(cache.lookup("sig"))
        r1 = cache.record("sig", persist=False)
        self.assertEqual(r1["weight"], 1)
        r2 = cache.record("sig", persist=False)
        self.assertEqual(r2["weight"], 2)
        self.assertIsNotNone(cache.lookup("sig"))

    def test_vptree_nearest(self):
        items = [
            ("origin", CoordinateVector("0", "0", "0", 0)),
            ("far", CoordinateVector("100", "100", "100", 1)),
            ("near", CoordinateVector("1", "1", "1", 2)),
        ]
        tree = VantagePointTree(items)
        key, dist = tree.nearest(CoordinateVector("1", "1", "0", 3))
        self.assertIn(key, ("near", "origin"))
        self.assertIsInstance(dist, Decimal)


# ---------------------------------------------------------------------------
# SHX topological evolution
# ---------------------------------------------------------------------------
class TestSHXEvolution(unittest.TestCase):
    def test_crossover_preserves_conservation(self):
        shx = evolve.SHXTopologicalEvolution()
        child = shx.execute_shx_crossover(_net(), _net())
        child.validate_conservation()
        self.assertGreater(len(child.nodes), 0)

    def test_type_safe_mutation_preserves_typing(self):
        shx = evolve.SHXTopologicalEvolution(seed=3)
        net = _net()
        ctors_before = sum(1 for n in net.nodes.values() if isinstance(n, ConstructorNode))
        mutated = shx.apply_type_safe_mutation(net, mutation_rate=1.0)
        self.assertGreater(mutated, 0)
        # Class swaps preserve every port -> conservation still holds.
        net.validate_conservation()
        # γ -> γ⁻¹ swaps actually changed agent classes.
        ctors_after = sum(1 for n in net.nodes.values() if isinstance(n, ConstructorNode))
        self.assertNotEqual(ctors_before, ctors_after)

    def test_compaction_reclaims_dead_code(self):
        shx = evolve.SHXTopologicalEvolution()
        net = _net()
        before = len(net.nodes)
        reclaimed = shx.compact(net)
        self.assertEqual(reclaimed, before - len(net.nodes))
        self.assertGreater(reclaimed, 0)

    def test_fitness_pareto_proxy(self):
        shx = evolve.SHXTopologicalEvolution()
        fitness = shx.evaluate_fitness(_net())
        self.assertIn("accuracy", fitness)
        self.assertGreaterEqual(fitness["accuracy"], 0.0)
        self.assertLessEqual(fitness["accuracy"], 1.0)


# ---------------------------------------------------------------------------
# Topological self-healing
# ---------------------------------------------------------------------------
class TestTopologicalHealing(unittest.TestCase):
    def test_reify_and_heal_unterminated_edge(self):
        from orchestrator import TopologicalSelfHealer
        from error_interceptor import reify_parse_failure_as_port

        net = _net()
        faulty = reify_parse_failure_as_port(net, "unresolved dependency: foo")
        self.assertIsNone(faulty.target)

        healer = TopologicalSelfHealer()
        self.assertEqual(len(healer.find_unterminated_ports(net)), 1)
        ok = healer.heal_unterminated_interface(net, faulty)
        self.assertTrue(ok)
        # The broken edge is now terminated; the whole net is conservative.
        self.assertIsNotNone(faulty.target)
        net.validate_conservation()
        self.assertEqual(len(healer.find_unterminated_ports(net)), 0)


# ---------------------------------------------------------------------------
# End-to-end CLI universalization
# ---------------------------------------------------------------------------
class TestUniversalCLI(unittest.TestCase):
    def _compile(self, d, reduce_graph=True):
        src = os.path.join(d, "prog.py")
        with open(src, "w") as fh:
            fh.write(_SRC)
        out = os.path.join(d, "prog.aeroc")
        report = main.handle_aero_calculus_build(src, out, reduce_graph=reduce_graph)
        return out, report

    def test_build_uses_universal_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            out, report = self._compile(d)
            self.assertIn("verified", report["rigidity"])
            self.assertIn("mitosis_split", report)
            self.assertTrue(os.path.isfile(out))
            load_aeroc(out).validate_conservation()
            self.assertTrue(os.path.isfile(os.path.join(d, "context.aero")))

    def test_evolve_cli_with_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = self._compile(d, reduce_graph=False)
            rc = main.main(["evolve", "--aeroc", out, "--generations", "30",
                            "--mutation-rate", "0.3"])
            self.assertEqual(rc, 0)
            load_aeroc(out).validate_conservation()

    def test_heal_cli_rewires_graph(self):
        from error_interceptor import reify_parse_failure_as_port

        with tempfile.TemporaryDirectory() as d:
            out, _ = self._compile(d, reduce_graph=False)
            net = load_aeroc(out)
            reify_parse_failure_as_port(net, "synthetic fault")
            save_aeroc(net, out)
            rc = main.main(["heal", "--aeroc", out])
            self.assertEqual(rc, 0)
            load_aeroc(out).validate_conservation()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
