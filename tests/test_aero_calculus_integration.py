"""End-to-end integration tests for the Aero-Calculus native pipeline.

Covers: compile a Python file -> .aeroc, rigidity validation, HIN-VM
reduction, lossless serialization round-trip, the plan topology renderer, and
type-safe graph evolution -- exercising main.py and evolve.py together.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import main
import evolve
from core.aeroc import deserialize_network, load_aeroc, save_aeroc, serialize_network
from core.aero_frontend import python_source_to_uast
from core.translator import UASTToHINTranslator


_SAMPLE = """
def f(x):
    return x

def g(a):
    y = a
    return y
"""


class TestAerocSerialization(unittest.TestCase):
    def _net(self):
        uast = python_source_to_uast(_SAMPLE)
        return UASTToHINTranslator().translate_uast(uast)

    def test_roundtrip_is_lossless(self):
        net = self._net()
        data = serialize_network(net)
        clone = deserialize_network(data)
        # Same node population.
        self.assertEqual(set(net.nodes), set(clone.nodes))
        # Same wiring: every port's endpoint is preserved.
        for nid, node in net.nodes.items():
            cnode = clone.nodes[nid]
            for p_orig, p_clone in zip(node.ports(), cnode.ports()):
                if p_orig.target is None:
                    self.assertIsNone(p_clone.target)
                else:
                    self.assertEqual(
                        (p_orig.target.owner.node_id, p_orig.target.name),
                        (p_clone.target.owner.node_id, p_clone.target.name),
                    )
        # Reconstructed net still satisfies Conservation of Edges.
        clone.validate_conservation()

    def test_disk_roundtrip(self):
        net = self._net()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prog.aeroc")
            save_aeroc(net, path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as fh:
                self.assertEqual(json.load(fh)["version"], "aeroc_v1")
            reloaded = load_aeroc(path)
            self.assertEqual(set(net.nodes), set(reloaded.nodes))


class TestFullBuildPipeline(unittest.TestCase):
    def test_compile_verify_reduce_serialize(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "prog.py")
            with open(src, "w") as fh:
                fh.write(_SAMPLE)
            out = os.path.join(d, "prog.aeroc")

            report = main.handle_aero_calculus_build(src, out, reduce_graph=True)

            # Compilation produced a non-trivial graph...
            self.assertGreater(report["compiled_nodes"], 0)
            # ...rigidity sweep ran...
            self.assertIn("verified", report["rigidity"])
            # ...reduction minimized it (reduced <= compiled)...
            self.assertLessEqual(report["reduced_nodes"], report["compiled_nodes"])
            # ...and the optimized graph is on disk and reloads cleanly.
            self.assertTrue(os.path.isfile(out))
            reloaded = load_aeroc(out)
            self.assertEqual(len(reloaded.nodes), report["reduced_nodes"])
            reloaded.validate_conservation()
            # A ledger of mutations was written alongside.
            self.assertTrue(os.path.isfile(os.path.join(d, "context.aero")))

    def test_build_cli_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "prog.py")
            with open(src, "w") as fh:
                fh.write(_SAMPLE)
            out = os.path.join(d, "prog.aeroc")
            rc = main.main(["build", "--source", src, "--aeroc-out", out])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(out))

    def test_plan_renders_aeroc_topology(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "prog.py")
            with open(src, "w") as fh:
                fh.write(_SAMPLE)
            out = os.path.join(d, "prog.aeroc")
            main.handle_aero_calculus_build(src, out, reduce_graph=False)
            parser = main.create_parser()
            args = parser.parse_args(["plan", "--aeroc", out])
            self.assertEqual(args.handler(args), 0)


class TestGraphEvolution(unittest.TestCase):
    def test_type_safe_evolution_minimizes_and_preserves(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "prog.py")
            with open(src, "w") as fh:
                fh.write(_SAMPLE)
            out = os.path.join(d, "prog.aeroc")
            # Compile WITHOUT reducing so evolution has work to do.
            main.handle_aero_calculus_build(src, out, reduce_graph=False)

            report = evolve.evolve_aeroc(out, generations=50)
            self.assertLessEqual(report["final_nodes"], report["start_nodes"])

            # The evolved graph remains well-typed and fully terminated.
            evolved = load_aeroc(out)
            evolved.validate_conservation()

    def test_crossover_preserves_typing(self):
        uast = python_source_to_uast(_SAMPLE)
        a = UASTToHINTranslator().translate_uast(uast)
        b = UASTToHINTranslator().translate_uast(uast)
        merged = evolve.topological_crossover(a, b)
        self.assertEqual(len(merged.nodes), len(a.nodes) + len(b.nodes))
        merged.validate_conservation()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
