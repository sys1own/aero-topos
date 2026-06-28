# -*- coding: utf-8 -*-
"""Unit tests for src.precision.selector."""

import unittest

import networkx as nx

from src.precision.selector import PrecisionRecommendation, PrecisionSelector


class TestPrecisionRecommendation(unittest.TestCase):
    def test_to_dict(self):
        rec = PrecisionRecommendation(
            location="src/math.rs",
            zone="solver_zone",
            current="double",
            recommended="quad",
            reason="transcendental call",
        )
        d = rec.to_dict()
        self.assertEqual(d["location"], "src/math.rs")
        self.assertEqual(d["recommended"], "quad")


class TestPrecisionSelectorConfig(unittest.TestCase):
    def test_defaults(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.default_float, "double")
        self.assertEqual(ps.arbitrary_bits, 128)
        self.assertFalse(ps.auto_detect_need)
        self.assertEqual(ps.per_zone_overrides, {})

    def test_custom_config(self):
        config = {
            "precision_shield": {
                "default_float": "quad",
                "arbitrary_precision_bits": 256,
                "auto_detect_need": True,
                "per_zone_overrides": {
                    "crypto": "quad",
                    "ml": {"float": "double"},
                },
            }
        }
        ps = PrecisionSelector(config=config)
        self.assertEqual(ps.default_float, "quad")
        self.assertEqual(ps.arbitrary_bits, 256)
        self.assertTrue(ps.auto_detect_need)
        self.assertEqual(ps.per_zone_overrides["crypto"], "quad")
        self.assertEqual(ps.per_zone_overrides["ml"], "double")


class TestFloatForZone(unittest.TestCase):
    def test_default_zone(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.float_for_zone("anything"), "double")

    def test_overridden_zone(self):
        config = {"precision_shield": {"per_zone_overrides": {"crypto": "quad"}}}
        ps = PrecisionSelector(config=config)
        self.assertEqual(ps.float_for_zone("crypto"), "quad")
        self.assertEqual(ps.float_for_zone("other"), "double")


class TestRecommend(unittest.TestCase):
    def test_without_auto_detect(self):
        config = {"precision_shield": {"per_zone_overrides": {"zone1": "quad"}}}
        ps = PrecisionSelector(config=config)
        decisions = ps.recommend()
        self.assertEqual(decisions, {"zone1": "quad"})

    def test_with_auto_detect(self):
        config = {"precision_shield": {"auto_detect_need": True}}
        ps = PrecisionSelector(config=config)
        # Create a mock UAST graph with transcendental function nodes
        g = nx.DiGraph()
        g.add_node(0, metadata={"uast_kind": "uast_call", "name": "exp"},
                   data={"source": ""}, source_location=["math.py"])
        decisions = ps.recommend(g)
        self.assertIn("math.py", decisions)
        self.assertEqual(decisions["math.py"], "quad")


class TestAnalyzeUAST(unittest.TestCase):
    def test_transcendental_detection(self):
        config = {"precision_shield": {"auto_detect_need": True}}
        ps = PrecisionSelector(config=config)
        g = nx.DiGraph()
        g.add_node(0, metadata={"uast_kind": "uast_call", "name": "sin"},
                   data={"source": ""}, source_location=["trig.py"])
        recs = ps.analyze_uast(g)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].location, "trig.py")
        self.assertIn("transcendental", recs[0].reason)

    def test_iterative_solver_detection(self):
        ps = PrecisionSelector()
        g = nx.DiGraph()
        g.add_node(0, metadata={"name": "conjugate_gradient_solve"},
                   data={"source": ""}, source_location=["solver.py"])
        recs = ps.analyze_uast(g)
        self.assertEqual(len(recs), 1)
        self.assertIn("iterative", recs[0].reason)

    def test_linear_algebra_detection(self):
        ps = PrecisionSelector()
        g = nx.DiGraph()
        g.add_node(0, metadata={"name": ""},
                   data={"source": "cholesky decomposition"},
                   source_location=["linalg.py"])
        recs = ps.analyze_uast(g)
        self.assertEqual(len(recs), 1)
        self.assertIn("ill-conditioned", recs[0].reason)

    def test_no_detection_for_normal_code(self):
        ps = PrecisionSelector()
        g = nx.DiGraph()
        g.add_node(0, metadata={"name": "process_data"},
                   data={"source": "x = 1 + 2"},
                   source_location=["normal.py"])
        recs = ps.analyze_uast(g)
        self.assertEqual(len(recs), 0)


class TestCompilerFlags(unittest.TestCase):
    def test_c_double(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.compiler_flags("c", "double"), [])

    def test_c_quad(self):
        ps = PrecisionSelector()
        flags = ps.compiler_flags("c", "quad")
        self.assertIn("-DAERO_FLOAT=__float128", flags)
        self.assertIn("-lquadmath", flags)

    def test_c_arbitrary(self):
        ps = PrecisionSelector()
        flags = ps.compiler_flags("c", "arbitrary")
        self.assertIn("-lgmp", flags)
        self.assertIn("-lmpfr", flags)

    def test_rust_quad(self):
        ps = PrecisionSelector()
        flags = ps.compiler_flags("rust", "quad")
        self.assertIn("--cfg", flags)
        self.assertIn("aero_quad", flags)

    def test_rust_arbitrary(self):
        config = {"precision_shield": {"arbitrary_precision_bits": 256}}
        ps = PrecisionSelector(config=config)
        flags = ps.compiler_flags("rust", "arbitrary")
        self.assertIn("--cfg", flags)
        self.assertTrue(any("256" in f for f in flags))

    def test_python_always_empty(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.compiler_flags("python", "double"), [])
        self.assertEqual(ps.compiler_flags("python", "quad"), [])
        self.assertEqual(ps.compiler_flags("python", "arbitrary"), [])

    def test_cpp_alias(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.compiler_flags("c++", "quad"), ps.compiler_flags("cpp", "quad"))


class TestTypeMapping(unittest.TestCase):
    def test_c_double(self):
        ps = PrecisionSelector()
        m = ps.type_mapping("c", "double")
        self.assertEqual(m["type"], "double")

    def test_rust_quad(self):
        ps = PrecisionSelector()
        m = ps.type_mapping("rust", "quad")
        self.assertEqual(m["type"], "f128::f128")
        self.assertIn("f128", m["crates"])

    def test_python_arbitrary(self):
        ps = PrecisionSelector()
        m = ps.type_mapping("python", "arbitrary")
        self.assertEqual(m["type"], "gmpy2.mpfr")

    def test_unknown_language(self):
        ps = PrecisionSelector()
        m = ps.type_mapping("fortran", "double")
        self.assertEqual(m, {"type": "double"})


class TestRequiredDependencies(unittest.TestCase):
    def test_double_no_deps(self):
        ps = PrecisionSelector()
        self.assertEqual(ps.required_dependencies("double"), {})

    def test_quad_deps(self):
        ps = PrecisionSelector()
        deps = ps.required_dependencies("quad")
        self.assertIn("system", deps)
        self.assertIn("libquadmath", deps["system"])
        self.assertIn("rust", deps)
        self.assertIn("python", deps)

    def test_arbitrary_deps(self):
        ps = PrecisionSelector()
        deps = ps.required_dependencies("arbitrary")
        self.assertIn("libgmp", deps["system"])
        self.assertIn("rug", deps["rust"])
        self.assertIn("gmpy2", deps["python"])


class TestEpsilonFor(unittest.TestCase):
    def test_double_epsilon(self):
        ps = PrecisionSelector()
        eps = ps.epsilon_for("double")
        self.assertAlmostEqual(eps, 2.0 ** -52)

    def test_quad_epsilon(self):
        ps = PrecisionSelector()
        eps = ps.epsilon_for("quad")
        self.assertAlmostEqual(eps, 2.0 ** -112)

    def test_arbitrary_epsilon(self):
        config = {"precision_shield": {"arbitrary_precision_bits": 256}}
        ps = PrecisionSelector(config=config)
        eps = ps.epsilon_for("arbitrary")
        self.assertAlmostEqual(eps, 2.0 ** -255)


if __name__ == "__main__":
    unittest.main()
