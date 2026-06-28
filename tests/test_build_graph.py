# -*- coding: utf-8 -*-
"""Tests for ``build_graph`` -- DAG construction, topological ordering, and tree rendering.

Covers:
* :func:`blueprint_to_dag` round-trip from a DSL source string
* Topological sort correctness (linear chain, diamond, independent targets)
* ``BuildGraph.levels`` parallel-stage grouping
* ``BuildGraph.render_tree`` visual output
* ``BuildGraph.to_build_context`` shape consumed by the engine
* ``parse_dsl_blueprint`` integration in ``blueprint_parser``
* ``plan`` CLI subcommand (both block-DSL and legacy INI)
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

import blueprint_lang
from build_graph import BuildGraph, TargetNode, blueprint_to_dag


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dag(src: str) -> BuildGraph:
    bp = blueprint_lang.load_source(src, "test.aero")
    return blueprint_to_dag(bp)


SIMPLE_DSL = '''
project "app" {
    version = "2.0.0"
}

target "lib" {
    language = "c"
    sources  = ["src/lib.c"]
}

target "main" {
    language = "c"
    sources  = ["src/main.c"]
    requires = ["lib"]
}
'''

DIAMOND_DSL = '''
project "diamond" {
    version = "1.0.0"
}

target "base" {
    language = "c"
    sources  = ["base.c"]
}

target "left" {
    language = "c"
    sources  = ["left.c"]
    requires = ["base"]
}

target "right" {
    language = "rust"
    sources  = ["right.rs"]
    requires = ["base"]
}

target "top" {
    language = "python"
    sources  = ["top.py"]
    requires = ["left", "right"]
}
'''


# ---------------------------------------------------------------------------
# blueprint_to_dag
# ---------------------------------------------------------------------------


class TestBlueprintToDag(unittest.TestCase):
    def test_simple_chain(self):
        g = _dag(SIMPLE_DSL)
        self.assertEqual(g.build_order, ["lib", "main"])
        self.assertEqual(g.dependency_map, {"lib": [], "main": ["lib"]})
        self.assertEqual(g.project_name, "app")
        self.assertEqual(g.project_version, "2.0.0")

    def test_diamond(self):
        g = _dag(DIAMOND_DSL)
        self.assertEqual(g.build_order[0], "base")
        self.assertEqual(g.build_order[-1], "top")
        self.assertIn("left", g.build_order[1:3])
        self.assertIn("right", g.build_order[1:3])

    def test_independent_targets(self):
        src = '''
project "p" { version = "1.0.0" }
target "a" { language = "c" sources = ["a.c"] }
target "b" { language = "c" sources = ["b.c"] }
target "c" { language = "c" sources = ["c.c"] }
'''
        g = _dag(src)
        self.assertEqual(len(g.build_order), 3)
        self.assertEqual(g.dependency_map, {"a": [], "b": [], "c": []})

    def test_target_node_fields(self):
        src = '''
project "p" { version = "1.0.0" }
target "t" {
    language = "rust"
    sources  = ["src/*.rs"]
    flags    = ["-O2"]
    defines  = ["NDEBUG"]
    output   = "build/t"
    optional = true
}
'''
        g = _dag(src)
        node = g.targets["t"]
        self.assertEqual(node.language, "rust")
        self.assertEqual(node.sources, ["src/*.rs"])
        self.assertEqual(node.flags, ["-O2"])
        self.assertEqual(node.defines, ["NDEBUG"])
        self.assertEqual(node.output, "build/t")
        self.assertTrue(node.optional)


# ---------------------------------------------------------------------------
# levels (parallel stage grouping)
# ---------------------------------------------------------------------------


class TestLevels(unittest.TestCase):
    def test_simple_chain_levels(self):
        g = _dag(SIMPLE_DSL)
        self.assertEqual(g.levels, [["lib"], ["main"]])

    def test_diamond_levels(self):
        g = _dag(DIAMOND_DSL)
        levels = g.levels
        self.assertEqual(levels[0], ["base"])
        self.assertEqual(sorted(levels[1]), ["left", "right"])
        self.assertEqual(levels[2], ["top"])

    def test_all_independent(self):
        src = '''
project "p" { version = "1.0.0" }
target "x" { language = "c" sources = ["x.c"] }
target "y" { language = "c" sources = ["y.c"] }
'''
        g = _dag(src)
        self.assertEqual(len(g.levels), 1)
        self.assertEqual(sorted(g.levels[0]), ["x", "y"])


# ---------------------------------------------------------------------------
# render_tree
# ---------------------------------------------------------------------------


class TestRenderTree(unittest.TestCase):
    def test_header_contains_project_info(self):
        tree = _dag(SIMPLE_DSL).render_tree()
        self.assertIn("Build Plan: app v2.0.0", tree)

    def test_contains_all_target_names(self):
        tree = _dag(DIAMOND_DSL).render_tree()
        for name in ("base", "left", "right", "top"):
            self.assertIn(name, tree)

    def test_contains_language_tags(self):
        tree = _dag(DIAMOND_DSL).render_tree()
        self.assertIn("[c]", tree)
        self.assertIn("[rust]", tree)
        self.assertIn("[python]", tree)

    def test_shows_requires(self):
        tree = _dag(SIMPLE_DSL).render_tree()
        self.assertIn("requires: lib", tree)

    def test_shows_optional(self):
        src = '''
project "p" { version = "1.0.0" }
target "t" { language = "c" sources = ["a.c"] optional = true }
'''
        tree = _dag(src).render_tree()
        self.assertIn("(optional)", tree)

    def test_shows_source_count(self):
        tree = _dag(SIMPLE_DSL).render_tree()
        self.assertIn("1 source pattern", tree)

    def test_summary_line(self):
        tree = _dag(DIAMOND_DSL).render_tree()
        self.assertIn("4 targets", tree)
        self.assertIn("3 stages", tree)

    def test_tree_connectors(self):
        tree = _dag(SIMPLE_DSL).render_tree()
        self.assertIn("├──", tree)
        self.assertIn("└──", tree)


# ---------------------------------------------------------------------------
# to_build_context
# ---------------------------------------------------------------------------


class TestToBuildContext(unittest.TestCase):
    def test_context_shape(self):
        ctx = _dag(SIMPLE_DSL).to_build_context()
        self.assertEqual(ctx["compilation_targets"], ["lib", "main"])
        self.assertIn("dependency_matrix", ctx)
        self.assertIn("graph", ctx)
        self.assertEqual(ctx["graph"]["targets"], ["lib", "main"])
        self.assertEqual(ctx["graph"]["dependencies"], {"lib": [], "main": ["lib"]})

    def test_target_metadata_in_context(self):
        ctx = _dag(SIMPLE_DSL).to_build_context()
        metadata = ctx["graph"]["target_metadata"]
        self.assertEqual(len(metadata), 2)
        names = [m["name"] for m in metadata]
        self.assertEqual(names, ["lib", "main"])


# ---------------------------------------------------------------------------
# parse_dsl_blueprint (blueprint_parser integration)
# ---------------------------------------------------------------------------


class TestParseDslBlueprint(unittest.TestCase):
    def test_returns_build_context(self):
        from blueprint_parser import parse_dsl_blueprint
        ctx = parse_dsl_blueprint(SIMPLE_DSL, "test.aero")
        self.assertEqual(ctx["workspace_status"], "stable_active")
        self.assertEqual(ctx["blueprint_format"], "dsl")
        self.assertEqual(ctx["compilation_targets"], ["lib", "main"])
        self.assertIn("graph", ctx)
        self.assertIn("active_optimizer_flags", ctx)
        self.assertIn("environment_targets", ctx)
        self.assertIn("resource_metrics", ctx)

    def test_parse_blueprint_routes_dsl(self):
        from blueprint_parser import parse_blueprint
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as fh:
            fh.write(SIMPLE_DSL)
            path = fh.name
        try:
            ctx = parse_blueprint(path)
            self.assertEqual(ctx["workspace_status"], "stable_active")
            self.assertEqual(ctx["blueprint_format"], "dsl")
            self.assertEqual(ctx["compilation_targets"], ["lib", "main"])
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# plan CLI subcommand
# ---------------------------------------------------------------------------


class TestPlanCommand(unittest.TestCase):
    def test_plan_dsl_blueprint(self):
        from main import main as cli_main
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as fh:
            fh.write(SIMPLE_DSL)
            path = fh.name
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli_main(["plan", "--blueprint", path])
            self.assertEqual(rc, 0)
            output = out.getvalue()
            self.assertIn("Build Plan: app v2.0.0", output)
            self.assertIn("lib", output)
            self.assertIn("main", output)
        finally:
            os.remove(path)

    def test_plan_legacy_ini_blueprint(self):
        from main import main as cli_main
        # The repo's blueprint.aero migrated to the living-blueprint schema, so the
        # legacy INI/JSON parser is exercised against a preserved legacy fixture.
        ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "legacy_blueprint.aero")
        if not os.path.exists(ini_path):
            self.skipTest("legacy_blueprint.aero fixture not found")
        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli_main(["plan", "--blueprint", ini_path])
        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("Build Plan (legacy INI/JSON)", output)
        self.assertIn("scanner", output)

    def test_plan_missing_file(self):
        from main import main as cli_main
        err = io.StringIO()
        with redirect_stderr(err):
            rc = cli_main(["plan", "--blueprint", "/no/such/file.aero"])
        self.assertEqual(rc, 1)

    def test_plan_invalid_dsl(self):
        from main import main as cli_main
        with tempfile.NamedTemporaryFile("w", suffix=".aero", delete=False, encoding="utf-8") as fh:
            fh.write('target "t" {\n    language = bad\n}\n')
            path = fh.name
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                rc = cli_main(["plan", "--blueprint", path])
            self.assertNotEqual(rc, 0)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
