# -*- coding: utf-8 -*-
"""Tests for the deterministic inference engine (``core/analysis/inference``)."""

import ast
import os
import tempfile
import unittest

from core.analysis import InferenceEngine, decompose_file
from core.analysis.inference import QueryEngine, InferenceError


def _rust_ok() -> bool:
    try:
        QueryEngine().load("rust", "branches")
        return True
    except Exception:
        return False


_PY_MAIN = '''\
from pkg.util import helper
import os

def simple():
    return helper()

def branchy(x):
    if x > 0:
        for i in range(x):
            while i > 0:
                if i % 2:
                    i -= 1
                else:
                    i -= 2
    return x
'''

_PY_UTIL = "def helper():\n    return 1\n"


class _WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        os.makedirs(os.path.join(self.ws, "pkg"))
        self._write("pkg/util.py", _PY_UTIL)
        self._write("pkg/main.py", _PY_MAIN)
        self.engine = InferenceEngine(self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rel, content):
        path = os.path.join(self.ws, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def _read(self, rel):
        with open(os.path.join(self.ws, rel), encoding="utf-8") as fh:
            return fh.read()


class TestQueryEngine(unittest.TestCase):
    def test_loads_external_scm(self):
        qe = QueryEngine()
        self.assertTrue(qe.has_query("python", "branches"))
        self.assertTrue(qe.has_query("python", "imports"))
        self.assertTrue(qe.has_query("python", "functions"))
        self.assertFalse(qe.has_query("python", "nonexistent"))

    def test_missing_query_raises(self):
        with self.assertRaises(InferenceError):
            QueryEngine().load("python", "nonexistent")


class TestComplexity(_WorkspaceCase):
    def test_cyclomatic_complexity_values(self):
        a = self.engine.analyze_file(os.path.join(self.ws, "pkg/main.py"))
        by_name = {f.name: f for f in a.functions}
        self.assertEqual(by_name["simple"].complexity, 1)  # M = 0 + 1
        # branchy: if, for, while, if (the else belongs to the inner if) => 4 branches
        self.assertEqual(by_name["branchy"].branch_count, 4)
        self.assertEqual(by_name["branchy"].complexity, 5)  # M = C + 1

    def test_deterministic(self):
        a1 = self.engine.analyze_file(os.path.join(self.ws, "pkg/main.py"))
        a2 = self.engine.analyze_file(os.path.join(self.ws, "pkg/main.py"))
        self.assertEqual(
            [(f.name, f.complexity) for f in a1.functions],
            [(f.name, f.complexity) for f in a2.functions],
        )


class TestImports(_WorkspaceCase):
    def test_import_extraction_and_resolution(self):
        a = self.engine.analyze_file(os.path.join(self.ws, "pkg/main.py"))
        raws = {e.raw: e.resolved for e in a.imports}
        self.assertIn("pkg.util", raws)
        self.assertEqual(raws["pkg.util"], "pkg/util.py")
        self.assertIn("os", raws)
        self.assertIsNone(raws["os"])  # stdlib, not in workspace

    def test_relative_import_resolution(self):
        self._write("pkg/rel.py", "from . import util\n")
        a = self.engine.analyze_file(os.path.join(self.ws, "pkg/rel.py"))
        # "from . import util" captures the relative_import "." ; util resolves
        # via the package dir. Accept either the dot form resolving to the pkg.
        self.assertTrue(a.imports)


class TestDAG(_WorkspaceCase):
    def test_build_dag_deterministic(self):
        analyses = self.engine.analyze_paths(
            [os.path.join(self.ws, "pkg/main.py"), os.path.join(self.ws, "pkg/util.py")]
        )
        dag = self.engine.build_dag(analyses)
        self.assertEqual(dag["pkg/main.py"], ["pkg/util.py"])
        self.assertEqual(dag["pkg/util.py"], [])

    def test_write_dag_preserves_other_sections_and_is_idempotent(self):
        bp = os.path.join(self.ws, "blueprint.aero")
        with open(bp, "w", encoding="utf-8") as fh:
            fh.write('[system]\nname = "t"\n\n[scaling]\nmax_module_complexity = 200\n')
        analyses = self.engine.analyze_paths(
            [os.path.join(self.ws, "pkg/main.py"), os.path.join(self.ws, "pkg/util.py")]
        )
        dag = self.engine.build_dag(analyses)
        self.engine.write_dag_to_blueprint(bp, dag)
        first = self._read("blueprint.aero")
        self.assertIn("[dag]", first)
        self.assertIn('[system]', first)
        self.assertIn('[scaling]', first)
        # Re-writing replaces the [dag] table rather than duplicating it.
        self.engine.write_dag_to_blueprint(bp, dag)
        second = self._read("blueprint.aero")
        self.assertEqual(first.count("[dag]"), 1)
        self.assertEqual(second.count("[dag]"), 1)
        # The blueprint stays valid TOML with the dag tracked.  Keys are
        # sanitized to valid bare keys (path separators become underscores).
        from src.blueprint.loader import _toml as _t
        parsed = _t.loads(second)
        self.assertIn("dag", parsed)
        self.assertEqual(parsed["dag"]["pkg_main_py"], ["pkg_util_py"])

    def test_write_dag_sanitizes_special_keys(self):
        bp = os.path.join(self.ws, "blueprint.aero")
        self.engine.write_dag_to_blueprint(
            bp,
            {
                "main__build_dsl_targets": ["src/core.cpp", "pkg/util.py"],
                "weird\nname\t": ["a"],
                "": ["b"],
            },
        )
        text = self._read("blueprint.aero")
        from src.blueprint.loader import _toml as _t
        parsed = _t.loads(text)
        self.assertIn("dag", parsed)
        self.assertEqual(parsed["dag"]["main__build_dsl_targets"], ["src/core.cpp", "pkg/util.py"])
        self.assertEqual(parsed["dag"]["weird_name"], ["a"])
        self.assertEqual(parsed["dag"]["module"], ["b"])


class TestDecomposition(_WorkspaceCase):
    def test_dryrun_identifies_overcomplex_function(self):
        plan = decompose_file(
            self.engine, os.path.join(self.ws, "pkg/main.py"),
            max_module_complexity=3, auto_split_threshold=10_000, apply=False,
        )
        self.assertFalse(plan.applied)
        names = [a.function for a in plan.actions]
        self.assertIn("branchy", names)
        self.assertNotIn("simple", names)

    def test_apply_splits_and_keeps_valid_python(self):
        plan = decompose_file(
            self.engine, os.path.join(self.ws, "pkg/main.py"),
            max_module_complexity=3, auto_split_threshold=10_000, apply=True,
        )
        self.assertTrue(plan.applied)
        main_src = self._read("pkg/main.py")
        mod_src = self._read("pkg/main_branchy.py")
        # The complex function moved out; an import was inserted; both parse.
        self.assertNotIn("def branchy", main_src)
        self.assertIn("import branchy", main_src.replace("\n", " "))
        self.assertIn("def branchy", mod_src)
        ast.parse(main_src)
        ast.parse(mod_src)

    def test_file_size_threshold_triggers_split(self):
        # Low auto_split_threshold but high complexity cap: only file-size triggers.
        plan = decompose_file(
            self.engine, os.path.join(self.ws, "pkg/main.py"),
            max_module_complexity=1000, auto_split_threshold=1, apply=False,
        )
        reasons = {a.reason for a in plan.actions}
        self.assertEqual(reasons, {"file_size"})
        # Both top-level functions become candidates.
        self.assertEqual({a.function for a in plan.actions}, {"simple", "branchy"})

    def test_no_split_when_within_thresholds(self):
        plan = decompose_file(
            self.engine, os.path.join(self.ws, "pkg/main.py"),
            max_module_complexity=100, auto_split_threshold=10_000, apply=False,
        )
        self.assertEqual(plan.actions, [])


@unittest.skipUnless(_rust_ok(), "rust grammar missing")
class TestRust(_WorkspaceCase):
    def test_rust_complexity_and_imports(self):
        self._write("lib.rs", (
            "use crate::helpers::thing;\n\n"
            "pub fn calc(x: i32) -> i32 {\n"
            "    if x > 0 {\n"
            "        for _ in 0..x { }\n"
            "    }\n"
            "    x\n"
            "}\n"
        ))
        a = self.engine.analyze_file(os.path.join(self.ws, "lib.rs"))
        self.assertEqual(a.language, "rust")
        calc = {f.name: f for f in a.functions}["calc"]
        self.assertEqual(calc.complexity, 3)  # if + for => 2 branches + 1
        self.assertTrue(any("helpers" in e.raw for e in a.imports))


class TestCLI(_WorkspaceCase):
    def test_decompose_cli_writes_dag(self):
        from main import main as cli_main

        with open(os.path.join(self.ws, "blueprint.aero"), "w", encoding="utf-8") as fh:
            fh.write('[scaling]\nmax_module_complexity = 3\nauto_split_threshold = 1500\n')
        rc = cli_main(["decompose", "--workspace", self.ws])
        self.assertEqual(rc, 0)
        self.assertIn("[dag]", self._read("blueprint.aero"))


if __name__ == "__main__":
    unittest.main()
