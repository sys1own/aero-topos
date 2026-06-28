# -*- coding: utf-8 -*-
"""Tests for the AST registry ingestion (``src/registry``)."""

import os
import tempfile
import unittest

from src.blueprint import load_blueprint
from src.registry import ASTDatabase, ingest_context, semantic_hash
from src.registry.ingest import IngestError

try:
    import tree_sitter_rust  # noqa: F401

    _HAS_RUST = True
except ImportError:  # pragma: no cover
    _HAS_RUST = False


_PY_SOURCE = '''\
"""A small module."""


def add(a, b):
    return a + b


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
'''

# Same code, only comments and whitespace differ.
_PY_SOURCE_RECOMMENTED = '''\
"""A small module."""
# a leading comment


def add(a, b):
    # adds two numbers
    return a + b



class Point:

    def __init__(self, x, y):
        self.x = x   # store x
        self.y = y
'''

_RS_SOURCE = """\
// a rust module
struct Point {
    x: i32,
    y: i32,
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""

_RS_SOURCE_RECOMMENTED = """\
struct Point {
    x: i32,   // horizontal
    y: i32,   // vertical
}

// adds two numbers
fn add(a: i32, b: i32) -> i32 {
    /* sum */
    a + b
}
"""


class TestPythonHashing(unittest.TestCase):
    def test_hash_is_deterministic(self):
        h1 = semantic_hash(_PY_SOURCE, "python")
        h2 = semantic_hash(_PY_SOURCE, "python")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # sha-256 hex

    def test_comments_and_whitespace_do_not_change_hash(self):
        self.assertEqual(
            semantic_hash(_PY_SOURCE, "python"),
            semantic_hash(_PY_SOURCE_RECOMMENTED, "python"),
        )

    def test_structural_change_changes_hash(self):
        altered = _PY_SOURCE + "\n\ndef sub(a, b):\n    return a - b\n"
        self.assertNotEqual(
            semantic_hash(_PY_SOURCE, "python"),
            semantic_hash(altered, "python"),
        )


@unittest.skipUnless(_HAS_RUST, "tree-sitter-rust not installed")
class TestRustHashing(unittest.TestCase):
    def test_parses_and_hashes(self):
        h = semantic_hash(_RS_SOURCE, "rust")
        self.assertEqual(len(h), 64)

    def test_comments_and_whitespace_do_not_change_hash(self):
        self.assertEqual(
            semantic_hash(_RS_SOURCE, "rust"),
            semantic_hash(_RS_SOURCE_RECOMMENTED, "rust"),
        )


class TestIngestContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.db_path = os.path.join(self.root, "registry.db")
        self.blueprint = os.path.join(self.root, "blueprint.aero")
        with open(self.blueprint, "w", encoding="utf-8") as fh:
            fh.write(
                '[system]\nname = "t"\nversion = "0.1.0"\n\n'
                "[context_registry]\n\n[scaling]\nauto_split_threshold = 1500\n"
            )

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_ingest_python_stores_metadata(self):
        py = self._write("mod.py", _PY_SOURCE)
        result = ingest_context(
            "mymod", py, db_path=self.db_path, blueprint_path=self.blueprint
        )
        self.assertEqual(len(result.files), 1)
        fr = result.files[0]
        self.assertEqual(fr.language, "python")
        self.assertIn("add", fr.functions)
        self.assertIn("Point", fr.types)

        with ASTDatabase(self.db_path) as db:
            entries = db.entries_for_context("mymod")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].semantic_hash, fr.semantic_hash)
        self.assertIn("add", entries[0].functions)

    def test_hash_consistent_across_runs(self):
        py = self._write("mod.py", _PY_SOURCE)
        r1 = ingest_context("c", py, db_path=self.db_path, blueprint_path=self.blueprint)
        r2 = ingest_context("c", py, db_path=self.db_path, blueprint_path=self.blueprint)
        self.assertEqual(r1.files[0].semantic_hash, r2.files[0].semantic_hash)
        # Re-ingest must not duplicate rows (upsert keyed on context+path).
        with ASTDatabase(self.db_path) as db:
            self.assertEqual(len(db.entries_for_context("c")), 1)

    def test_comment_only_edit_keeps_hash(self):
        py = self._write("mod.py", _PY_SOURCE)
        r1 = ingest_context("c", py, db_path=self.db_path, blueprint_path=self.blueprint)
        self._write("mod.py", _PY_SOURCE_RECOMMENTED)
        r2 = ingest_context("c", py, db_path=self.db_path, blueprint_path=self.blueprint)
        self.assertEqual(r1.files[0].semantic_hash, r2.files[0].semantic_hash)

    def test_blueprint_context_registry_updated(self):
        py = self._write("mod.py", _PY_SOURCE)
        result = ingest_context(
            "payroll", py, db_path=self.db_path, blueprint_path=self.blueprint
        )
        self.assertTrue(result.blueprint_updated)
        bp = load_blueprint(self.blueprint)
        self.assertIn("payroll", bp.context_registry)
        self.assertEqual(bp.context_registry["payroll"].language, "python")

        # Second ingest of the same context must not re-register.
        result2 = ingest_context(
            "payroll", py, db_path=self.db_path, blueprint_path=self.blueprint
        )
        self.assertFalse(result2.blueprint_updated)

    def test_directory_ingest_filters_by_language(self):
        self._write("a.py", _PY_SOURCE)
        self._write("notes.txt", "not source")
        result = ingest_context(
            "dir", self.root, language="python",
            db_path=self.db_path, blueprint_path=self.blueprint,
        )
        langs = {f.language for f in result.files}
        self.assertEqual(langs, {"python"})
        self.assertTrue(any(f.path.endswith("a.py") for f in result.files))

    def test_missing_path_raises(self):
        with self.assertRaises(IngestError):
            ingest_context("x", os.path.join(self.root, "nope.py"), db_path=self.db_path)

    @unittest.skipUnless(_HAS_RUST, "tree-sitter-rust not installed")
    def test_ingest_rust(self):
        rs = self._write("lib.rs", _RS_SOURCE)
        result = ingest_context(
            "rustlib", rs, db_path=self.db_path, blueprint_path=self.blueprint
        )
        self.assertEqual(len(result.files), 1)
        fr = result.files[0]
        self.assertEqual(fr.language, "rust")
        self.assertIn("add", fr.functions)
        self.assertIn("Point", fr.types)


class TestIngestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_ingest_and_list(self):
        import io
        from contextlib import redirect_stdout

        from main import main as cli_main

        py_path = os.path.join(self.root, "svc.py")
        with open(py_path, "w", encoding="utf-8") as fh:
            fh.write(_PY_SOURCE)

        rc = cli_main(["ingest", "--workspace", self.root, "--path", py_path])
        self.assertEqual(rc, 0)
        # Blueprint was created/updated with the derived context name.
        bp = load_blueprint(os.path.join(self.root, "blueprint.aero"))
        self.assertTrue(bp.context_registry)  # derived from the dir name

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cli_main(["ingest", "--workspace", self.root, "--list"])
        self.assertEqual(rc, 0)
        self.assertIn("Ingested contexts:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
