# -*- coding: utf-8 -*-
"""Tests for the two-tiered structural 3-way AST merge engine."""

import os
import tempfile
import unittest
from pathlib import Path

from core.overlay.structural_merger import (
    StructuralMerger,
    detect_pcs_conflicts,
    extract_entities,
    pcs_triples,
    three_way_merge,
    verify_merge,
)


def _rust_ok():
    try:
        from core.parser.universal import _load_language
        _load_language("rust")
        return True
    except Exception:
        return False


BASE = "import os\n\ndef f(x):\n    return x\n"


class TestEntityExtraction(unittest.TestCase):
    def test_entities_and_ids(self):
        ef = extract_entities("import os\n\ndef f():\n    return 1\n\nclass C:\n    pass\n", "python")
        constructs = [(e.construct, e.name) for e in ef.entities]
        self.assertIn(("import", "import os"), constructs)
        self.assertIn(("function", "f"), constructs)
        self.assertIn(("type", "C"), constructs)

    def test_layout_captured(self):
        ef = extract_entities(BASE, "python")
        self.assertEqual(ef.prefix, "")
        self.assertTrue(any("\n\n" in g for g in ef.gaps))  # blank line between import and def


class TestTier1(unittest.TestCase):
    def test_left_only_change_applied_onto_right(self):
        left = "import os\n\ndef f(x):\n    return x * 2\n"
        right = "import os\n\ndef f(x):\n    return x\n\ndef g(y):\n    return y + 1\n"
        r = three_way_merge(BASE, left, right, "python")
        self.assertTrue(r.success)
        self.assertEqual(r.tier1_clean, 1)
        self.assertIn("return x * 2", r.text)   # user change kept
        self.assertIn("def g", r.text)          # generator addition kept

    def test_right_only_change_taken(self):
        right = "import os\n\ndef f(x):\n    return x + 1\n"
        r = three_way_merge(BASE, BASE, right, "python")
        self.assertTrue(r.success)
        self.assertIn("return x + 1", r.text)

    def test_user_added_function_preserved(self):
        left = BASE + "\ndef helper():\n    return 9\n"
        r = three_way_merge(BASE, left, BASE, "python")
        self.assertTrue(r.success)
        self.assertIn("def helper", r.text)


class TestCommutativeImports(unittest.TestCase):
    def test_independent_imports_no_conflict(self):
        left = "import os\nimport sys\n\ndef f(x):\n    return x\n"     # user added sys
        right = "import json\nimport os\n\ndef f(x):\n    return x\n"   # gen added json + reorder
        r = three_way_merge(BASE, left, right, "python")
        self.assertTrue(r.success, r.conflicts)
        for imp in ("import os", "import sys", "import json"):
            self.assertIn(imp, r.text)

    def test_no_duplicate_imports(self):
        left = "import os\n\ndef f(x):\n    return x\n"
        right = "import os\n\ndef f(x):\n    return x\n"
        r = three_way_merge(BASE, left, right, "python")
        self.assertEqual(r.text.count("import os"), 1)


class TestTier2(unittest.TestCase):
    BODY = "def f():\n    a = 1\n    b = 2\n    return a + b\n"

    def test_concurrent_non_overlapping_merged(self):
        left = self.BODY.replace("a = 1", "a = 100")
        right = self.BODY.replace("b = 2", "b = 200")
        r = three_way_merge(self.BODY, left, right, "python")
        self.assertTrue(r.success, r.conflicts)
        self.assertEqual(r.tier2_merged, 1)
        self.assertIn("a = 100", r.text)
        self.assertIn("b = 200", r.text)
        self.assertEqual(r.text.count("def f"), 1)
        self.assertTrue(verify_merge(r.text, "python").ok)

    def test_overlapping_edit_conflicts(self):
        left = self.BODY.replace("a = 1", "a = 100")
        right = self.BODY.replace("a = 1", "a = 999")
        r = three_way_merge(self.BODY, left, right, "python")
        self.assertFalse(r.success)
        self.assertTrue(r.has_conflicts)

    def test_user_insert_and_generator_edit(self):
        base = "def f():\n    a = 1\n    return a\n"
        left = "def f():\n    a = 1\n    log()\n    return a\n"
        right = "def f():\n    a = 2\n    return a\n"
        r = three_way_merge(base, left, right, "python")
        self.assertTrue(r.success, r.conflicts)
        self.assertIn("log()", r.text)
        self.assertIn("a = 2", r.text)
        self.assertTrue(verify_merge(r.text, "python").ok)


class TestPCS(unittest.TestCase):
    def test_pcs_triples_nonempty(self):
        triples = pcs_triples("def f():\n    return 1\n", "python")
        self.assertTrue(triples)
        self.assertTrue(all(len(t) == 3 for t in triples))

    def test_detect_conflicts(self):
        base = "def f():\n    a = 1\n    return a\n"
        left = base.replace("a = 1", "a = 2")
        right = base.replace("a = 1", "a = 3")
        # Same (parent, child) position given different successors/content.
        self.assertIsInstance(detect_pcs_conflicts(base, left, right, "python"), list)


class TestVerification(unittest.TestCase):
    def test_clean_parses(self):
        self.assertTrue(verify_merge("def f():\n    return 1\n", "python").ok)

    def test_syntax_error_rejected(self):
        v = verify_merge("def f(:\n", "python")
        self.assertFalse(v.ok)
        self.assertIn("syntax", v.reason)

    def test_duplicate_entity_rejected(self):
        v = verify_merge("def f():\n    return 1\ndef f():\n    return 2\n", "python")
        self.assertFalse(v.ok)
        self.assertIn("duplicate", v.reason)


class TestStructuralMergerDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = Path(self.ws, "m.py")
        self.merger = StructuralMerger(self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_writes_merged(self):
        base = "def f():\n    a = 1\n    b = 2\n    return a + b\n"
        self.file.write_text(base.replace("a = 1", "a = 100"))  # left (user edit) on disk
        right = base.replace("b = 2", "b = 200")
        outcome = self.merger.merge_file(self.file, base, right, language="python")
        self.assertTrue(outcome.accepted)
        text = self.file.read_text()
        self.assertIn("a = 100", text)
        self.assertIn("b = 200", text)

    def test_conflict_rejected_and_flagged(self):
        base = "def f():\n    a = 1\n    return a\n"
        left = base.replace("a = 1", "a = 100")
        self.file.write_text(left)
        right = base.replace("a = 1", "a = 999")
        bp = Path(self.ws, "blueprint.aero")
        bp.write_text('[system]\nname = "t"\n')
        outcome = self.merger.merge_file(self.file, base, right, language="python", blueprint_path=bp)
        self.assertFalse(outcome.accepted)
        self.assertEqual(self.file.read_text(), left)  # original left untouched
        self.assertIn("[merge_collisions]", bp.read_text())

    def test_verification_failure_rejected(self):
        # Force a verify failure by making right introduce a duplicate function
        # that, combined with left, yields a structurally unsound merge.
        base = "def f():\n    return 1\n"
        left = "def f():\n    return 2\n"   # user edit
        self.file.write_text(left)
        # right deletes f and re-adds an f with different body via add → handled,
        # but we simulate unsoundness by syntactically breaking through a build_fn.
        bp = Path(self.ws, "blueprint.aero")
        bp.write_text("")

        def failing_build(path):
            from core.toolchain.self_healing import Diagnostic
            return [Diagnostic("boom", str(path), code="E1")]

        right = "def f():\n    return 3\n"
        outcome = self.merger.merge_file(
            self.file, base, right, language="python", blueprint_path=bp, build_fn=failing_build)
        self.assertFalse(outcome.accepted)
        self.assertEqual(self.file.read_text(), left)  # restored / untouched
        self.assertIn("merge_collisions", bp.read_text())


class TestOverlayManagerIntegration(unittest.TestCase):
    """The structural merge is wired into OverlayManager as the reapply path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = Path(self.ws, "m.py")

    def tearDown(self):
        self.tmp.cleanup()

    def test_structural_reapply_merges(self):
        from src.overlay import OverlayManager, ReapplyStatus

        base = "def f():\n    a = 1\n    b = 2\n    return a + b\n"
        self.file.write_text(base)
        mgr = OverlayManager(self.ws)
        mgr.record_generated(self.file)                       # pristine baseline
        self.file.write_text(base.replace("a = 1", "a = 100"))  # user edit (Left)
        regenerated = base.replace("b = 2", "b = 200")          # Right
        status = mgr.structural_reapply(self.file, regenerated, language="python")
        self.assertEqual(status, ReapplyStatus.APPLIED)
        text = self.file.read_text()
        self.assertIn("a = 100", text)
        self.assertIn("b = 200", text)

    def test_structural_reapply_conflict_keeps_file(self):
        from src.overlay import OverlayManager, ReapplyStatus

        base = "def f():\n    a = 1\n    return a\n"
        self.file.write_text(base)
        mgr = OverlayManager(self.ws)
        mgr.record_generated(self.file)
        left = base.replace("a = 1", "a = 7")
        self.file.write_text(left)
        status = mgr.structural_reapply(
            self.file, base.replace("a = 1", "a = 9"), language="python",
            blueprint_path=Path(self.ws, "blueprint.aero"))
        self.assertEqual(status, ReapplyStatus.CONFLICT)
        self.assertEqual(self.file.read_text(), left)  # untouched


@unittest.skipUnless(_rust_ok(), "rust grammar missing")
class TestRust(unittest.TestCase):
    def test_rust_tier1(self):
        base = "fn f() -> i32 { 1 }\n"
        left = "fn f() -> i32 { 2 }\n"           # user edit
        right = "fn f() -> i32 { 1 }\n\nfn g() -> i32 { 3 }\n"  # gen added g
        r = three_way_merge(base, left, right, "rust")
        self.assertTrue(r.success, r.conflicts)
        self.assertIn("fn g", r.text)
        self.assertIn("2", r.text)


if __name__ == "__main__":
    unittest.main()
