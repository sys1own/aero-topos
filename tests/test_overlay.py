# -*- coding: utf-8 -*-
"""Tests for the overlay system (``src/overlay`` + ``src/build/overlay_stage``)."""

import os
import tempfile
import unittest

from src.build.overlay_stage import apply_overlays_stage
from src.overlay import (
    OverlayError,
    OverlayManager,
    ReapplyStatus,
    apply_patch,
    is_empty_patch,
    make_patch,
)

_GEN_V1 = '''\
def greet(name):
    return "hello " + name


def farewell(name):
    return "bye " + name
'''

# A regenerated version where unrelated code changed but greet/farewell remain.
_GEN_V2 = '''\
def greet(name):
    return "hello " + name


def farewell(name):
    return "bye " + name


def added_by_blueprint():
    return 42
'''


class TestPatch(unittest.TestCase):
    def test_roundtrip(self):
        a = "a\nb\nc\n"
        b = "a\nB\nc\n"
        patch = make_patch(a, b)
        self.assertFalse(is_empty_patch(patch))
        merged, conflict = apply_patch(a, patch)
        self.assertFalse(conflict)
        self.assertEqual(merged, b)

    def test_empty_patch_for_identical(self):
        self.assertTrue(is_empty_patch(make_patch("x\ny\n", "x\ny\n")))

    def test_apply_tolerates_line_shift(self):
        # Edit made against base; applied to a version with extra leading lines.
        base = "one\ntwo\nthree\n"
        edited = "one\nTWO\nthree\n"
        patch = make_patch(base, edited)
        shifted = "header\nzero\none\ntwo\nthree\n"
        merged, conflict = apply_patch(shifted, patch)
        self.assertFalse(conflict)
        self.assertIn("TWO", merged)
        self.assertIn("header", merged)

    def test_conflict_when_context_gone(self):
        base = "alpha\nbeta\ngamma\n"
        edited = "alpha\nBETA\ngamma\n"
        patch = make_patch(base, edited)
        # Regenerated text no longer contains the edited region's context.
        regenerated = "completely\ndifferent\ncontent\n"
        merged, conflict = apply_patch(regenerated, patch)
        self.assertTrue(conflict)
        self.assertEqual(merged, regenerated)  # generated version kept


class TestOverlayManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = os.path.join(self.ws, "module.py")
        self.mgr = OverlayManager(self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def _generate(self, content):
        """Simulate the build generating a pristine file."""
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(content)
        self.mgr.record_generated(self.file)

    def test_commit_requires_baseline(self):
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        with self.assertRaises(OverlayError):
            self.mgr.commit_overlay(self.file)

    def test_commit_with_no_edits_returns_none(self):
        self._generate(_GEN_V1)
        self.assertIsNone(self.mgr.commit_overlay(self.file))

    def test_edit_survives_regeneration(self):
        # 1. Generate.
        self._generate(_GEN_V1)
        # 2. User edits a line.
        edited = _GEN_V1.replace('return "hello " + name', 'return "HELLO, " + name + "!"')
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(edited)
        # 3. Commit the overlay.
        patch = self.mgr.commit_overlay(self.file)
        self.assertIsNotNone(patch)
        # 4. Blueprint change triggers a rebuild -> file regenerated (pristine V2).
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(_GEN_V2)
        # 5. Overlay re-applied.
        status = self.mgr.reapply(self.file)
        self.assertEqual(status, ReapplyStatus.APPLIED)
        final = open(self.file, encoding="utf-8").read()
        self.assertIn('return "HELLO, " + name + "!"', final)  # manual edit preserved
        self.assertIn("def added_by_blueprint", final)  # new generated code present

    def test_recommit_after_regeneration_is_stable(self):
        self._generate(_GEN_V1)
        edited = _GEN_V1.replace('"bye " + name', '"goodbye " + name')
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(edited)
        self.mgr.commit_overlay(self.file)
        # Regenerate + reapply, then committing again should reproduce the overlay.
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(_GEN_V2)
        self.mgr.reapply(self.file)
        patch2 = self.mgr.commit_overlay(self.file)
        self.assertIsNotNone(patch2)
        self.assertIn("goodbye", patch2)

    def test_conflict_keeps_generated(self):
        self._generate(_GEN_V1)
        edited = _GEN_V1.replace('"hello " + name', '"hi " + name')
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(edited)
        self.mgr.commit_overlay(self.file)
        # Regenerate with the edited region's context removed entirely.
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write("def totally_new():\n    return None\n")
        status = self.mgr.reapply(self.file)
        self.assertEqual(status, ReapplyStatus.CONFLICT)
        final = open(self.file, encoding="utf-8").read()
        self.assertIn("totally_new", final)  # generated version kept


class TestBuildStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = os.path.join(self.ws, "pkg", "svc.py")
        os.makedirs(os.path.dirname(self.file))
        self.mgr = OverlayManager(self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, content):
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(content)

    def test_stage_reapplies_overlays(self):
        self._write(_GEN_V1)
        self.mgr.record_generated(self.file)
        self._write(_GEN_V1.replace('"bye " + name', '"farewell " + name'))
        self.mgr.commit_overlay(self.file)
        # Build regenerates pristine, then the stage runs.
        self._write(_GEN_V2)
        results = apply_overlays_stage(self.ws, enabled=True, log=lambda *_: None)
        self.assertEqual(set(results.values()), {ReapplyStatus.APPLIED})
        self.assertIn('"farewell " + name', open(self.file, encoding="utf-8").read())

    def test_stage_disabled_is_noop(self):
        self._write(_GEN_V1)
        self.mgr.record_generated(self.file)
        self._write(_GEN_V1.replace('"bye " + name', '"farewell " + name'))
        self.mgr.commit_overlay(self.file)
        self._write(_GEN_V2)
        results = apply_overlays_stage(self.ws, enabled=False, log=lambda *_: None)
        self.assertEqual(results, {})
        # File left as the regenerated pristine (no overlay applied).
        self.assertNotIn('"farewell " + name', open(self.file, encoding="utf-8").read())


class TestCommitOverlayCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = os.path.join(self.ws, "gen.py")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_commit_then_rebuild_preserves_edits(self):
        from main import main as cli_main

        mgr = OverlayManager(self.ws)
        # Generate + record baseline.
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(_GEN_V1)
        mgr.record_generated(self.file)
        # Manual edit.
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(_GEN_V1.replace('"hello " + name', '"HOWDY " + name'))

        rc = cli_main(["commit-overlay", self.file, "--workspace", self.ws])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.ws, ".overlays", "gen.py.patch")))

        # Rebuild regenerates pristine; the build stage re-applies the overlay.
        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write(_GEN_V2)
        apply_overlays_stage(self.ws, enabled=True, log=lambda *_: None)
        self.assertIn('"HOWDY " + name', open(self.file, encoding="utf-8").read())

    def test_cli_no_baseline_errors(self):
        from main import main as cli_main

        with open(self.file, "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        rc = cli_main(["commit-overlay", self.file, "--workspace", self.ws])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
