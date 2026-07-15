# -*- coding: utf-8 -*-
"""Tests for delegated pre-write validation in the scaffold engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.scaffold import ScaffoldEngine


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)


class TestPreWriteValidation(_Tmp):
    def test_validation_cmd_success_promotes_distribution(self):
        src = self.tmp / "src" / "main.py"
        src.parent.mkdir(parents=True)
        src.write_text("print('hello')\n")
        dist = self.tmp / "dist"

        context = {"validation": {"validation_cmd": "python3 -m py_compile main.py"}}
        engine = ScaffoldEngine(verbose=False)
        result = engine.scaffold(
            source_entry=str(src),
            name="py_test",
            distribution_directory=dist,
            build=False,
            context=context,
        )

        self.assertTrue(dist.is_dir())
        self.assertEqual(Path(result.workspace).resolve(), dist.resolve())
        self.assertTrue((dist / "main.py").is_file())
        build = result.build or {}
        self.assertTrue(build.get("succeeded", True))
        self.assertTrue(build.get("pre_write_validation", {}).get("succeeded"))

    def test_validation_cmd_failure_blocks_distribution(self):
        src = self.tmp / "src" / "main.py"
        src.parent.mkdir(parents=True)
        src.write_text("print('hello')\n")
        dist = self.tmp / "dist"

        context = {"validation": {"validation_cmd": "sh -c 'echo failure; exit 1'"}}
        engine = ScaffoldEngine(verbose=False)
        result = engine.scaffold(
            source_entry=str(src),
            name="py_test",
            distribution_directory=dist,
            build=False,
            context=context,
        )

        self.assertFalse(dist.exists())
        build = result.build or {}
        self.assertFalse(build.get("succeeded", True))
        self.assertFalse(build.get("pre_write_validation", {}).get("succeeded"))
        self.assertIn("failure", build.get("pre_write_validation", {}).get("output", ""))

    def test_no_validation_cmd_commits_by_default(self):
        src = self.tmp / "src" / "main.py"
        src.parent.mkdir(parents=True)
        src.write_text("print('hello')\n")
        dist = self.tmp / "dist"

        engine = ScaffoldEngine(verbose=False)
        result = engine.scaffold(
            source_entry=str(src),
            name="py_test",
            distribution_directory=dist,
            build=False,
        )

        self.assertTrue(dist.is_dir())
        self.assertEqual(Path(result.workspace).resolve(), dist.resolve())
