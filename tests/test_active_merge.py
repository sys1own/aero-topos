# -*- coding: utf-8 -*-
"""Tests for the active in-tree merger (scaffold --merge-active)."""

import sys
import tempfile
import unittest
from pathlib import Path

from core.extensions import (
    PRIMARY_SUFFIX,
    _module_name_for,
    load_extensions,
)
from src.scaffold.active_merge import (
    MergeResult,
    find_compiled_library,
    merge_active,
)
from src.scaffold.repo_generator import detect_pymodule


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def _target(self, profile: str) -> Path:
        d = self.tmp / "target" / profile
        d.mkdir(parents=True, exist_ok=True)
        return d


class TestFindCompiledLibrary(_Tmp):
    def test_finds_unix_cdylib_in_debug(self):
        (self._target("debug") / "libanyon.so").write_bytes(b"\x7fELF")
        found = find_compiled_library(self.tmp, "anyon")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "libanyon.so")

    def test_prefers_release_over_debug(self):
        (self._target("debug") / "libanyon.so").write_bytes(b"d")
        (self._target("release") / "libanyon.so").write_bytes(b"r")
        found = find_compiled_library(self.tmp, "anyon")
        self.assertIn("release", str(found))

    def test_fallback_to_any_shared_lib(self):
        (self._target("debug") / "libsomethingelse.dylib").write_bytes(b"x")
        found = find_compiled_library(self.tmp, "anyon")
        self.assertIsNotNone(found)
        self.assertEqual(found.suffix, ".dylib")

    def test_returns_none_when_absent(self):
        self._target("debug")  # empty
        self.assertIsNone(find_compiled_library(self.tmp, "anyon"))


class TestMergeActive(_Tmp):
    def test_copies_and_names_after_module(self):
        (self._target("debug") / "libanyon.so").write_bytes(b"\x7fELF")
        dest = self.tmp / "extensions"
        result = merge_active(
            self.tmp, "anyon", "anyon_simulator", dest_dir=dest, load=False
        )
        self.assertIsInstance(result, MergeResult)
        self.assertTrue(result.merged)
        self.assertEqual(result.module_name, "anyon_simulator")
        copied = dest / f"anyon_simulator{PRIMARY_SUFFIX}"
        self.assertTrue(copied.is_file())
        self.assertEqual(result.destination, str(copied))

    def test_missing_library_reports_reason(self):
        self._target("debug")
        dest = self.tmp / "extensions"
        result = merge_active(self.tmp, "anyon", dest_dir=dest, load=False)
        self.assertFalse(result.merged)
        self.assertIn("no compiled shared library", result.reason)

    def test_defaults_module_name_to_crate(self):
        (self._target("debug") / "libanyon.so").write_bytes(b"x")
        dest = self.tmp / "extensions"
        result = merge_active(self.tmp, "anyon", dest_dir=dest, load=False)
        self.assertEqual(result.module_name, "anyon")


class TestExtensionLoader(_Tmp):
    def test_module_name_strips_suffix(self):
        self.assertEqual(_module_name_for(f"anyon{PRIMARY_SUFFIX}"), "anyon")
        self.assertEqual(
            _module_name_for("anyon.cpython-311-x86_64-linux-gnu.so"), "anyon"
        )
        self.assertIsNone(_module_name_for("notes.txt"))

    def test_load_extensions_empty_dir_is_noop(self):
        before = dict(sys.modules)
        loaded = load_extensions(self.tmp)
        self.assertNotIn("does_not_exist", loaded)
        # No stray modules were registered from an empty directory.
        self.assertEqual(set(sys.modules) - set(before), set())


class TestDetectPymodule(unittest.TestCase):
    def test_function_form(self):
        self.assertEqual(
            detect_pymodule("#[pymodule]\nfn anyon_sim(m: &Bound) {}"), "anyon_sim"
        )

    def test_declarative_mod_form(self):
        self.assertEqual(
            detect_pymodule("#[pymodule]\nmod anyon_sim {\n}"), "anyon_sim"
        )

    def test_none_when_absent(self):
        self.assertIsNone(detect_pymodule("fn main() {}"))


if __name__ == "__main__":
    unittest.main()
