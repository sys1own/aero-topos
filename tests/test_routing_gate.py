# -*- coding: utf-8 -*-
"""Tests for the strict Rust language-routing gate in the scaffold engine."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.scaffold.engine import ScaffoldEngine
from src.scaffold.language_router import (
    cargo_bypass_warning,
    is_native_crate_language,
)
from src.scaffold.recovery import RecoveryResult


class TestRouterGateHelpers(unittest.TestCase):
    def test_only_rust_is_native_crate(self):
        self.assertTrue(is_native_crate_language("rust"))
        for lang in ("python", "cpp", "c", "fortran", "unknown", ""):
            self.assertFalse(is_native_crate_language(lang))

    def test_bypass_warning_format(self):
        msg = cargo_bypass_warning("unknown")
        self.assertEqual(
            msg,
            "language router -> 'unknown' (Bypassing Rust cargo build layer; "
            "target is not a native crate component)",
        )


class TestRustRoutingGate(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)

    def test_non_rust_target_bypasses_cargo(self):
        asset = self.tmp / "asset.sh"
        asset.write_text("#!/bin/bash\necho hi\n")
        logs = []
        engine = ScaffoldEngine(logger=logs.append, verbose=True)

        # The cargo build wrapper must never be reached for a non-rust target.
        with mock.patch.object(
            ScaffoldEngine, "_build_rust_with_recovery"
        ) as build_mock:
            result = engine.scaffold(str(asset), name="asset", build=True, keep=False)

        build_mock.assert_not_called()
        self.assertIsNotNone(result.build)
        self.assertTrue(result.build["bypassed"])
        self.assertEqual(result.build["language"], "unknown")
        self.assertTrue(result.build["succeeded"])  # exits cleanly
        self.assertTrue(
            any("Bypassing Rust cargo build layer" in m for m in logs),
            f"expected bypass warning in logs, got: {logs}",
        )

    def test_rust_target_runs_cargo(self):
        src = self.tmp / "lib.rs"
        src.write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
        engine = ScaffoldEngine()

        with mock.patch.object(
            ScaffoldEngine,
            "_build_rust_with_recovery",
            return_value=RecoveryResult(succeeded=True),
        ) as build_mock:
            result = engine.scaffold(str(src), name="plain", build=True, keep=False)

        build_mock.assert_called_once()
        self.assertIsNotNone(result.build)
        self.assertFalse(result.build.get("bypassed", False))
        self.assertEqual(result.build["language"], "rust")

    def test_no_build_flag_never_bypasses_or_compiles(self):
        asset = self.tmp / "asset.txt"
        asset.write_text("just some text\n")
        engine = ScaffoldEngine()

        with mock.patch.object(
            ScaffoldEngine, "_build_rust_with_recovery"
        ) as build_mock:
            result = engine.scaffold(str(asset), name="asset", build=False, keep=False)

        build_mock.assert_not_called()
        self.assertIsNone(result.build)


if __name__ == "__main__":
    unittest.main()
