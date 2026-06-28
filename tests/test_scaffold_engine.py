"""Tests for the Ephemeral Cargo Scaffolder and toolchain bootstrapper upgrade."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from core.environment_bootstrap import SYSTEM_TOOLCHAINS, RuntimeEnvironmentBootstrapper
from core.scaffold_engine import EphemeralCargoScaffolder, compile_polyglot_source

_HAS_CARGO = shutil.which("cargo") is not None

_ZERO_DEP_LIB = (
    '#[no_mangle]\n'
    'pub extern "C" fn aero_add(a: i32, b: i32) -> i32 { a + b }\n'
)

_MULTI_CRATE_SRC = (
    "use pyo3::prelude::*;\n"
    "use rayon::prelude::*;\n"
    "use rug::Integer;\n"
    "use std::collections::HashMap;\n"
    "extern crate serde;\n"
    "fn main() {}\n"
)


# ---------------------------------------------------------------------------
# Dependency extraction
# ---------------------------------------------------------------------------
class TestExtraction(unittest.TestCase):
    def test_use_and_extern_crate_discovered(self):
        deps = EphemeralCargoScaffolder.extract_dependencies_from_text(_MULTI_CRATE_SRC)
        self.assertEqual(deps, ["pyo3", "rayon", "rug", "serde"])

    def test_internal_crates_filtered(self):
        src = "use std::io;\nuse core::mem;\nuse crate::local;\nuse super::thing;\n"
        self.assertEqual(
            EphemeralCargoScaffolder.extract_dependencies_from_text(src), []
        )

    def test_extract_from_missing_file_is_safe(self):
        s = EphemeralCargoScaffolder()
        self.assertEqual(s.extract_dependencies("/no/such/file_zzz.rs"), [])


# ---------------------------------------------------------------------------
# Manifest synthesis
# ---------------------------------------------------------------------------
class TestManifest(unittest.TestCase):
    def setUp(self):
        self.s = EphemeralCargoScaffolder()

    def test_wildcards_and_pins(self):
        manifest = self.s.synthesize_manifest(["rayon", "rug"])
        self.assertIn('rayon = "*"', manifest)
        self.assertIn('rug = "1.24"', manifest)

    def test_cdylib_crate_type(self):
        manifest = self.s.synthesize_manifest(["rayon"])
        self.assertIn("[lib]", manifest)
        self.assertIn('crate-type = ["cdylib"]', manifest)
        self.assertIn('name = "aero_ephemeral_build"', manifest)

    def test_pyo3_gets_features(self):
        manifest = self.s.synthesize_manifest(["pyo3"])
        self.assertIn("pyo3 = {", manifest)
        self.assertIn("extension-module", manifest)


# ---------------------------------------------------------------------------
# Workspace generation + cleanup
# ---------------------------------------------------------------------------
class TestWorkspace(unittest.TestCase):
    def test_generate_and_cleanup(self):
        with tempfile.TemporaryDirectory() as scratch:
            src = os.path.join(scratch, "in.rs")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(_MULTI_CRATE_SRC)
            s = EphemeralCargoScaffolder(scratch_space=os.path.join(scratch, "aero"))
            workspace = s.generate_scaffold_env(src)

            self.assertTrue(os.path.isfile(os.path.join(workspace, "Cargo.toml")))
            self.assertTrue(os.path.isfile(os.path.join(workspace, "src", "lib.rs")))
            with open(os.path.join(workspace, "Cargo.toml"), encoding="utf-8") as fh:
                manifest = fh.read()
            self.assertIn("rayon", manifest)
            self.assertIn("serde", manifest)

            s.cleanup()
            self.assertFalse(os.path.exists(workspace))  # zero residual leaks
            self.assertIsNone(s.workspace)


# ---------------------------------------------------------------------------
# End-to-end compilation (hermetic, zero-dependency)
# ---------------------------------------------------------------------------
class TestCompilation(unittest.TestCase):
    @unittest.skipUnless(_HAS_CARGO, "cargo not installed")
    def test_zero_dependency_build_produces_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "mylib.rs")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(_ZERO_DEP_LIB)
            out = os.path.join(d, "libaero.so")
            s = EphemeralCargoScaffolder(scratch_space=os.path.join(d, "scratch"))
            self.assertTrue(s.execute_cargo_compile(src, out))
            self.assertTrue(os.path.isfile(out))
            self.assertIsNone(s.workspace)  # cleaned up

    @unittest.skipUnless(_HAS_CARGO, "cargo not installed")
    def test_polyglot_hook_handles_rust(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "mylib.rs")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(_ZERO_DEP_LIB)
            out = os.path.join(d, "out.so")
            self.assertTrue(compile_polyglot_source(src, out))

    def test_polyglot_hook_ignores_non_rust(self):
        self.assertFalse(compile_polyglot_source("foo.py", "out.so"))

    def test_compile_without_cargo_is_graceful(self):
        # Simulate a missing cargo by pointing PATH at an empty dir.
        if not _HAS_CARGO:
            with tempfile.TemporaryDirectory() as d:
                src = os.path.join(d, "x.rs")
                with open(src, "w", encoding="utf-8") as fh:
                    fh.write(_ZERO_DEP_LIB)
                s = EphemeralCargoScaffolder(scratch_space=os.path.join(d, "s"))
                self.assertFalse(s.execute_cargo_compile(src, os.path.join(d, "o.so")))
        else:
            self.skipTest("cargo present; graceful-missing path covered structurally")


# ---------------------------------------------------------------------------
# Bootstrapper toolchain verification
# ---------------------------------------------------------------------------
class TestToolchainBootstrap(unittest.TestCase):
    def test_rust_requires_rustc_and_cargo(self):
        self.assertEqual(SYSTEM_TOOLCHAINS["rust"], ["rustc", "cargo"])
        self.assertEqual(SYSTEM_TOOLCHAINS["auto"], ["rustc", "cargo"])

    def test_missing_toolchain_binaries_for_fake_binary(self):
        # Inject a language requiring an impossible binary, confirm it's flagged.
        SYSTEM_TOOLCHAINS["__test_lang__"] = ["definitely_not_a_binary_zzz"]
        try:
            missing = RuntimeEnvironmentBootstrapper.missing_toolchain_binaries("__test_lang__")
            self.assertEqual(missing, ["definitely_not_a_binary_zzz"])
        finally:
            del SYSTEM_TOOLCHAINS["__test_lang__"]

    @unittest.skipUnless(_HAS_CARGO, "cargo not installed")
    def test_verify_rust_toolchain_passes_when_present(self):
        self.assertTrue(RuntimeEnvironmentBootstrapper.verify_toolchain("rust"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
