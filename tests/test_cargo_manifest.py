# -*- coding: utf-8 -*-
"""Tests for Rust/Cargo manifest handling (``src.build.cargo_manifest``) and the
manifest-aware :class:`RustCompiler`.

Covers:
* crate-root resolution (sources, ``manifest_path``, ``root``, subdir crates);
* respecting an existing ``Cargo.toml`` verbatim (no synthesis, no overwrite);
* synthesising a manifest with blueprint-pinned dependency versions;
* the TOML emitter (bare versions + inline-table specs);
* ``RustCompiler`` building the right cargo command + artefact directory;
* the block-DSL schema / ``TargetNode`` carrying ``manifest_path`` / ``root``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.build.cargo_manifest import (
    CargoPlan,
    extract_cargo_options,
    find_existing_manifest,
    prepare_crate,
    read_manifest_package_name,
    render_manifest,
    resolve_crate_root,
    sanitize_crate_name,
)
from src.build.compilers import RustCompiler, compile_target


class _TmpMixin(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, rel: str, content: str = "") -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Crate-root resolution
# ---------------------------------------------------------------------------


class TestResolveCrateRoot(_TmpMixin):
    def test_existing_manifest_above_sources_wins(self):
        self.write("crates/foo/Cargo.toml", '[package]\nname="foo"\nversion="0.1.0"\n')
        self.write("crates/foo/src/lib.rs", "pub fn f() {}")
        crate_root = resolve_crate_root(self.root, sources=["crates/foo/src/lib.rs"])
        self.assertEqual(crate_root, (self.root / "crates" / "foo").resolve())

    def test_manifest_path_to_file_uses_its_parent(self):
        self.write("native/Cargo.toml", '[package]\nname="n"\nversion="0.1.0"\n')
        crate_root = resolve_crate_root(self.root, manifest_path="native/Cargo.toml")
        self.assertEqual(crate_root, (self.root / "native").resolve())

    def test_manifest_path_to_dir_uses_dir(self):
        (self.root / "native").mkdir()
        crate_root = resolve_crate_root(self.root, manifest_path="native")
        self.assertEqual(crate_root, (self.root / "native").resolve())

    def test_root_field_points_at_subdirectory(self):
        crate_root = resolve_crate_root(self.root, root="crates/bar")
        self.assertEqual(crate_root, (self.root / "crates" / "bar").resolve())

    def test_src_layout_resolves_to_parent(self):
        # No manifest yet; sources under src/ -> crate root is src/'s parent.
        crate_root = resolve_crate_root(self.root, sources=["src/lib.rs"])
        self.assertEqual(crate_root, self.root.resolve())

    def test_defaults_to_workspace_when_nothing_known(self):
        self.assertEqual(resolve_crate_root(self.root), self.root.resolve())


# ---------------------------------------------------------------------------
# Manifest detection + synthesis
# ---------------------------------------------------------------------------


class TestPrepareCrate(_TmpMixin):
    def test_existing_manifest_is_used_verbatim(self):
        original = '[package]\nname = "foo_existing"\nversion = "0.3.0"\nedition = "2018"\n\n[dependencies]\nrug = "0.19"\n'
        self.write("crates/foo/Cargo.toml", original)
        self.write("crates/foo/src/lib.rs", "pub fn f() {}")

        plan = prepare_crate(self.root, "foo", sources=["crates/foo/src/lib.rs"])

        self.assertTrue(plan.used_existing)
        self.assertFalse(plan.synthesized)
        self.assertEqual(plan.crate_name, "foo_existing")
        # The committed manifest (with the older pin) is left exactly as-is.
        self.assertEqual(plan.manifest_path.read_text(), original)

    def test_existing_manifest_not_overwritten_even_with_cargo_deps(self):
        original = '[package]\nname="foo"\nversion="0.1.0"\n[dependencies]\nrug = "0.19"\n'
        self.write("foo/Cargo.toml", original)
        plan = prepare_crate(
            self.root, "foo", root="foo", cargo_options={"dependencies": {"rug": "0.22"}}
        )
        self.assertTrue(plan.used_existing)
        self.assertIn('rug = "0.19"', plan.manifest_path.read_text())
        self.assertNotIn("0.22", plan.manifest_path.read_text())

    def test_synthesizes_when_absent_with_pinned_versions(self):
        self.write("src/lib.rs", "pub fn g() {}")
        plan = prepare_crate(
            self.root, "mycrate", sources=["src/lib.rs"],
            cargo_options={"dependencies": {"rug": "0.22", "serde": "1.0"}},
        )
        self.assertTrue(plan.synthesized)
        self.assertFalse(plan.used_existing)
        content = plan.manifest_path.read_text()
        self.assertIn('name = "mycrate"', content)
        self.assertIn('rug = "0.22"', content)
        self.assertIn('serde = "1.0"', content)

    def test_manifest_path_to_existing_file_is_respected(self):
        self.write("native/Cargo.toml", '[package]\nname="native"\nversion="1.0.0"\n')
        plan = prepare_crate(self.root, "whatever", manifest_path="native/Cargo.toml")
        self.assertTrue(plan.used_existing)
        self.assertEqual(plan.crate_root, (self.root / "native").resolve())

    def test_write_false_does_not_create_manifest(self):
        self.write("src/lib.rs", "pub fn g(){}")
        plan = prepare_crate(self.root, "c", sources=["src/lib.rs"], write=False)
        self.assertTrue(plan.synthesized)
        self.assertFalse(plan.manifest_path.exists())

    def test_target_and_profile_dirs(self):
        plan = prepare_crate(self.root, "c", root="crates/c")
        self.assertEqual(plan.target_dir, (self.root / "crates" / "c" / "target").resolve())
        self.assertTrue(str(plan.profile_dir(release=True)).endswith("target/release"))
        self.assertTrue(str(plan.profile_dir(release=False)).endswith("target/debug"))


# ---------------------------------------------------------------------------
# TOML emitter
# ---------------------------------------------------------------------------


class TestRenderManifest(unittest.TestCase):
    def test_bare_version_string(self):
        out = render_manifest("c", {"rug": "0.22"})
        self.assertIn('rug = "0.22"', out)
        self.assertIn('edition = "2021"', out)

    def test_inline_table_spec(self):
        out = render_manifest("c", {"serde": {"version": "1.0", "features": ["derive"], "optional": True}})
        self.assertIn('serde = { version = "1.0", features = ["derive"], optional = true }', out)

    def test_crate_type_emits_lib_section(self):
        out = render_manifest("c", {}, crate_type=["cdylib", "rlib"])
        self.assertIn("[lib]", out)
        self.assertIn('crate-type = ["cdylib", "rlib"]', out)

    def test_custom_edition_and_version(self):
        out = render_manifest("c", {}, edition="2018", version="2.5.0")
        self.assertIn('edition = "2018"', out)
        self.assertIn('version = "2.5.0"', out)

    def test_sanitize_crate_name(self):
        self.assertEqual(sanitize_crate_name("python_dashboard"), "python_dashboard")
        self.assertEqual(sanitize_crate_name("My-Crate"), "my_crate")
        self.assertEqual(sanitize_crate_name("123abc"), "crate_123abc")
        self.assertEqual(sanitize_crate_name("!!!"), "aero_crate")


class TestReadPackageName(_TmpMixin):
    def test_reads_name(self):
        m = self.write("Cargo.toml", '[package]\nname = "hello_world"\nversion = "0.1.0"\n')
        self.assertEqual(read_manifest_package_name(m), "hello_world")

    def test_missing_file_returns_none(self):
        self.assertIsNone(read_manifest_package_name(self.root / "nope.toml"))

    def test_malformed_returns_none(self):
        m = self.write("Cargo.toml", "this is not valid toml = = =")
        self.assertIsNone(read_manifest_package_name(m))


class TestExtractCargoOptions(unittest.TestCase):
    def test_nested_cargo_block(self):
        opts = extract_cargo_options({"cargo": {"dependencies": {"rug": "0.22"}, "edition": "2018"}})
        self.assertEqual(opts["dependencies"], {"rug": "0.22"})
        self.assertEqual(opts["edition"], "2018")

    def test_flat_cargo_dependencies_dict_form(self):
        opts = extract_cargo_options({"cargo_dependencies": {"rug": "0.22"}})
        self.assertEqual(opts["dependencies"], {"rug": "0.22"})

    def test_flat_cargo_dependencies_list_form(self):
        opts = extract_cargo_options({"cargo_dependencies": ["rug=0.22", 'serde = "1.0"']})
        self.assertEqual(opts["dependencies"], {"rug": "0.22", "serde": "1.0"})

    def test_nested_and_flat_merge(self):
        opts = extract_cargo_options(
            {"cargo": {"dependencies": {"rug": "0.22"}, "edition": "2018"},
             "cargo_dependencies": ["serde=1.0"]}
        )
        self.assertEqual(opts["dependencies"], {"rug": "0.22", "serde": "1.0"})
        self.assertEqual(opts["edition"], "2018")

    def test_empty(self):
        self.assertEqual(extract_cargo_options({}), {})


# ---------------------------------------------------------------------------
# RustCompiler integration
# ---------------------------------------------------------------------------


class TestRustCompilerCommand(unittest.TestCase):
    def test_build_command_includes_manifest_path(self):
        rc = RustCompiler()
        if not (rc.discover() or "").endswith("cargo"):
            self.skipTest("cargo not installed")
        cmd = rc.build_command(["src/lib.rs"], options={"manifest_path": "/x/Cargo.toml"})
        self.assertIn("--manifest-path", cmd)
        self.assertIn("/x/Cargo.toml", cmd)

    def test_build_command_release_flag(self):
        rc = RustCompiler()
        if not (rc.discover() or "").endswith("cargo"):
            self.skipTest("cargo not installed")
        cmd = rc.build_command(["src/lib.rs"], flags=["--release"])
        self.assertIn("--release", cmd)
        # --release is hoisted to a cargo flag, not duplicated after a separator.
        self.assertEqual(cmd.count("--release"), 1)


class TestRustCompilerCompile(_TmpMixin):
    def test_no_toolchain_reports_error(self):
        rc = RustCompiler()
        if rc.discover() is not None:
            self.skipTest("a rust toolchain is installed; cannot test the missing-toolchain path")
        result = rc.compile("t", ["src/lib.rs"], workdir=self.root)
        self.assertFalse(result.success)
        self.assertIn("rust compiler", result.stderr)

    def test_compile_synthesizes_manifest_before_invoking_cargo(self):
        # The manifest-prep step must run and write the manifest in the resolved
        # subdirectory crate root, and cargo must run from there (artefact dir
        # points at that crate's target/).  Dependency-free so the real cargo
        # build stays offline and fast.
        rc = RustCompiler()
        if not (rc.discover() or "").endswith("cargo"):
            self.skipTest("cargo not installed")
        self.write("crates/demo/src/lib.rs", "pub fn f() {}")
        result = rc.compile("demo", ["crates/demo/src/lib.rs"], workdir=self.root)
        manifest = self.root / "crates" / "demo" / "Cargo.toml"
        self.assertTrue(manifest.exists())
        self.assertIn('name = "demo"', manifest.read_text())
        crate_root = (self.root / "crates" / "demo").resolve()
        self.assertEqual(result.details.get("crate_root"), str(crate_root))
        # Artefacts are collected from this crate's own target/ directory.
        self.assertEqual(result.details.get("artifact_dir"), str(crate_root / "target" / "debug"))

    def test_compile_respects_existing_manifest(self):
        rc = RustCompiler()
        if not (rc.discover() or "").endswith("cargo"):
            self.skipTest("cargo not installed")
        # Dependency-free existing manifest (offline-safe) -- must be left as-is
        # even though the blueprint requests a different dependency pin.
        original = '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
        self.write("crates/demo/Cargo.toml", original)
        self.write("crates/demo/src/lib.rs", "pub fn f() {}")
        result = rc.compile("demo", ["crates/demo/src/lib.rs"], workdir=self.root,
                            options={"cargo": {"dependencies": {"rug": "0.22"}}})
        # Existing manifest left untouched; synthesis was skipped.
        self.assertEqual((self.root / "crates" / "demo" / "Cargo.toml").read_text(), original)
        self.assertTrue(result.details.get("used_existing"))


# ---------------------------------------------------------------------------
# Blueprint schema / TargetNode plumbing
# ---------------------------------------------------------------------------


class TestBlueprintPlumbing(unittest.TestCase):
    def test_block_dsl_accepts_manifest_path_and_root(self):
        import blueprint_lang

        source = (
            'project "p" { version = "1.0" }\n'
            'target "engine" {\n'
            '    language = "rust"\n'
            '    sources  = ["crates/engine/src/lib.rs"]\n'
            '    root     = "crates/engine"\n'
            '    manifest_path = "crates/engine/Cargo.toml"\n'
            "}\n"
        )
        self.assertIsNone(blueprint_lang.check_source(source))

    def test_target_node_carries_rust_fields_into_metadata(self):
        import blueprint_lang
        from build_graph import blueprint_to_dag

        source = (
            'project "p" { version = "1.0" }\n'
            'target "engine" {\n'
            '    language = "rust"\n'
            '    sources  = ["crates/engine/src/lib.rs"]\n'
            '    root     = "crates/engine"\n'
            '    cargo_dependencies = ["rug=0.22"]\n'
            "}\n"
        )
        bp = blueprint_lang.load_source(source)
        graph = blueprint_to_dag(bp)
        meta = graph.targets["engine"].to_dict()
        self.assertEqual(meta["root"], "crates/engine")
        self.assertEqual(meta["cargo_dependencies"], ["rug=0.22"])
        # ...and that metadata yields the pinned dependency for synthesis.
        self.assertEqual(extract_cargo_options(meta)["dependencies"], {"rug": "0.22"})


if __name__ == "__main__":
    unittest.main()
