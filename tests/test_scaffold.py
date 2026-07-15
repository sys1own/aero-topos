# -*- coding: utf-8 -*-
"""Tests for the out-of-tree standalone repository generator (``src.scaffold``).

Layers:
* ``TestSourceResolver``  -- resolve source_entry from anywhere; single-file copy.
* ``TestRustShield``      -- extension-trait injection, type alignment, mutability.
* ``TestWorkspace``       -- out-of-tree isolation + tool-tree guard.
* ``TestRepoGenerator``   -- the turn-key Cargo project files.
* ``TestRecovery``        -- build + auto-correct + re-dispatch loop.
* ``TestEngine``          -- end-to-end scaffolding, tool tree stays clean.
* ``TestContextSingleFile`` -- ContextIngestor handles a single file anywhere.
* ``TestScaffoldCli``     -- the ``main.py`` ``scaffold`` subcommand.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main
from src.scaffold import (
    DiagnosticRecoveryRunner,
    OutOfTreeWorkspace,
    RustSemanticShield,
    ScaffoldEngine,
    SourceEntryNotFound,
    WorkspaceLocationError,
    build_spec,
    generate_repo,
    infer_dependencies,
    resolve_source_entry,
)
from src.scaffold.repo_generator import detect_pymodule
from src.scaffold.source_resolver import copy_into_workspace, infer_language
from src.scaffold.workspace import TOOL_ROOT, assert_out_of_tree

RUG_PYO3_SOURCE = """\
//! Anyon simulator core
use pyo3::prelude::*;
use rug::{Float, Complex};

#[pymodule]
fn anyon_sim(m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }

fn sector_dim(sec: u8) -> usize {
    let q_dim = match sec { 0 => 2, 1 => 3, _ => 5 };
    q_dim
}
"""


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)
        self.addCleanup(self._t.cleanup)


# ---------------------------------------------------------------------------
# Source resolver
# ---------------------------------------------------------------------------


class TestSourceResolver(_Tmp):
    def test_resolves_absolute_path(self):
        f = self.tmp / "lib.rs"
        f.write_text("fn x() {}")
        entry = resolve_source_entry(str(f))
        self.assertEqual(entry.path, f.resolve())
        self.assertEqual(entry.language, "rust")

    def test_resolves_relative_to_base_dir(self):
        (self.tmp / "data").mkdir()
        f = self.tmp / "data" / "core.rs"
        f.write_text("fn x() {}")
        entry = resolve_source_entry("data/core.rs", base_dir=self.tmp)
        self.assertEqual(entry.path, f.resolve())

    def test_missing_raises_with_helpful_message(self):
        with self.assertRaises(SourceEntryNotFound) as ctx:
            resolve_source_entry("/no/such/file_xyz.rs")
        self.assertIn("not found", str(ctx.exception))

    def test_infer_language(self):
        self.assertEqual(infer_language(Path("a.rs")), "rust")
        self.assertEqual(infer_language(Path("a.py")), "python")
        self.assertEqual(infer_language(Path("a.weird")), "unknown")

    def test_copy_into_workspace_with_transformed_content(self):
        f = self.tmp / "lib.rs"
        f.write_text("original")
        entry = resolve_source_entry(str(f))
        dest = self.tmp / "out" / "src" / "lib.rs"
        copy_into_workspace(entry, dest, content="shielded")
        self.assertEqual(dest.read_text(), "shielded")
        self.assertEqual(f.read_text(), "original")  # source untouched


# ---------------------------------------------------------------------------
# Rust shield
# ---------------------------------------------------------------------------


class TestRustShield(unittest.TestCase):
    def setUp(self):
        self.shield = RustSemanticShield()

    def test_detect_anchors(self):
        self.assertEqual(self.shield.detect_anchors(RUG_PYO3_SOURCE), {"rug", "pyo3"})
        self.assertEqual(self.shield.detect_anchors("fn main() {}"), set())

    def test_injects_traits_after_crate_attributes(self):
        src = "#![allow(dead_code)]\n//! docs\nuse rug::Float;\n"
        out, injected = self.shield.inject_extension_traits(src)
        self.assertTrue(injected)
        lines = out.splitlines()
        self.assertEqual(lines[0], "#![allow(dead_code)]")  # attr stays first
        self.assertIn("AeroNegMutExt", out)
        self.assertLess(out.index("AeroNegMutExt"), out.index("use rug::Float"))

    def test_injection_is_idempotent(self):
        once, _ = self.shield.inject_extension_traits("use rug::Float;\n")
        twice, injected = self.shield.inject_extension_traits(once)
        self.assertFalse(injected)
        self.assertEqual(once, twice)

    def test_align_match_types_only_integer_arms(self):
        src = "    let q_dim = match sec { 0 => 2, _ => 5 };\n    let name = match x { 0 => \"a\", _ => \"b\" };\n"
        out, count = self.shield.align_match_types(src)
        self.assertEqual(count, 1)
        self.assertIn("let q_dim: usize = match", out)
        self.assertIn("let name = match", out)  # string arms untouched

    def test_align_skips_already_annotated(self):
        src = "    let q_dim: u8 = match sec { 0 => 2, _ => 5 };\n"
        out, count = self.shield.align_match_types(src)
        self.assertEqual(count, 0)
        self.assertEqual(out, src)

    def test_apply_full_report(self):
        report = self.shield.apply(RUG_PYO3_SOURCE)
        self.assertEqual(report.anchors, {"rug", "pyo3"})
        self.assertTrue(any("extension-traits" in a for a in report.applied))
        self.assertTrue(any("type-cascade" in a for a in report.applied))

    def test_apply_noop_without_anchors(self):
        report = self.shield.apply("fn main() { let x = 1; }")
        self.assertFalse(report.changed)

    def test_named_shim_rug_v1_30_patch(self):
        src = "use rug::Float;\nfn f() {}\n"
        report = self.shield.apply(src, compatibility_shims=["rug_v1_30_patch"])
        self.assertIn("AeroNegMutExt", report.source)
        self.assertTrue(any("extension-traits" in a for a in report.applied))

    def test_pymodule_glob_expansion(self):
        src = (
            "use pyo3::prelude::*;\n"
            "struct Alpha;\n"
            "fn beta() {}\n"
            "#[pymodule]\n"
            "mod ext {\n"
            "    use super::*;\n"
            "}\n"
        )
        report = self.shield.apply(src)
        self.assertIn("use super::{Alpha, beta};", report.source)
        self.assertNotIn("use super::*;", report.source)
        self.assertTrue(any("pymodule-glob" in a for a in report.applied))

    def test_named_shim_pyo3_usize_alignment(self):
        src = "    let q_dim = match sec { 0 => 2, _ => 5 };\n"
        report = self.shield.apply(src, compatibility_shims=["pyo3_usize_alignment"])
        self.assertIn("let q_dim: usize = match", report.source)
        self.assertTrue(report.applied)

    def test_named_shims_empty_list_is_noop(self):
        report = self.shield.apply(RUG_PYO3_SOURCE, compatibility_shims=[])
        self.assertEqual(report.source, RUG_PYO3_SOURCE)
        self.assertFalse(report.applied)

    def test_fix_mutability(self):
        diag = "error[E0596]: cannot borrow `acc` as mutable, as it is not declared as mutable"
        out, applied = self.shield.fix_mutability("    let acc = Float::new(53);\n", diag)
        self.assertIn("let mut acc", out)
        self.assertEqual(applied, ["mut(acc)"])

    def test_correct_from_diagnostics_combines(self):
        diag = "error[E0596]: cannot borrow `z` as mutable\nerror[E0308]: mismatched types, expected `usize`"
        src = "    let z = Float::new(1);\n    let d = match s { 0 => 1, _ => 2 };\n"
        out, applied = self.shield.correct_from_diagnostics(src, diag)
        self.assertIn("let mut z", out)
        self.assertIn("let d: usize = match", out)
        self.assertTrue(any(a.startswith("mut") for a in applied))


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------


class TestWorkspace(_Tmp):
    def test_temporary_workspace_is_cleaned(self):
        with OutOfTreeWorkspace() as ws:
            root = ws.root
            self.assertTrue(root.exists())
            self.assertTrue(ws.is_temporary)
        self.assertFalse(root.exists())

    def test_distribution_directory_is_kept(self):
        dist = self.tmp / "out"
        with OutOfTreeWorkspace(distribution_directory=dist) as ws:
            (ws.root / "marker").write_text("staged")
            ws.commit()
            self.assertEqual(ws.root, dist.resolve())
        self.assertTrue(dist.exists())  # preserved -- it's the deliverable
        self.assertEqual((dist / "marker").read_text(), "staged")

    def test_refuses_inside_tool_tree(self):
        inside = TOOL_ROOT / "build_artifacts" / "x"
        with self.assertRaises(WorkspaceLocationError):
            OutOfTreeWorkspace(distribution_directory=inside).create()

    def test_assert_out_of_tree_guard(self):
        with self.assertRaises(WorkspaceLocationError):
            assert_out_of_tree(TOOL_ROOT / "src")
        assert_out_of_tree(self.tmp)  # outside -> ok


# ---------------------------------------------------------------------------
# Repo generator
# ---------------------------------------------------------------------------


class TestRepoGenerator(_Tmp):
    def test_infer_dependencies(self):
        deps = infer_dependencies(RUG_PYO3_SOURCE)
        self.assertIn("rug", deps)
        self.assertIn("pyo3", deps)
        self.assertEqual(
            deps["pyo3"]["features"],
            ["extension-module", "experimental-declarative-modules"],
        )

    def test_infer_dependencies_override(self):
        deps = infer_dependencies(RUG_PYO3_SOURCE, overrides={"rug": "0.22"})
        self.assertEqual(deps["rug"], "0.22")

    def test_detect_pymodule(self):
        self.assertEqual(detect_pymodule(RUG_PYO3_SOURCE), "anyon_sim")
        self.assertIsNone(detect_pymodule("fn main() {}"))

    def test_build_spec_pyo3_is_cdylib(self):
        spec = build_spec("anyon", RUG_PYO3_SOURCE)
        self.assertEqual(spec.crate_type, ["cdylib"])
        self.assertEqual(spec.python_module, "anyon_sim")

    def test_build_spec_plain_rug_is_rlib(self):
        spec = build_spec("calc", "use rug::Float;\nfn f() {}")
        self.assertEqual(spec.crate_type, ["rlib"])

    def test_generate_repo_writes_all_files(self):
        spec = build_spec("anyon", RUG_PYO3_SOURCE)
        repo = generate_repo(spec, self.tmp / "repo")
        names = set(repo.files)
        self.assertEqual(names, {"Cargo.toml", "src/lib.rs", ".gitignore", "README.md", "test_binding.py"})
        root = self.tmp / "repo"
        self.assertIn("crate-type = [\"cdylib\"]", (root / "Cargo.toml").read_text())
        self.assertIn(
            'features = ["extension-module", "experimental-declarative-modules"]',
            (root / "Cargo.toml").read_text(),
        )
        self.assertIn("/target/", (root / ".gitignore").read_text())
        self.assertIn("build_artifacts/", (root / ".gitignore").read_text())
        self.assertIn("anyon_sim", (root / "test_binding.py").read_text())
        # generate_repo writes the spec's source verbatim (shielding is the
        # engine's job, exercised in TestEngine).
        self.assertIn("sector_dim", (root / "src" / "lib.rs").read_text())

    def test_manifest_header_is_repo_appropriate(self):
        spec = build_spec("anyon", RUG_PYO3_SOURCE)
        generate_repo(spec, self.tmp / "repo")
        manifest = (self.tmp / "repo" / "Cargo.toml").read_text()
        self.assertNotIn("Commit a Cargo.toml to take full control", manifest)
        self.assertIn("standalone crate generated by Aero Universal", manifest)


# ---------------------------------------------------------------------------
# Diagnostic recovery loop
# ---------------------------------------------------------------------------


class TestRecovery(_Tmp):
    def _crate(self, lib_source: str) -> Path:
        root = self.tmp / "crate"
        (root / "src").mkdir(parents=True)
        (root / "src" / "lib.rs").write_text(lib_source)
        return root

    def test_is_recoverable(self):
        runner = DiagnosticRecoveryRunner()
        self.assertTrue(runner.is_recoverable("error[E0596]: cannot borrow `x` as mutable", 101))
        self.assertFalse(runner.is_recoverable("undefined reference to foo", 1))
        self.assertFalse(runner.is_recoverable("cannot borrow `x` as mutable", 0))

    def test_succeeds_first_try(self):
        root = self._crate("fn x() {}")
        runner = DiagnosticRecoveryRunner()
        result = runner.run(root, lambda: (True, "ok", 0))
        self.assertTrue(result.succeeded)
        self.assertFalse(result.recovered)
        self.assertEqual(len(result.attempts), 1)

    def test_recovers_after_correction(self):
        root = self._crate("fn f() { let acc = 1; modify(&mut acc); }\n")
        calls = {"n": 0}

        def build():
            calls["n"] += 1
            if calls["n"] == 1:
                return False, "error[E0596]: cannot borrow `acc` as mutable", 101
            return True, "ok", 0

        runner = DiagnosticRecoveryRunner()
        result = runner.run(root, build)
        self.assertTrue(result.succeeded)
        self.assertTrue(result.recovered)
        self.assertEqual(calls["n"], 2)
        # The correction was written to the crate's lib.rs.
        self.assertIn("let mut acc", (root / "src" / "lib.rs").read_text())

    def test_gives_up_on_unrecoverable(self):
        root = self._crate("fn x() {}")
        runner = DiagnosticRecoveryRunner()
        result = runner.run(root, lambda: (False, "linker error: undefined symbol", 1))
        self.assertFalse(result.succeeded)
        self.assertEqual(len(result.attempts), 1)  # no retry

    def test_bounded_retries(self):
        root = self._crate("fn f() { let acc = 1; }\n")
        # Always fails recoverably, but the patch only applies once (idempotent),
        # so the loop terminates.
        result = DiagnosticRecoveryRunner(max_retries=1).run(
            root, lambda: (False, "cannot borrow `acc` as mutable", 101)
        )
        self.assertFalse(result.succeeded)
        self.assertLessEqual(len(result.attempts), 2)


# ---------------------------------------------------------------------------
# Engine (end-to-end, no build)
# ---------------------------------------------------------------------------


class TestEngine(_Tmp):
    def test_scaffold_end_to_end_out_of_tree(self):
        src = self.tmp / "external" / "lib.rs"
        src.parent.mkdir()
        src.write_text(RUG_PYO3_SOURCE)
        dist = self.tmp / "dist" / "anyon"

        msgs = []
        engine = ScaffoldEngine(logger=msgs.append, verbose=True)
        result = engine.scaffold(str(src), name="anyon", distribution_directory=dist)

        self.assertTrue(result.out_of_tree)
        self.assertEqual(Path(result.workspace), dist.resolve())
        self.assertTrue((dist / "Cargo.toml").exists())
        self.assertTrue((dist / "src" / "lib.rs").exists())
        # The shielded source landed in the repo.
        self.assertIn("AeroNegMutExt", (dist / "src" / "lib.rs").read_text())
        self.assertTrue(result.shield["applied"])
        # Workspace is genuinely outside the tool tree.
        self.assertNotIn(str(TOOL_ROOT), str(dist.resolve()))
        self.assertTrue(msgs)  # verbose logging happened

    def test_non_rust_source_is_not_shielded(self):
        src = self.tmp / "script.py"
        src.write_text("print('hi')\n")
        dist = self.tmp / "pyrepo"
        result = ScaffoldEngine().scaffold(str(src), distribution_directory=dist)
        self.assertEqual(result.shield["applied"], [])

    def test_temp_workspace_when_no_distribution_dir(self):
        src = self.tmp / "lib.rs"
        src.write_text("use rug::Float;\nfn f() {}\n")
        result = ScaffoldEngine().scaffold(str(src), name="c", keep=True)
        # Lands in the system temp dir, not the tool tree.
        self.assertNotIn(str(TOOL_ROOT), result.workspace)
        self.assertTrue((Path(result.workspace) / "Cargo.toml").exists())


# ---------------------------------------------------------------------------
# ContextIngestor single-file support (requirement 1)
# ---------------------------------------------------------------------------


class TestContextSingleFile(_Tmp):
    def test_single_file_from_anywhere_is_ingested(self):
        from src.context.ingest import ContextIngestor

        external = self.tmp / "anywhere" / "lib.rs"
        external.parent.mkdir()
        external.write_text("fn x() {}")
        workspace = self.tmp / "ws"
        config = {
            "context": {
                "sources": [
                    {"path": str(external), "language": "rust", "target_mapping": "src/lib.rs"}
                ]
            }
        }
        report = ContextIngestor(config, workspace).ingest_all(write_report=False)
        self.assertEqual(report["files_ingested"], 1)
        self.assertEqual(report["sources"][0]["error"], None)
        self.assertTrue((workspace / "src" / "lib.rs").exists())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestScaffoldCli(_Tmp):
    def test_parser_wires_scaffold(self):
        args = main.create_parser().parse_args(["scaffold", "--source-entry", "/x/lib.rs"])
        self.assertIs(args.handler, main.scaffold_command)
        self.assertEqual(args.source_entry, "/x/lib.rs")

    def test_scaffold_command_end_to_end(self):
        src = self.tmp / "lib.rs"
        src.write_text(RUG_PYO3_SOURCE)
        dist = self.tmp / "out"
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main.main(["scaffold", "--source-entry", str(src), "--name", "anyon",
                            "--distribution-directory", str(dist)])
        self.assertEqual(rc, 0)
        self.assertIn("Standalone repository generated", out.getvalue())
        self.assertTrue((dist / "Cargo.toml").exists())

    def test_scaffold_command_missing_source(self):
        import sys
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            rc = main.main(["scaffold", "--source-entry", "/no/such/path_zzz.rs"])
        finally:
            sys.stderr = old
        self.assertEqual(rc, 1)
        self.assertIn("not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
