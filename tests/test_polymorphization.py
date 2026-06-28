"""Tests for the Autonomous Hardware-Polymerization subsystem.

Organised by layer:

* ``TestTopology``            -- the HardwareTopology data model + derived helpers.
* ``TestHardwareProfiler``    -- dependency-free host probing + graceful fallbacks.
* ``TestPolymorphicRewriter`` -- in-memory rewriting of C/C++, Rust and LLVM IR.
* ``TestRewriteTree``         -- ephemeral-cache rewriting leaves source untouched.
* ``TestPolymorphizationEngine`` -- end-to-end profile + rewrite + report.
* ``TestPolymorphizeCli``     -- the ``main.py`` ``polymorphize`` subcommand.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main
from src.polymorphization import (
    CacheLevel,
    GpuDevice,
    HardwareProfiler,
    HardwareTopology,
    PolymorphicRewriter,
    PolymorphizationEngine,
)


def _avx2_host() -> HardwareTopology:
    return HardwareTopology(
        arch="x86_64",
        physical_cores=8,
        logical_cores=16,
        cpu_features=["sse2", "avx", "avx2"],
        cache_levels=[CacheLevel(1, 32 * 1024, 64), CacheLevel(2, 256 * 1024, 64)],
    )


def _neon_host() -> HardwareTopology:
    return HardwareTopology(
        arch="aarch64",
        physical_cores=4,
        logical_cores=4,
        cpu_features=["neon"],
        cache_levels=[CacheLevel(1, 64 * 1024, 128)],
    )


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


class TestTopology(unittest.TestCase):
    def test_best_simd_prefers_strongest_isa(self):
        topo = HardwareTopology(cpu_features=["sse2", "avx2", "avx512"])
        self.assertEqual(topo.best_simd(), "avx512")

    def test_best_simd_falls_back_to_scalar(self):
        self.assertEqual(HardwareTopology(cpu_features=[]).best_simd(), "scalar")

    def test_vector_width_matches_isa(self):
        self.assertEqual(HardwareTopology(cpu_features=["avx2"]).vector_width_bytes(), 32)
        self.assertEqual(HardwareTopology(cpu_features=["neon"]).vector_width_bytes(), 16)
        self.assertEqual(HardwareTopology(cpu_features=[]).vector_width_bytes(), 8)

    def test_cache_line_prefers_l1_then_defaults_to_64(self):
        topo = HardwareTopology(cache_levels=[CacheLevel(2, 0, 256), CacheLevel(1, 0, 128)])
        self.assertEqual(topo.cache_line_bytes(), 128)
        self.assertEqual(HardwareTopology().cache_line_bytes(), 64)

    def test_alignment_is_max_of_cache_line_and_vector_width(self):
        # AVX-512 (64B vectors) on a 64B cache line -> 64.
        topo = HardwareTopology(cpu_features=["avx512"], cache_levels=[CacheLevel(1, 0, 64)])
        self.assertEqual(topo.alignment_bytes(), 64)
        # NEON (16B) on a 128B cache line -> 128.
        self.assertEqual(_neon_host().alignment_bytes(), 128)

    def test_has_gpu(self):
        self.assertFalse(HardwareTopology().has_gpu())
        self.assertTrue(HardwareTopology(gpus=[GpuDevice("cuda", "A100", "sm_80")]).has_gpu())

    def test_round_trip_serialisation(self):
        topo = _avx2_host()
        topo.gpus = [GpuDevice("cuda", "RTX 4090", "sm_89")]
        restored = HardwareTopology.from_dict(json.loads(json.dumps(topo.to_dict())))
        self.assertEqual(restored.arch, topo.arch)
        self.assertEqual(restored.physical_cores, topo.physical_cores)
        self.assertEqual(restored.cpu_features, topo.cpu_features)
        self.assertEqual(restored.gpus[0].architecture, "sm_89")
        self.assertEqual(restored.cache_levels[0].line_size_bytes, 64)

    def test_to_dict_exposes_derived_block(self):
        derived = _avx2_host().to_dict()["derived"]
        self.assertEqual(derived["best_simd"], "avx2")
        self.assertEqual(derived["alignment_bytes"], 64)


# ---------------------------------------------------------------------------
# Hardware profiler
# ---------------------------------------------------------------------------


class TestHardwareProfiler(unittest.TestCase):
    def test_probe_returns_sane_topology(self):
        topo = HardwareProfiler(allow_subprocess=False).probe()
        self.assertGreaterEqual(topo.logical_cores, 1)
        self.assertGreaterEqual(topo.physical_cores, 1)
        self.assertLessEqual(topo.physical_cores, topo.logical_cores)
        self.assertTrue(topo.cache_levels)  # real sysfs or defaults
        self.assertIn(topo.memory_bandwidth_class, ("low", "medium", "high", "unknown"))

    def test_no_subprocess_means_no_gpu_probing(self):
        # With subprocess disabled, GPU detection cannot shell out.
        topo = HardwareProfiler(allow_subprocess=False).probe()
        self.assertEqual(topo.gpus, [])

    def test_parse_size_handles_units(self):
        self.assertEqual(HardwareProfiler._parse_size("32K"), 32 * 1024)
        self.assertEqual(HardwareProfiler._parse_size("8M"), 8 * 1024 * 1024)
        self.assertEqual(HardwareProfiler._parse_size("1G"), 1024**3)
        self.assertEqual(HardwareProfiler._parse_size(None), 0)

    def test_default_cache_levels_cover_l1_l2_l3(self):
        levels = {c.level for c in HardwareProfiler._default_cache_levels()}
        self.assertEqual(levels, {1, 2, 3})

    def test_feature_normalisation_maps_asimd_to_neon(self):
        # _FEATURE_ALIASES is the contract the topology relies on.
        from src.polymorphization.hardware_profiler import _FEATURE_ALIASES

        self.assertEqual(_FEATURE_ALIASES["asimd"], "neon")
        self.assertEqual(_FEATURE_ALIASES["avx512f"], "avx512")


# ---------------------------------------------------------------------------
# Rewriter (in-memory)
# ---------------------------------------------------------------------------


class TestPolymorphicRewriter(unittest.TestCase):
    def test_cpp_alignment_and_workers_and_kernel(self):
        rw = PolymorphicRewriter(_avx2_host())
        src = "alignas(AERO_ALIGN) float b[8];\nint w = AERO_WORKERS;\nx = AERO_KERNEL(saxpy)(a);\n"
        result = rw.rewrite_text(src, "cpp")
        self.assertIn("alignas(64)", result.text)
        self.assertIn("int w = 8;", result.text)  # physical cores, not logical (16)
        self.assertIn("saxpy__avx2", result.text)
        self.assertEqual(result.changes["alignment"], 1)
        self.assertEqual(result.changes["workers"], 1)
        self.assertEqual(result.changes["kernels"], 1)

    def test_cpp_pragma_emits_openmp_simd(self):
        rw = PolymorphicRewriter(_avx2_host())
        result = rw.rewrite_text("    // AERO_PRAGMA_SIMD\n    for (...)\n", "cpp")
        self.assertIn("#pragma omp simd simdlen(8)", result.text)  # 32B / 4 = 8 f32 lanes
        self.assertIn("avx2", result.text)
        self.assertTrue(result.text.startswith("    #pragma"))  # indentation preserved

    def test_rust_uses_target_feature_and_repr_align(self):
        rw = PolymorphicRewriter(_neon_host())
        src = "#[repr(align(AERO_ALIGN))]\n// AERO_PRAGMA_SIMD\nfn f() { g(AERO_KERNEL(blur)); }\n"
        result = rw.rewrite_text(src, "rust")
        self.assertIn("#[repr(align(128))]", result.text)
        self.assertIn('#[target_feature(enable = "neon")]', result.text)
        self.assertIn("blur__neon", result.text)

    def test_llvm_ir_alignment_and_vectorize_comment(self):
        rw = PolymorphicRewriter(_avx2_host())
        src = "%p = alloca i8, align AERO_ALIGN\n; AERO_PRAGMA_SIMD\n"
        result = rw.rewrite_text(src, "llvm")
        self.assertIn("align 64", result.text)
        self.assertIn("aero: vectorize width 8", result.text)

    def test_scalar_host_emits_no_simd_directive(self):
        scalar = HardwareTopology(arch="x86_64", physical_cores=2, logical_cores=2, cpu_features=[])
        rw = PolymorphicRewriter(scalar)
        result = rw.rewrite_text("// AERO_PRAGMA_SIMD\nz = AERO_KERNEL(dot)(a);\n", "cpp")
        self.assertIn("scalar path", result.text)
        self.assertIn("dot__scalar", result.text)

    def test_token_word_boundaries(self):
        # AERO_ALIGN must not match inside AERO_ALIGNMENT.
        rw = PolymorphicRewriter(_avx2_host())
        result = rw.rewrite_text("int AERO_ALIGNMENT = AERO_ALIGN;\n", "cpp")
        self.assertIn("AERO_ALIGNMENT", result.text)
        self.assertIn("= 64;", result.text)

    def test_no_markers_means_no_changes(self):
        rw = PolymorphicRewriter(_avx2_host())
        result = rw.rewrite_text("int main() { return 0; }\n", "cpp")
        self.assertFalse(result.changed)


# ---------------------------------------------------------------------------
# Rewriter (ephemeral cache / tree)
# ---------------------------------------------------------------------------


class TestRewriteTree(unittest.TestCase):
    def test_tree_rewrite_leaves_source_untouched(self):
        rw = PolymorphicRewriter(_avx2_host())
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen"
            src.mkdir()
            original = "alignas(AERO_ALIGN) int x;\n"
            (src / "k.cpp").write_text(original)
            (src / "notes.md").write_text("ignore me")
            cache = Path(tmp) / "cache"

            report = rw.rewrite_tree(src, cache)

            # Source is read-only.
            self.assertEqual((src / "k.cpp").read_text(), original)
            # Cache holds the rewritten copy, structure preserved.
            self.assertEqual((cache / "k.cpp").read_text(), "alignas(64) int x;\n")
            # Unsupported files are not copied.
            self.assertFalse((cache / "notes.md").exists())
            self.assertEqual(report["files_processed"], 1)
            self.assertEqual(report["files_rewritten"], 1)

    def test_tree_preserves_nested_layout(self):
        rw = PolymorphicRewriter(_avx2_host())
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen"
            (src / "sub").mkdir(parents=True)
            (src / "sub" / "a.rs").write_text("let n = AERO_WORKERS;\n")
            cache = Path(tmp) / "cache"
            rw.rewrite_tree(src, cache)
            self.assertEqual((cache / "sub" / "a.rs").read_text(), "let n = 8;\n")

    def test_reset_cache_removes_stale_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "stale.cpp").write_text("old")
            PolymorphicRewriter.reset_cache(cache)
            self.assertFalse(cache.exists())


# ---------------------------------------------------------------------------
# Engine (end-to-end)
# ---------------------------------------------------------------------------


class TestPolymorphizationEngine(unittest.TestCase):
    def test_polymerize_tree_resets_cache_each_run(self):
        engine = PolymorphizationEngine()
        topo = _avx2_host()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen"
            src.mkdir()
            (src / "k.cpp").write_text("alignas(AERO_ALIGN) int x;\n")
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "leftover.cpp").write_text("stale")

            report = engine.polymerize_tree(src, cache, topology=topo)

            # Stale artifacts are gone; only the freshly rewritten file remains.
            self.assertFalse((cache / "leftover.cpp").exists())
            self.assertTrue((cache / "k.cpp").exists())
            self.assertEqual(report["topology"]["derived"]["best_simd"], "avx2")

    def test_write_report_persists_topology_and_report(self):
        engine = PolymorphizationEngine()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen"
            src.mkdir()
            (src / "k.cpp").write_text("int w = AERO_WORKERS;\n")
            cache = Path(tmp) / "cache"
            report = engine.polymerize_tree(src, cache, topology=_avx2_host())
            report_path = engine.write_report(report, cache)

            self.assertTrue(report_path.exists())
            self.assertTrue((cache / "hardware_topology.json").exists())
            on_disk = json.loads(report_path.read_text())
            self.assertIn("topology", on_disk)
            self.assertIn("rewrite", on_disk)

    def test_polymerize_text_round_trips_through_profile(self):
        engine = PolymorphizationEngine()
        out = engine.polymerize_text("int w = AERO_WORKERS;\n", "cpp", topology=_neon_host())
        self.assertEqual(out, "int w = 4;\n")

    def test_profile_host_uses_injected_profiler(self):
        engine = PolymorphizationEngine(profiler=HardwareProfiler(allow_subprocess=False))
        topo = engine.profile_host()
        self.assertIsInstance(topo, HardwareTopology)
        self.assertIs(engine.last_topology, topo)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestPolymorphizeCli(unittest.TestCase):
    def test_parser_wires_polymorphize_subcommand(self):
        args = main.create_parser().parse_args(["polymorphize", "--source-dir", "gen"])
        self.assertIs(args.handler, main.polymorphize_command)
        self.assertEqual(args.source_dir, "gen")
        self.assertFalse(args.profile_only)

    def test_build_parser_has_no_polymorph_flag(self):
        args = main.create_parser().parse_args(["build", "--no-polymorph"])
        self.assertTrue(args.no_polymorph)

    def test_profile_only_prints_topology(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main.main(["polymorphize", "--profile-only"])
        self.assertEqual(rc, 0)
        self.assertIn("Hardware Topology:", out.getvalue())

    def test_polymorphize_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen"
            src.mkdir()
            (src / "k.cpp").write_text("alignas(AERO_ALIGN) int x;\n")
            cache = Path(tmp) / "cache"
            out = io.StringIO()
            with redirect_stdout(out):
                rc = main.main(
                    ["polymorphize", "--source-dir", str(src), "--cache-dir", str(cache)]
                )
            self.assertEqual(rc, 0)
            self.assertIn("Autonomous Hardware-Polymerization:", out.getvalue())
            self.assertTrue((cache / "k.cpp").exists())
            # Source directory is never modified.
            self.assertEqual((src / "k.cpp").read_text(), "alignas(AERO_ALIGN) int x;\n")

    def test_polymorphize_reports_missing_source_dir(self):
        import sys
        err = io.StringIO()
        with redirect_stdout(io.StringIO()):
            old = sys.stderr
            sys.stderr = err
            try:
                rc = main.main(["polymorphize", "--source-dir", "/nonexistent/missing-12345"])
            finally:
                sys.stderr = old
        self.assertEqual(rc, 1)
        self.assertIn("directory not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
