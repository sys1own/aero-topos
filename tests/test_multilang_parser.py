"""Tests for multi-language UAST generation (C / C++ / Fortran / GPU) + FFI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis.semantic_mapper import SemanticMapper

_CONFIG = {
    "analysis": {
        "semantic_proximity_mapping": {
            "source_roots": {
                "python": "app",
                "rust": "native",
                "c": "csrc",
                "cpp": "cppsrc",
                "fortran": "fsrc",
            }
        }
    },
    "gpu": {"kernel_sources": ["kernels/*.cu"]},
}


def _build_project(root: Path) -> None:
    for d in ("app", "native", "csrc", "cppsrc", "fsrc", "kernels"):
        (root / d).mkdir()
    (root / "app" / "main.py").write_text(
        "import ctypes\n"
        "PI = 3.14159\n"
        "def run():\n"
        "    lib = ctypes.CDLL('libsim.so')\n"
        "    return compute(area(2.0))\n"
    )
    (root / "native" / "lib.rs").write_text(
        "#[pyfunction]\nfn compute(x: f64) -> f64 { x * 2.0 }\n"
        'extern "C" fn c_export(v: f64) -> f64 { v }\n'
    )
    (root / "csrc" / "core.c").write_text(
        "typedef struct { double re; } Complex;\n"
        "double GLOBAL = 3.14;\n"
        "extern int ext_fn(int);\n"
        "double compute_c(double x){ return ext_fn(1) * GLOBAL; }\n"
    )
    (root / "cppsrc" / "api.cpp").write_text(
        'extern "C" { int api_fn(int a); }\n'
        "typedef double real_t;\n"
        "class Solver { public: int run(); };\n"
        "int call_site(){ return api_fn(2); }\n"
    )
    (root / "fsrc" / "phys.f90").write_text(
        "module phys\n  real(8), parameter :: PI = 3.14159d0\ncontains\n"
        '  function area(r) bind(c, name="area")\n'
        "    real(8) :: r, area\n    area = PI*r*r\n  end function\nend module\n"
    )
    (root / "kernels" / "k.cu").write_text(
        "__global__ void vecAdd(float* a){ }\n"
        "void launch(){ vecAdd<<<1,256>>>(0); }\n"
    )


class TestMultiLanguageParsing(unittest.TestCase):
    def _mapper(self, root: Path) -> SemanticMapper:
        mapper = SemanticMapper(_CONFIG)
        mapper.build_uast(root)
        return mapper

    def test_all_languages_produce_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            stats = self._mapper(root).get_statistics()
            for key in ("python_nodes", "rust_nodes", "c_nodes", "cpp_nodes", "fortran_nodes"):
                self.assertGreater(stats[key], 0, f"no nodes for {key}")
            self.assertGreater(stats["gpu_kernel_nodes"], 0)

    def test_unified_node_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            counts = self._mapper(root).get_statistics()["unified_node_counts"]
            # Functions, calls, globals, types, gpu kernels across languages.
            self.assertGreater(counts.get("uast_function", 0), 0)
            self.assertGreater(counts.get("uast_call", 0), 0)
            self.assertGreater(counts.get("uast_global", 0), 0)
            self.assertGreater(counts.get("uast_type", 0), 0)
            self.assertGreater(counts.get("uast_gpu_kernel", 0), 0)

    def test_ffi_detection_across_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            mapper = self._mapper(root)
            ffi = mapper.get_ffi_bindings()
            types = {info["type"] for info in ffi.values()}
            self.assertIn("pyo3_function", types)
            self.assertIn("rust_extern_c", types)
            self.assertIn("c_abi", types)  # extern "C" linkage block (api_fn)
            self.assertIn("c_extern", types)  # extern declaration (ext_fn)
            self.assertIn("fortran_c_abi", types)  # bind(c) (area)
            self.assertIn("python_ctypes", types)  # CDLL load
            self.assertIn("gpu_kernel", types)

    def test_gpu_kernel_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            stats = self._mapper(root).get_statistics()
            self.assertGreaterEqual(stats["gpu_kernel_edges"], 1)

    def test_pyo3_bridge_edge_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_project(root)
            mapper = self._mapper(root)
            self.assertIn("compute", mapper.get_ffi_bindings())
            self.assertEqual(mapper.get_ffi_bindings()["compute"]["type"], "pyo3_function")
            self.assertGreaterEqual(mapper.get_statistics()["ffi_edges"], 1)

    def test_missing_grammar_is_graceful(self):
        # An empty project must not raise even if some grammars are absent.
        with tempfile.TemporaryDirectory() as tmp:
            mapper = SemanticMapper(_CONFIG)
            uast = mapper.build_uast(Path(tmp))
            self.assertEqual(uast.number_of_nodes(), 0)


if __name__ == "__main__":
    unittest.main()
