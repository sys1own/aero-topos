# -*- coding: utf-8 -*-
"""Tests for ``src.build.compilers`` -- compiler backend registry, discovery, and invocation.

Covers:
* Backend registry + factory (:func:`get_backend`)
* :func:`compile_target` one-shot helper
* Command-line construction for C, C++, Rust, Python, Node, Fortran
* Discovery (we only test that ``discover()`` returns a string or None)
* :class:`CompileResult` error_summary
* Graceful handling of unsupported languages
"""

from __future__ import annotations

import unittest

from src.build.compilers import (
    BACKENDS,
    CCompiler,
    CompileResult,
    CppCompiler,
    FortranCompiler,
    NodeRuntime,
    PythonRuntime,
    RustCompiler,
    compile_target,
    get_backend,
)


class TestBackendRegistry(unittest.TestCase):
    def test_all_supported_languages_registered(self):
        for lang in ("c", "cpp", "rust", "python", "node", "fortran"):
            self.assertIn(lang, BACKENDS)

    def test_get_backend_returns_instance(self):
        for lang in BACKENDS:
            backend = get_backend(lang)
            self.assertIsNotNone(backend)
            self.assertEqual(backend.language, lang)

    def test_get_backend_unknown_returns_none(self):
        self.assertIsNone(get_backend("brainfuck"))

    def test_get_backend_case_insensitive(self):
        self.assertIsNotNone(get_backend("Python"))
        self.assertIsNotNone(get_backend("CPP"))


class TestCCompiler(unittest.TestCase):
    def test_build_command_basic(self):
        cc = CCompiler()
        cmd = cc.build_command(["main.c"])
        self.assertIn("main.c", cmd)

    def test_build_command_with_output(self):
        cc = CCompiler()
        cmd = cc.build_command(["main.c"], output="a.out")
        self.assertIn("-o", cmd)
        self.assertIn("a.out", cmd)

    def test_build_command_with_flags_and_defines(self):
        cc = CCompiler()
        cmd = cc.build_command(["main.c"], flags=["-O2", "-Wall"], defines=["NDEBUG"])
        self.assertIn("-O2", cmd)
        self.assertIn("-Wall", cmd)
        self.assertIn("-D", cmd)
        self.assertIn("NDEBUG", cmd)


class TestCppCompiler(unittest.TestCase):
    def test_build_command_basic(self):
        cpp = CppCompiler()
        cmd = cpp.build_command(["main.cpp"])
        self.assertIn("main.cpp", cmd)

    def test_build_command_with_output(self):
        cpp = CppCompiler()
        cmd = cpp.build_command(["main.cpp"], output="prog")
        self.assertIn("-o", cmd)
        self.assertIn("prog", cmd)


class TestRustCompiler(unittest.TestCase):
    def test_build_command_cargo(self):
        rc = RustCompiler()
        binary = rc.discover()
        if binary and "cargo" in binary:
            cmd = rc.build_command(["src/lib.rs"])
            self.assertEqual(cmd[0], binary)
            self.assertIn("build", cmd)

    def test_build_command_flags(self):
        rc = RustCompiler()
        binary = rc.discover()
        if binary and "cargo" in binary:
            # --release is a cargo build flag (passed to `cargo build`, not after
            # a `--` separator, which would forward it to the built binary).
            cmd = rc.build_command(["src/lib.rs"], flags=["--release"])
            self.assertIn("--release", cmd)
            self.assertEqual(cmd[:2], [binary, "build"])

    def test_build_command_strips_edition_flag(self):
        rc = RustCompiler()
        binary = rc.discover()
        if binary and "cargo" in binary:
            cmd = rc.build_command(
                ["src/lib.rs"],
                flags=["--edition", "2021", "--features", "foo"],
            )
            self.assertNotIn("--edition", cmd)
            self.assertNotIn("2021", cmd)
            self.assertIn("--features", cmd)
            self.assertIn("foo", cmd)
            self.assertEqual(cmd[:2], [binary, "build"])

    def test_build_command_strips_edition_equals_flag(self):
        rc = RustCompiler()
        binary = rc.discover()
        if binary and "cargo" in binary:
            cmd = rc.build_command(
                ["src/lib.rs"],
                flags=["--edition=2021", "--release"],
            )
            self.assertNotIn("--edition=2021", cmd)
            self.assertIn("--release", cmd)
            self.assertEqual(cmd[:2], [binary, "build"])


class TestPythonRuntime(unittest.TestCase):
    def test_discover_finds_python(self):
        pr = PythonRuntime()
        binary = pr.discover()
        self.assertIsNotNone(binary)

    def test_build_command(self):
        pr = PythonRuntime()
        cmd = pr.build_command(["script.py"])
        self.assertIn("-m", cmd)
        self.assertIn("py_compile", cmd)
        self.assertIn("script.py", cmd)


class TestNodeRuntime(unittest.TestCase):
    def test_build_command(self):
        nr = NodeRuntime()
        cmd = nr.build_command(["app.js"])
        self.assertIn("--check", cmd)
        self.assertIn("app.js", cmd)


class TestFortranCompiler(unittest.TestCase):
    def test_build_command_basic(self):
        fc = FortranCompiler()
        cmd = fc.build_command(["sim.f90"])
        self.assertIn("sim.f90", cmd)


class TestCompileResult(unittest.TestCase):
    def test_error_summary_first_line(self):
        r = CompileResult(
            target_name="t",
            success=False,
            command=["gcc", "x.c"],
            stderr="x.c:1:1: error: expected ';'\nnote: something else",
            return_code=1,
        )
        self.assertEqual(r.error_summary, "x.c:1:1: error: expected ';'")

    def test_error_summary_empty_stderr(self):
        r = CompileResult(target_name="t", success=False, command=[], stderr="")
        self.assertEqual(r.error_summary, "unknown error")

    def test_error_summary_success(self):
        r = CompileResult(target_name="t", success=True, command=[])
        self.assertEqual(r.error_summary, "")


class TestCompileTarget(unittest.TestCase):
    def test_unsupported_language(self):
        r = compile_target("t", "cobol", ["main.cob"])
        self.assertFalse(r.success)
        self.assertIn("unsupported", r.stderr)

    def test_python_syntax_check(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("x = 1\n")
            path = fh.name
        try:
            r = compile_target("t", "python", [path])
            self.assertTrue(r.success)
        finally:
            os.remove(path)

    def test_python_syntax_error(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("def broken(\n")
            path = fh.name
        try:
            r = compile_target("t", "python", [path])
            self.assertFalse(r.success)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
