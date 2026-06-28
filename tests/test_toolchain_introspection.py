# -*- coding: utf-8 -*-
"""Tests for the host toolchain introspection engine."""

import os
import shutil
import tempfile
import unittest

from core.toolchain import Toolchain, ToolchainIntrospector


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.intro = ToolchainIntrospector(workspace=self.ws)

    def tearDown(self):
        self.tmp.cleanup()


class TestDiscovery(_TmpCase):
    @unittest.skipUnless(_has("cc") or _has("gcc") or _has("clang"), "no C compiler")
    def test_discover_c(self):
        tc = self.intro.discover("c")
        self.assertIsNotNone(tc)
        self.assertEqual(tc.kind, "compiler")
        self.assertTrue(tc.path)
        self.assertIsNotNone(tc.version)
        self.assertRegex(tc.version, r"\d+\.\d+")

    @unittest.skipUnless(_has("rustc"), "no rustc")
    def test_discover_rust_arch_signature(self):
        tc = self.intro.discover("rust")
        self.assertIsNotNone(tc)
        # rustc -vV exposes a 'host:' target triple.
        self.assertIsNotNone(tc.target)
        self.assertIn("-", tc.target)

    @unittest.skipUnless(_has("python3") or _has("python"), "no python")
    def test_discover_python_runtime(self):
        tc = self.intro.discover("python")
        self.assertIsNotNone(tc)
        self.assertEqual(tc.kind, "runtime")

    def test_unknown_language(self):
        self.assertIsNone(self.intro.discover("klingon"))

    def test_languages_from_extensions(self):
        langs = self.intro.languages_from_extensions([".c", ".rs", ".py", ".txt"])
        self.assertEqual(langs, ["c", "python", "rust"])


class TestSanityAndCache(_TmpCase):
    @unittest.skipUnless(_has("cc") or _has("gcc") or _has("clang"), "no C compiler")
    def test_sanity_check_and_cache(self):
        tc = self.intro.discover("c")
        self.assertTrue(self.intro.sanity_check(tc))
        self.assertTrue(tc.sane)
        # Successful validation caches the config locally.
        cached = self.intro.cached_toolchain("c")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.path, tc.path)
        self.assertTrue(cached.sane)

    @unittest.skipUnless(_has("python3") or _has("python"), "no python")
    def test_runtime_sanity(self):
        tc = self.intro.discover("python")
        self.assertTrue(self.intro.sanity_check(tc))


class TestEnvOverrides(_TmpCase):
    @unittest.skipUnless(_has("clang"), "no clang for override test")
    def test_cc_override_with_flags(self):
        env = dict(os.environ, CC="clang -O2", CFLAGS="-Wall -Wextra", LDFLAGS="-s")
        intro = ToolchainIntrospector(workspace=self.ws, env=env)
        tc = intro.discover("c")
        self.assertEqual(tc.binary, "clang")
        self.assertIn("-O2", tc.extra_flags)
        self.assertEqual(tc.compile_flags, ["-Wall", "-Wextra"])
        self.assertEqual(tc.link_flags, ["-s"])
        # Still compiles cleanly with the overridden binary + flags.
        self.assertTrue(intro.sanity_check(tc))


class TestRegistryDriven(_TmpCase):
    def test_languages_from_registry(self):
        bp = os.path.join(self.ws, "blueprint.aero")
        with open(bp, "w", encoding="utf-8") as fh:
            fh.write(
                '[context_registry.legacy]\npath = "x"\nlanguage = "c"\n\n'
                '[context_registry.modern]\npath = "y"\nlanguage = "rust"\n'
            )
        langs = self.intro.languages_from_registry(bp)
        self.assertEqual(langs, ["c", "rust"])

    def test_introspect_returns_mapping(self):
        # Default to all specs; result only contains toolchains actually present.
        found = self.intro.introspect(validate=False)
        self.assertIsInstance(found, dict)
        for lang, tc in found.items():
            self.assertIsInstance(tc, Toolchain)
            self.assertTrue(tc.path)


class TestLinker(_TmpCase):
    @unittest.skipUnless(_has("ld") or _has("ld.lld") or _has("lld"), "no linker")
    def test_discover_linker(self):
        ld = self.intro.discover_linker()
        self.assertIsNotNone(ld)
        self.assertEqual(ld.kind, "linker")
        self.assertTrue(ld.path)


if __name__ == "__main__":
    unittest.main()
