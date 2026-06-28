"""Tests for the production-hardening layer: bootstrapper + pre-flight auditor."""

from __future__ import annotations

import os
import sys
import textwrap
import unittest
from pathlib import Path

import main
from core.environment_bootstrap import (
    DEFAULT_BLUEPRINT,
    RuntimeEnvironmentBootstrapper,
)
from core.test_auditor import PreFlightTestAuditor, normalize_path_literals


# ---------------------------------------------------------------------------
# Bootstrapper
# ---------------------------------------------------------------------------
class TestBootstrapper(unittest.TestCase):
    def test_missing_dependencies_detects_absent_module(self):
        missing = RuntimeEnvironmentBootstrapper.missing_dependencies(
            {"definitely_not_a_real_module_xyz": "nope"}
        )
        self.assertEqual(missing, ["definitely_not_a_real_module_xyz"])

    def test_present_dependency_not_flagged(self):
        # numpy is installed in this environment.
        self.assertEqual(
            RuntimeEnvironmentBootstrapper.missing_dependencies({"numpy": "numpy"}),
            [],
        )

    def test_ensure_blueprint_seeds_when_absent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            created = RuntimeEnvironmentBootstrapper.ensure_blueprint(d)
            self.assertTrue(created)
            self.assertTrue(os.path.isfile(os.path.join(d, "blueprint.aero")))
            self.assertTrue(os.path.isfile(os.path.join(d, "src", "app_logic.py")))
            # Idempotent: a second pass preserves the existing blueprint.
            self.assertFalse(RuntimeEnvironmentBootstrapper.ensure_blueprint(d))

    def test_init_workspace_reports(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            report = RuntimeEnvironmentBootstrapper.init_workspace(d)
            self.assertTrue(report["blueprint_created"])
            self.assertEqual(report["root"], os.path.abspath(d))

    def test_default_blueprint_carries_auto_split_threshold(self):
        self.assertIn("auto_split_threshold = 120", DEFAULT_BLUEPRINT)


# ---------------------------------------------------------------------------
# Path-literal normalization
# ---------------------------------------------------------------------------
class TestPathNormalization(unittest.TestCase):
    def test_windows_separators_rewritten(self):
        src = 'p = open("sub\\\\data.txt")\n'  # source contains sub\\data.txt
        out, changed = normalize_path_literals(src)
        self.assertTrue(changed)
        self.assertIn("sub/data.txt", out)

    def test_control_escapes_preserved(self):
        src = 'print("line\\n\\tindented")\n'  # \n and \t must survive
        out, changed = normalize_path_literals(src)
        self.assertFalse(changed)
        self.assertEqual(out, src)

    def test_forward_slash_paths_untouched(self):
        src = 'open("a/b/c.txt")\n'
        out, changed = normalize_path_literals(src)
        self.assertFalse(changed)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
class TestClassification(unittest.TestCase):
    def setUp(self):
        self.auditor = PreFlightTestAuditor()

    def test_module_not_found_is_environment(self):
        tb = "ERROR: test_x\nModuleNotFoundError: No module named 'foo'\n"
        self.assertEqual(self.auditor.classify(tb), "environment")

    def test_assertion_is_logic(self):
        tb = "FAIL: test_y\nAssertionError: 1 != 2\n"
        self.assertEqual(self.auditor.classify(tb), "logic")

    def test_route_env_error_extracts_module(self):
        # Provisioning a clearly-bogus module returns False (no crash).
        tb = "ModuleNotFoundError: No module named 'this_module_does_not_exist_zzz'"
        self.assertFalse(self.auditor._route_env_error(tb))


# ---------------------------------------------------------------------------
# End-to-end self-healing audit (isolated temp workspace)
# ---------------------------------------------------------------------------
class TestAuditorSelfHeal(unittest.TestCase):
    def _make_workspace(self, d: str) -> None:
        # A buggy module that opens a file using a hardcoded Windows path
        # separator -- fails on POSIX until self-healed to a portable path.
        Path(d, "sub").mkdir()
        Path(d, "sub", "data.txt").write_text("payload", encoding="utf-8")
        Path(d, "buggy.py").write_text(
            'def read_payload():\n'
            '    with open("sub\\\\data.txt", "r", encoding="utf-8") as fh:\n'
            '        return fh.read()\n',
            encoding="utf-8",
        )
        tests_dir = Path(d, "tests")
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "test_buggy.py").write_text(
            textwrap.dedent(
                """
                import unittest
                import buggy

                class T(unittest.TestCase):
                    def test_reads(self):
                        self.assertEqual(buggy.read_payload(), "payload")

                if __name__ == "__main__":
                    unittest.main()
                """
            ),
            encoding="utf-8",
        )

    def test_pathing_bug_is_self_healed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._make_workspace(d)
            auditor = PreFlightTestAuditor(
                test_dir=os.path.join(d, "tests"), top_level=d, max_rounds=3
            )
            # On POSIX the suite fails first, then the auditor patches buggy.py.
            ok = auditor.run_suite_and_heal()
            self.assertTrue(ok)
            # The fix was applied in place: the literal is now portable.
            patched = Path(d, "buggy.py").read_text(encoding="utf-8")
            self.assertIn("sub/data.txt", patched)
            self.assertIn(os.path.join(d, "buggy.py"), auditor.patched_files)

    def test_clean_suite_passes_without_patching(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tests_dir = Path(d, "tests")
            tests_dir.mkdir()
            (tests_dir / "test_ok.py").write_text(
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            auditor = PreFlightTestAuditor(test_dir=str(tests_dir), top_level=d)
            self.assertTrue(auditor.run_suite_and_heal())
            self.assertEqual(auditor.patched_files, [])


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------
class TestCLIHardening(unittest.TestCase):
    def test_init_command(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            rc = main.main(["init", "--workspace", d])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(os.path.join(d, "blueprint.aero")))

    def test_parser_knows_init_and_audit(self):
        parser = main.create_parser()
        self.assertIs(parser.parse_args(["init"]).handler, main.init_command)
        self.assertIs(parser.parse_args(["audit"]).handler, main.audit_command)

    def test_main_does_not_bootstrap_under_tests(self):
        # 'unittest' is in sys.modules here, so _maybe_bootstrap is a no-op:
        # calling a normal command must not seed blueprint.aero in the CWD.
        self.assertIn("unittest", sys.modules)
        main._BOOTSTRAP_DONE = False
        # init on a temp dir is fine; the guard concerns implicit CWD seeding.
        # Just assert the guard short-circuits cleanly.
        main._maybe_bootstrap()
        # No exception, and the guard left the flag unset (skipped under tests).
        self.assertFalse(main._BOOTSTRAP_DONE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
