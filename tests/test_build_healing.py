# -*- coding: utf-8 -*-
"""Tests for the orchestrator's retry-with-healing build hook."""

import os
import tempfile
import unittest
from unittest import mock

from orchestrator import handle_aero_calculus_build


class TestBuildHealing(unittest.TestCase):
    def test_retries_after_invariant_failure_with_topological_healing(self):
        """A build that fails during reduction is retried once after healing."""
        calls = []

        def fake_adopt(primary, **kwargs):
            calls.append("adopt")
            if len(calls) == 1:
                raise ValueError("simulated invariant violation")
            fake = mock.MagicMock()
            fake.run_to_completion.return_value = 5
            return fake

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "prog.py")
            out = os.path.join(tmp, "prog.aeroc")
            with open(src, "w", encoding="utf-8") as handle:
                handle.write("def f(x): return x + 1\n")

            with mock.patch("core.hin_vm.UniversalHINNetwork.adopt", side_effect=fake_adopt):
                report = handle_aero_calculus_build(src, out, reduce_graph=True, max_healing_attempts=1)

            self.assertEqual(len(calls), 2)
            self.assertTrue(os.path.isfile(out))
            self.assertEqual(report["reduction_steps"], 5)

    def test_gives_up_after_healing_callback_returns_false(self):
        """A build that cannot be healed propagates the original exception."""
        calls = []

        def fake_adopt(primary, **kwargs):
            calls.append("adopt")
            raise ValueError("persistent invariant violation")

        def failing_heal(_network):
            return False

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "prog.py")
            out = os.path.join(tmp, "prog.aeroc")
            with open(src, "w", encoding="utf-8") as handle:
                handle.write("def f(x): return x + 1\n")

            with mock.patch("core.hin_vm.UniversalHINNetwork.adopt", side_effect=fake_adopt):
                with self.assertRaises(ValueError):
                    handle_aero_calculus_build(
                        src,
                        out,
                        reduce_graph=True,
                        heal_callback=failing_heal,
                        max_healing_attempts=1,
                    )

            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
