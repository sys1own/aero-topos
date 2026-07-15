# -*- coding: utf-8 -*-
"""Tests that ``main.py build --cycles N`` triggers the cNrGA evolution loop."""

import argparse
import os
import tempfile
import unittest
from unittest import mock

import main


class TestBuildEvolutionUnification(unittest.TestCase):
    def test_cycles_positive_triggers_evolution_loop(self):
        """--cycles > 0 seeds self_host.aero and runs execute_evolution_loop."""
        with tempfile.TemporaryDirectory() as tmp:
            blueprint = os.path.join(tmp, "blueprint.aero")
            with open(blueprint, "w", encoding="utf-8") as handle:
                handle.write('[system]\nstrategy = "DIRECT_COMPILE"\n')

            with mock.patch("evolve.execute_evolution_loop") as mock_evolve, \
                 mock.patch("orchestrator.run_direct_compile", return_value={
                     "compiled_target_count": 1,
                     "bytes_written": 1,
                     "aeroc_output": os.path.join(tmp, "out.aeroc"),
                 }) as mock_compile:
                args = argparse.Namespace(
                    source=None,
                    workspace=tmp,
                    blueprint=blueprint,
                    config=None,
                    cycles=3,
                    no_scaffold_build=False,
                    no_reduce=False,
                )
                rc = main.build_command(args)

            self.assertEqual(rc, 0)
            mock_evolve.assert_called_once_with(tmp, 3)
            mock_compile.assert_called_once()
            self.assertTrue(os.path.isfile(os.path.join(tmp, "self_host.aero")))

    def test_cycles_zero_does_not_trigger_evolution(self):
        """--cycles 0 falls through to the normal direct compile path."""
        with tempfile.TemporaryDirectory() as tmp:
            blueprint = os.path.join(tmp, "blueprint.aero")
            with open(blueprint, "w", encoding="utf-8") as handle:
                handle.write('[system]\nstrategy = "DIRECT_COMPILE"\n')

            with mock.patch("evolve.execute_evolution_loop") as mock_evolve, \
                 mock.patch("orchestrator.run_direct_compile", return_value={
                     "compiled_target_count": 1,
                     "bytes_written": 1,
                     "aeroc_output": os.path.join(tmp, "out.aeroc"),
                 }) as mock_compile:
                args = argparse.Namespace(
                    source=None,
                    workspace=tmp,
                    blueprint=blueprint,
                    config=None,
                    cycles=0,
                    no_scaffold_build=False,
                    no_reduce=False,
                )
                rc = main.build_command(args)

            self.assertEqual(rc, 0)
            mock_evolve.assert_not_called()
            mock_compile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
