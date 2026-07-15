"""Tests for the strict Environment Contract verifier."""

from __future__ import annotations

import shutil
import unittest
from unittest import mock

from core.verify_dependencies import (
    SYSTEM_TOOLCHAINS,
    ContractViolationError,
    VerifyDependencies,
)


class TestVerifyDependencies(unittest.TestCase):
    def test_empty_contract_passes(self):
        """A blueprint with no declared languages or tools passes by default."""
        VerifyDependencies({}).verify()

    def test_missing_tool_raises_contract_violation(self):
        """A declared Rust target without cargo/rustc on PATH is a violation."""
        with mock.patch.object(shutil, "which", return_value=None):
            with self.assertRaises(ContractViolationError) as ctx:
                VerifyDependencies.for_language("rust").verify()
        self.assertIn("cargo", str(ctx.exception))
        self.assertIn("rustc", str(ctx.exception))
        self.assertIn("Contract Violation", str(ctx.exception))

    def test_present_toolchain_passes(self):
        """When cargo and rustc are available, the rust contract is satisfied."""
        # Assume a real environment has at least python3; mock rust as present.
        with mock.patch.object(
            shutil, "which", side_effect=lambda name: "/fake/" + name
        ):
            VerifyDependencies.for_language("rust").verify()

    def test_blueprint_derives_languages_from_context_registry(self):
        """Languages in [context_registry] drive tool requirements."""
        blueprint = {
            "context_registry": {
                "web": {"language": "python"},
                "native": {"language": "rust"},
            }
        }
        tools = VerifyDependencies(blueprint).required_tools()
        self.assertIn("python3", tools)
        self.assertIn("cargo", tools)
        self.assertIn("rustc", tools)

    def test_environment_contract_extends_requirements(self):
        """[environment_contract] can add extra tools and packages."""
        blueprint = {
            "environment_contract": {
                "required_tools": ["cc"],
                "required_python_packages": {"definitely_not_a_real_module_xyz": "nope"},
            }
        }
        with self.assertRaises(ContractViolationError) as ctx:
            VerifyDependencies(blueprint).verify()
        msg = str(ctx.exception)
        self.assertIn("definitely_not_a_real_module_xyz", msg)

    def test_system_toolchains_map(self):
        """The canonical toolchain mapping is exported and complete."""
        self.assertEqual(SYSTEM_TOOLCHAINS["rust"], ["rustc", "cargo"])
        self.assertEqual(SYSTEM_TOOLCHAINS["python"], ["python3"])


if __name__ == "__main__":
    unittest.main()
