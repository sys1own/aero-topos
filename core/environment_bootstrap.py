# -*- coding: utf-8 -*-
"""Environment contract helpers and explicit workspace initialization.

This module no longer performs automatic package or toolchain installation.
Provisioning has been replaced by :mod:`core.verify_dependencies`, which
raises :class:`core.verify_dependencies.ContractViolationError` when the host
environment does not satisfy the active blueprint.  The only side effects this
module is allowed to produce are explicit ``main.py init`` workspace seeds.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.verify_dependencies import (
    DEFAULT_PYTHON_PACKAGES,
    SYSTEM_TOOLCHAINS,
    ContractViolationError,
    VerifyDependencies,
)

# Backwards-compatible re-exports.
REQUIRED_PACKAGES = DEFAULT_PYTHON_PACKAGES


# Default living-blueprint template seeded into a fresh workspace.
DEFAULT_BLUEPRINT = """[system]
name = "autonomous-autobootstrap-workspace"
strategy = "universal-engine"
ephemeral_code = true

[context_registry.core_application]
path = "./src/app_logic.py"
language = "python"
preserve_original_logic = false
compile_target = "aero-calculus"

[scaling]
auto_split_threshold = 120
max_module_complexity = 12
hierarchy_depth = 4
"""

DEFAULT_APP_LOGIC = (
    "# Auto-generated core application logic stub.\n"
    "# Replace this with your program; Aero Future compiles it to Aero-Calculus.\n"
    "def main() -> str:\n"
    '    return "Aero substrate online"\n'
    "\n\n"
    'if __name__ == "__main__":\n'
    "    print(main())\n"
)


def _telemetry_banner(title: str) -> None:
    rule = "=" * 78
    print(rule)
    print(f" {title}")
    print(rule)


def _telemetry(line: str) -> None:
    print(line)


class RuntimeEnvironmentBootstrapper(VerifyDependencies):
    """Read-only environment verifier plus explicit workspace initialization.

    The legacy provisioning methods (``provision``, ``provision_system_binary``,
    ``verify_and_bootstrap``) are gone.  Any code that still calls them receives
    a :class:`ContractViolationError` instead of mutating the host.
    """

    # -- dependency checks --------------------------------------------------
    @classmethod
    def missing_dependencies(
        cls, packages: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Return importable names from *packages* that are not installed."""
        return VerifyDependencies.missing_dependencies(packages)

    @classmethod
    def missing_toolchain_binaries(cls, language: str) -> List[str]:
        """Return required binaries for *language* that are absent from PATH."""
        return VerifyDependencies.missing_toolchain_binaries(language)

    @classmethod
    def verify_dependencies(cls, packages: Optional[Dict[str, str]] = None) -> bool:
        """Return ``True`` when all *packages* are importable.

        No installation is attempted; this is a pure read-only check.
        """
        return not cls.missing_dependencies(packages)

    @classmethod
    def verify_toolchain(cls, language: str) -> bool:
        """Return ``True`` when every binary for *language* is on PATH."""
        return VerifyDependencies.verify_toolchain(language)

    # -- disabled provisioning hooks ----------------------------------------
    @classmethod
    def provision(cls, pip_name: str) -> bool:
        """Automatic package installation is disabled by the Environment Contract."""
        raise ContractViolationError(
            f"Contract Violation: automatic installation of '{pip_name}' is not allowed. "
            f"Install it manually, e.g. `pip install {pip_name}`."
        )

    @classmethod
    def provision_system_binary(cls, binary: str) -> bool:
        """Automatic toolchain installation is disabled by the Environment Contract."""
        raise ContractViolationError(
            f"Contract Violation: automatic installation of system binary '{binary}' "
            "is not allowed. Install the required toolchain manually."
        )

    @classmethod
    def verify_and_bootstrap(cls, root: str = ".", *, force: bool = False) -> bool:
        """Legacy pre-flight hook: disabled under the Environment Contract.

        Automatic dependency ingestion and blueprint seeding during command
        dispatch is no longer performed.  The active command handler is
        responsible for running :meth:`VerifyDependencies.verify` against the
        parsed blueprint before any build/scaffold work.
        """
        raise ContractViolationError(
            "Contract Violation: automatic environment bootstrapping is disabled. "
            "Use the active command's pre-flight contract check or `main.py init` instead."
        )

    # -- explicit workspace initialization ----------------------------------
    @classmethod
    def ensure_blueprint(cls, root: str = ".") -> bool:
        """Seed a default living ``blueprint.aero`` if one is absent.

        This is only triggered by ``main.py init``; it is never called
        automatically during a build.
        """
        blueprint_path = os.path.join(root, "blueprint.aero")
        if os.path.exists(blueprint_path):
            return False

        _telemetry("[*] No active blueprint.aero detected in workspace root.")
        _telemetry("[*] Generating production-grade living blueprint template...")

        src_dir = os.path.join(root, "src")
        os.makedirs(src_dir, exist_ok=True)
        app_logic = os.path.join(src_dir, "app_logic.py")
        if not os.path.exists(app_logic):
            with open(app_logic, "w", encoding="utf-8") as handle:
                handle.write(DEFAULT_APP_LOGIC)

        with open(blueprint_path, "w", encoding="utf-8") as handle:
            handle.write(DEFAULT_BLUEPRINT.strip() + "\n")

        _telemetry("[+] Living blueprint schema successfully seeded into workspace.")
        return True

    @classmethod
    def init_workspace(cls, root: str = ".") -> Dict[str, object]:
        """Explicitly (re)initialize a project architecture under ``root``.

        Backs the ``python main.py init`` command.
        """
        _telemetry_banner("AERO FUTURE WORKSPACE INITIALIZATION")
        os.makedirs(root, exist_ok=True)
        created_blueprint = cls.ensure_blueprint(root)

        # Seed the Block Universe ledger genesis block when available.
        ledger_created = False
        ledger_path = os.path.join(root, "context.aero")
        if not os.path.exists(ledger_path):
            try:
                from context_init import initialize_workspace_context

                initialize_workspace_context(root, "HEAD", "src/app_logic.py")
                ledger_created = True
            except Exception:  # noqa: BLE001 - ledger seeding is best-effort
                ledger_created = False

        report = {
            "root": os.path.abspath(root),
            "blueprint_created": created_blueprint,
            "ledger_created": ledger_created,
        }
        _telemetry(
            f"[+] Workspace ready at {report['root']} "
            f"(blueprint={'seeded' if created_blueprint else 'preserved'}, "
            f"ledger={'seeded' if ledger_created else 'present/unavailable'})"
        )
        return report


__all__ = [
    "ContractViolationError",
    "RuntimeEnvironmentBootstrapper",
    "REQUIRED_PACKAGES",
    "SYSTEM_TOOLCHAINS",
    "DEFAULT_BLUEPRINT",
]
