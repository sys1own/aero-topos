"""Automated runtime environment bootstrapper for Aero Future.

Wraps the entire ecosystem in a zero-friction installation shell.  Before any
pipeline command runs, :class:`RuntimeEnvironmentBootstrapper`:

* **ingests dependencies** -- scans the live interpreter for the packages the
  engine needs (``numpy`` and friends) and, if one is missing, silently
  provisions it via ``sys.executable -m pip install`` instead of letting a
  ``ModuleNotFoundError`` escape to the user;
* **auto-initializes the workspace** -- if ``blueprint.aero`` is absent it
  seeds a production-grade living blueprint (system definition, a mock context
  registry and default ``auto_split_threshold`` metrics) so the tool never
  crashes on a fresh checkout.

All output is routed through the standard Aero Future telemetry headers so the
user sees crisp, organized status lines rather than bare tracebacks.

The bootstrapper is idempotent and side-effect-light: it can be invoked at the
top of every command without repeating work, and it honors the
``AERO_DISABLE_BOOTSTRAP`` environment variable so test harnesses and CI can
opt out of provisioning entirely.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Dict, List

# Map an importable module name to the pip distribution that provides it.
REQUIRED_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
}

# System toolchains required per language: not just the compiler binary but the
# high-level package manager needed for external-registry resolution.  A Rust
# project needs both ``rustc`` AND ``cargo`` (cargo drives crate fetching).
SYSTEM_TOOLCHAINS: Dict[str, List[str]] = {
    "rust": ["rustc", "cargo"],
    "auto": ["rustc", "cargo"],
    "c": ["cc"],
    "cpp": ["c++"],
    "fortran": ["gfortran"],
    "python": ["python3"],
}

# Best-effort installer commands for provisioning a missing system binary.
_TOOLCHAIN_PROVISIONERS: Dict[str, List[List[str]]] = {
    "rustc": [["rustup", "toolchain", "install", "stable"]],
    "cargo": [["rustup", "toolchain", "install", "stable"]],
}

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


class RuntimeEnvironmentBootstrapper:
    """Autonomous pre-flight provisioning of dependencies and workspace."""

    #: set once a full ``verify_and_bootstrap`` pass has completed, so repeated
    #: invocations across commands are no-ops (idempotent).
    _completed = False

    # -- dependency ingestion ---------------------------------------------
    @classmethod
    def missing_dependencies(cls, packages: Dict[str, str] | None = None) -> List[str]:
        """Return the importable names that are not currently available."""
        packages = packages or REQUIRED_PACKAGES
        missing: List[str] = []
        for module_name in packages:
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
        return missing

    @classmethod
    def provision(cls, pip_name: str) -> bool:
        """Provision ``pip_name`` into the running instance sandbox.

        Returns ``True`` on success.  Never raises: a failed install is logged
        and reported as ``False`` so callers can degrade gracefully.
        """
        _telemetry(f"[*] Dynamically provisioning '{pip_name}' into sandbox instance...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            _telemetry(
                f"[-] Automated package provisioning failed for '{pip_name}': {exc}"
            )
            return False
        # Refresh import caches so the freshly installed module is importable.
        importlib.invalidate_caches()
        _telemetry(f"[+] Successfully bootstrapped package: '{pip_name}'")
        return True

    @classmethod
    def verify_dependencies(cls, packages: Dict[str, str] | None = None) -> bool:
        """Ensure every required dependency is importable, provisioning as needed."""
        packages = packages or REQUIRED_PACKAGES
        ok = True
        for module_name in cls.missing_dependencies(packages):
            _telemetry(
                f"[*] Missing environmental runtime dependency detected: '{module_name}'"
            )
            if not cls.provision(packages[module_name]):
                ok = False
        return ok

    # -- system toolchain verification ------------------------------------
    @classmethod
    def missing_toolchain_binaries(cls, language: str) -> List[str]:
        """Return required system binaries for ``language`` that are not on PATH."""
        required = SYSTEM_TOOLCHAINS.get(language, [])
        return [binary for binary in required if shutil.which(binary) is None]

    @classmethod
    def provision_system_binary(cls, binary: str) -> bool:
        """Best-effort provisioning of a missing system binary via a known installer.

        Returns ``True`` if the binary is present afterwards.  Never raises: a
        missing installer or a failed install is logged and reported as ``False``
        so the caller can surface actionable guidance instead of crashing.
        """
        if shutil.which(binary) is not None:
            return True
        for command in _TOOLCHAIN_PROVISIONERS.get(binary, []):
            installer = command[0]
            if shutil.which(installer) is None:
                continue
            _telemetry(f"[*] Provisioning system toolchain '{binary}' via '{installer}'...")
            try:
                subprocess.check_call(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
            except (subprocess.CalledProcessError, OSError) as exc:
                _telemetry(f"[-] Toolchain provisioning failed for '{binary}': {exc}")
                continue
            if shutil.which(binary) is not None:
                _telemetry(f"[+] Successfully provisioned system toolchain: '{binary}'")
                return True
        return shutil.which(binary) is not None

    @classmethod
    def verify_toolchain(cls, language: str) -> bool:
        """Verify (and best-effort provision) the system toolchain for ``language``.

        For a Rust/``auto`` project this guarantees both ``rustc`` and ``cargo``
        are available so external-registry crate resolution can proceed.
        Returns ``True`` when every required binary is present.
        """
        missing = cls.missing_toolchain_binaries(language)
        if not missing:
            return True
        _telemetry(
            f"[*] Missing system toolchain binaries for '{language}': {missing}"
        )
        ok = True
        for binary in missing:
            if not cls.provision_system_binary(binary):
                ok = False
                _telemetry(
                    f"[-] Toolchain binary '{binary}' is unavailable. Install it "
                    f"(e.g. via rustup) to enable '{language}' multi-crate builds."
                )
        return ok

    # -- workspace auto-init ----------------------------------------------
    @classmethod
    def ensure_blueprint(cls, root: str = ".") -> bool:
        """Seed a default living ``blueprint.aero`` if one is absent.

        Returns ``True`` if a blueprint was created, ``False`` if one already
        existed (and was therefore preserved untouched).
        """
        blueprint_path = os.path.join(root, "blueprint.aero")
        if os.path.exists(blueprint_path):
            return False

        _telemetry("[*] No active blueprint.aero detected in workspace root.")
        _telemetry("[*] Generating production-grade living blueprint template automatically...")

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

        Backs the ``python main.py init`` command.  Always reports what it did.
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

    # -- orchestration -----------------------------------------------------
    @classmethod
    def verify_and_bootstrap(cls, root: str = ".", *, force: bool = False) -> bool:
        """Full pre-flight pass: dependencies + workspace.  Idempotent.

        Honors ``AERO_DISABLE_BOOTSTRAP`` (skips entirely) so tests/CI opt out.
        Returns ``True`` when the environment is ready.
        """
        if os.environ.get("AERO_DISABLE_BOOTSTRAP"):
            return True
        if cls._completed and not force:
            return True

        _telemetry_banner("AERO FUTURE PRE-FLIGHT ENVIRONMENT BOOTSTRAPPING")
        deps_ok = cls.verify_dependencies()
        if not deps_ok:
            _telemetry(
                "[-] One or more dependencies could not be provisioned; "
                "continuing in degraded mode."
            )
        else:
            _telemetry("[+] All runtime dependencies satisfied.")
        cls.ensure_blueprint(root)
        cls._completed = True
        return deps_ok


__all__ = [
    "RuntimeEnvironmentBootstrapper",
    "REQUIRED_PACKAGES",
    "DEFAULT_BLUEPRINT",
]
