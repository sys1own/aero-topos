"""Ephemeral multi-crate workspace scaffolding for polyglot compilation.

When a polyglot source (e.g. a native Rust file) pulls in external registry
crates -- ``pyo3``, ``rug``, ``rayon`` -- a direct ``rustc`` invocation fails
because the crates are never resolved.  :class:`EphemeralCargoScaffolder`
intercepts the raw source, extracts its external crate handles, provisions a
transient Cargo workspace with an auto-synthesised ``Cargo.toml``, runs
``cargo build`` under the hood, and extracts the resulting native shared object.

The workspace lives under a sandboxed scratch root and is always cleaned up, so
no residual directories leak after compilation.  Manifest synthesis reuses the
repository's tested :func:`render_manifest` so generated manifests pick up the
same ``pyo3`` feature handling as the rest of the toolchain.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

from src.build.cargo_manifest import render_manifest

# Rust prelude / path keywords that are never external crates.
_INTERNAL_CRATES = {"std", "core", "alloc", "super", "crate", "self"}

# ``use crate_name::...`` and ``extern crate crate_name;`` handles.
_USE_RE = re.compile(r"\buse\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*::")
_EXTERN_CRATE_RE = re.compile(r"\bextern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*)")

# Version pins for crates whose wildcard resolution is unstable or where a
# known-good baseline materially helps (e.g. high-precision math).  Everything
# else defaults to a permissive wildcard so registry resolution succeeds.
_KNOWN_VERSIONS: Dict[str, str] = {
    "rug": "1.24",
    "pyo3": "0.22",
}

# Shared-object extensions emitted across platforms.
_ARTIFACT_EXTS = (".so", ".dll", ".dylib")


class EphemeralCargoScaffolder:
    """Build a single Rust source into a native shared object via cargo."""

    PACKAGE_NAME = "aero_ephemeral_build"

    def __init__(self, scratch_space: str = "/tmp/aero_scaffold"):
        self.scratch_space = scratch_space
        # Populated per build by :meth:`generate_scaffold_env`.
        self.workspace: Optional[str] = None
        self.src_dir: Optional[str] = None

    # -- dependency extraction --------------------------------------------
    def extract_dependencies(self, source_path: str) -> List[str]:
        """Isolate external crate handles referenced by the source file."""
        try:
            with open(source_path, "r", errors="ignore", encoding="utf-8") as handle:
                content = handle.read()
        except OSError:
            return []
        return self.extract_dependencies_from_text(content)

    @staticmethod
    def extract_dependencies_from_text(content: str) -> List[str]:
        """Extract external crate names from raw Rust source text."""
        handles = set(_USE_RE.findall(content)) | set(_EXTERN_CRATE_RE.findall(content))
        discovered = sorted(handles - _INTERNAL_CRATES)
        return discovered

    # -- manifest synthesis -----------------------------------------------
    def _dependency_specs(self, crates: List[str]) -> Dict[str, str]:
        """Map discovered crate handles to ``[dependencies]`` version specs."""
        return {crate: _KNOWN_VERSIONS.get(crate, "*") for crate in crates}

    def synthesize_manifest(self, crates: List[str]) -> str:
        """Render a ``Cargo.toml`` declaring a ``cdylib`` with the given crates."""
        return render_manifest(
            crate_name=self.PACKAGE_NAME,
            dependencies=self._dependency_specs(crates),
            edition="2021",
            crate_type=["cdylib"],
            header="Ephemeral Aero scaffold workspace -- regenerated per build.",
        )

    # -- workspace generation ---------------------------------------------
    def generate_scaffold_env(self, original_source_path: str) -> str:
        """Provision a fresh transient workspace and return its path."""
        os.makedirs(self.scratch_space, exist_ok=True)
        # A unique workspace per build keeps concurrent compilations isolated
        # and avoids clobbering a shared directory mid-build.
        self.workspace = tempfile.mkdtemp(prefix="ws_", dir=self.scratch_space)
        self.src_dir = os.path.join(self.workspace, "src")
        os.makedirs(self.src_dir, exist_ok=True)

        dependencies = self.extract_dependencies(original_source_path)
        print(f"[+] Scaffold Engine: discovered external crate dependencies: {dependencies}")

        manifest = self.synthesize_manifest(dependencies)
        with open(os.path.join(self.workspace, "Cargo.toml"), "w", encoding="utf-8") as handle:
            handle.write(manifest)

        shutil.copy(original_source_path, os.path.join(self.src_dir, "lib.rs"))
        return self.workspace

    # -- compilation -------------------------------------------------------
    def execute_cargo_compile(self, original_source_path: str, output_so_path: str) -> bool:
        """Scaffold, build and extract the shared object; always clean up."""
        if shutil.which("cargo") is None:
            print("[-] cargo binary not found in environment; cannot run multi-crate build.")
            return False

        workspace = self.generate_scaffold_env(original_source_path)
        print("[*] Scaffold workspace stabilized. Launching multi-crate cargo dependency compilation...")
        try:
            try:
                result = subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=workspace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=900,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"[-] Cargo invocation failed: {exc}")
                return False

            if result.returncode != 0:
                print("[-] Cargo compilation failed. Triage report:\n" + (result.stderr or ""))
                return False

            return self._relocate_artifact(workspace, output_so_path)
        finally:
            self.cleanup()

    def _relocate_artifact(self, workspace: str, output_so_path: str) -> bool:
        """Copy the built shared object to ``output_so_path``."""
        release_dir = os.path.join(workspace, "target", "release")
        if not os.path.isdir(release_dir):
            print("[-] No release directory produced; build emitted no artifact.")
            return False
        for name in sorted(os.listdir(release_dir)):
            if name.endswith(_ARTIFACT_EXTS):
                out_parent = os.path.dirname(os.path.abspath(output_so_path))
                if out_parent:
                    os.makedirs(out_parent, exist_ok=True)
                shutil.copy(os.path.join(release_dir, name), output_so_path)
                print(f"[+] Scaffold build matrix successful. Relocated artifact to: {output_so_path}")
                return True
        print("[-] Build succeeded but no shared object (.so/.dll/.dylib) was found.")
        return False

    # -- cleanup -----------------------------------------------------------
    def cleanup(self) -> None:
        """Remove the transient workspace, guaranteeing zero residual leaks."""
        if self.workspace and os.path.isdir(self.workspace):
            shutil.rmtree(self.workspace, ignore_errors=True)
        self.workspace = None
        self.src_dir = None


# ---------------------------------------------------------------------------
# Compiler-pipeline hook
# ---------------------------------------------------------------------------
def compile_polyglot_source(source_path: str, output_path: str) -> bool:
    """Route a polyglot source through the ephemeral scaffolder when applicable.

    Returns ``True`` on a successful native build, ``False`` otherwise (so the
    caller can fall back to baseline diagnostic recovery).  Currently handles
    Rust (``.rs``); other extensions return ``False`` to signal "not handled".
    """
    _, ext = os.path.splitext(source_path)
    if ext == ".rs":
        print("[*] Intercepting polyglot pipeline: deploying Ephemeral Cargo Scaffolder.")
        scaffolder = EphemeralCargoScaffolder()
        success = scaffolder.execute_cargo_compile(source_path, output_path)
        if success:
            print("[+] Structural pipeline lowering pass completed via temporary workspace aggregation.")
        else:
            print("[-] Directing pipeline to standard baseline diagnostic recovery routes.")
        return success
    return False


__all__ = ["EphemeralCargoScaffolder", "compile_polyglot_source"]
