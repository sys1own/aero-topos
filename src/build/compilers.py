# -*- coding: utf-8 -*-
"""Lean compiler invocation wrappers for the Aero build engine.

Each language backend is a subclass of :class:`CompilerBackend` — a minimal,
plugin-ready interface that knows how to:

1. **discover** the compiler binary on ``$PATH``,
2. **build a command line** from a :class:`~build_graph.TargetNode`, and
3. **execute** the compilation, returning a structured :class:`CompileResult`.

Supported backends
------------------
* :class:`CCompiler`    -- C  via ``gcc`` / ``clang``
* :class:`CppCompiler`  -- C++ via ``g++`` / ``clang++``
* :class:`RustCompiler` -- Rust via ``cargo`` (preferred) or ``rustc``
* :class:`PythonRuntime` -- validates/byte-compiles Python sources
* :class:`NodeRuntime`   -- validates Node.js sources via ``node --check``

The registry :data:`BACKENDS` maps language names (as they appear in
``blueprint.aero`` target blocks) to backend classes.  The factory
:func:`get_backend` looks up and instantiates the right one.

Design note: the interface is deliberately thin so each backend can later be
extracted into a standalone plugin package without touching the core engine.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

logger = logging.getLogger("aero.compilers")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompileResult:
    """Outcome of a single target compilation."""

    target_name: str
    success: bool
    command: List[str]
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    output_path: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_summary(self) -> str:
        """First non-empty line of stderr, or a generic message."""
        for line in self.stderr.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return "unknown error" if not self.success else ""


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class CompilerBackend(ABC):
    """Plugin-ready interface for a language compiler / runtime."""

    language: str = ""

    @abstractmethod
    def discover(self) -> Optional[str]:
        """Return the absolute path to the compiler binary, or ``None``."""

    @abstractmethod
    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Construct the full command-line argument list.

        ``options`` carries backend-specific extras (e.g. the Rust backend's
        ``manifest_path`` / ``root`` / ``cargo`` settings); backends that do not
        need them simply ignore the argument.
        """

    def compile(
        self,
        target_name: str,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> CompileResult:
        """Discover the compiler, build the command, execute, return result."""
        binary = self.discover()
        if binary is None:
            return CompileResult(
                target_name=target_name,
                success=False,
                command=[],
                stderr=f"no {self.language} compiler found on PATH",
                return_code=-1,
            )
        cmd = self.build_command(sources, output, flags, defines, workdir, options)
        return self._run(target_name, cmd, workdir)

    def _run(
        self,
        target_name: str,
        cmd: List[str],
        workdir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> CompileResult:
        cwd = str(workdir) if workdir else None
        # Always inherit the parent shell's full configuration footprint so that
        # low-memory limits (CARGO_BUILD_JOBS=1, codegen-units=1) and custom
        # compiler paths set in the environment are never dropped; any per-build
        # overrides (e.g. RUSTFLAGS) are layered on top.
        run_env: Dict[str, str] = dict(os.environ)
        if env:
            run_env.update(env)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
                env=run_env,
            )
            return CompileResult(
                target_name=target_name,
                success=proc.returncode == 0,
                command=cmd,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
            )
        except FileNotFoundError:
            return CompileResult(
                target_name=target_name,
                success=False,
                command=cmd,
                stderr=f"compiler binary not found: {cmd[0]}",
                return_code=-1,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                target_name=target_name,
                success=False,
                command=cmd,
                stderr="compilation timed out (300s limit)",
                return_code=-1,
            )


# ---------------------------------------------------------------------------
# C / C++
# ---------------------------------------------------------------------------

_C_CANDIDATES = ("gcc", "clang", "cc")
_CPP_CANDIDATES = ("g++", "clang++", "c++")


class CCompiler(CompilerBackend):
    language = "c"

    def discover(self) -> Optional[str]:
        for name in _C_CANDIDATES:
            path = shutil.which(name)
            if path:
                return path
        return None

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        binary = self.discover() or "gcc"
        cmd: List[str] = [binary]
        cmd.extend(flags)
        for define in defines:
            cmd.extend(["-D", define])
        cmd.extend(sources)
        if output:
            cmd.extend(["-o", output])
        return cmd


class CppCompiler(CompilerBackend):
    language = "cpp"

    def discover(self) -> Optional[str]:
        for name in _CPP_CANDIDATES:
            path = shutil.which(name)
            if path:
                return path
        return None

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        binary = self.discover() or "g++"
        cmd: List[str] = [binary]
        cmd.extend(flags)
        for define in defines:
            cmd.extend(["-D", define])
        cmd.extend(sources)
        if output:
            cmd.extend(["-o", output])
        return cmd


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def _cargo_build_flags(flags: Sequence[str]) -> List[str]:
    """Return *flags* with the edition argument stripped.

    ``--edition`` is a manifest / rustc-level setting; it is not accepted by
    ``cargo build`` and must be kept inside ``Cargo.toml``. Handles both
    ``--edition 2021`` and ``--edition=2021`` forms.
    """
    result: List[str] = []
    iterator = iter(flags)
    for flag in iterator:
        if flag == "--edition":
            try:
                next(iterator)
            except StopIteration:
                pass
            continue
        if flag.startswith("--edition="):
            continue
        # ``--release`` is hoisted to a dedicated cargo flag by the caller, so
        # drop it here to avoid emitting it twice.
        if flag == "--release":
            continue
        result.append(flag)
    return result


class RustCompiler(CompilerBackend):
    """Rust backend that respects user manifests and supports subdir crates.

    Build behaviour (see :mod:`src.build.cargo_manifest`):

    * if a ``Cargo.toml`` already exists at the crate root -- discovered from the
      sources, or pointed at via ``manifest_path`` / ``root`` -- it is used
      verbatim (no manifest is synthesised, so pinned/older deps are honoured);
    * otherwise a manifest is synthesised, with dependency versions taken from
      the blueprint's ``cargo.dependencies`` block;
    * ``cargo`` is always run from the resolved crate root, and artefacts are
      collected from *that* crate's ``target/`` directory.
    """

    language = "rust"

    def discover(self) -> Optional[str]:
        cargo = shutil.which("cargo")
        if cargo:
            return cargo
        return shutil.which("rustc")

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        options = options or {}
        binary = self.discover() or "cargo"
        if Path(binary).stem.lower() == "cargo":
            cmd: List[str] = [binary, "build"]
            manifest_path = options.get("manifest_path")
            if manifest_path:
                cmd.extend(["--manifest-path", str(manifest_path)])
            if options.get("release") or "--release" in flags:
                cmd.append("--release")
            # Remaining flags are cargo-level flags (e.g. --features ...).
            # The edition profile belongs in Cargo.toml, not on the cargo CLI,
            # so strip any --edition argument (and its value) that may have
            # leaked in from blueprint flags.
            cmd.extend(_cargo_build_flags(flags))
            return cmd
        # rustc fallback (no manifest / cargo): compile the sources directly.
        cmd = [binary]
        cmd.extend(flags)
        cmd.extend(sources)
        if output:
            cmd.extend(["-o", output])
        return cmd

    def compile(
        self,
        target_name: str,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> CompileResult:
        binary = self.discover()
        if binary is None:
            return CompileResult(
                target_name=target_name,
                success=False,
                command=[],
                stderr="no rust compiler found on PATH (need cargo or rustc)",
                return_code=-1,
            )

        # Without cargo we cannot honour manifests; fall back to plain rustc.
        if Path(binary).stem.lower() != "cargo":
            return super().compile(target_name, sources, output, flags, defines, workdir, options)

        from src.build.cargo_manifest import extract_cargo_options, prepare_crate

        options = options or {}
        workspace = Path(workdir) if workdir else Path.cwd()
        # Merge the nested ``cargo`` object (JSON) and the flat
        # ``cargo_dependencies`` form (block-DSL/INI) into one options dict.
        cargo_options = extract_cargo_options(options)

        try:
            plan = prepare_crate(
                workspace=workspace,
                target_name=target_name,
                sources=sources,
                cargo_options=cargo_options,
                manifest_path=options.get("manifest_path"),
                root=options.get("root"),
            )
        except OSError as exc:
            return CompileResult(
                target_name=target_name,
                success=False,
                command=[],
                stderr=f"failed to prepare cargo crate: {exc}",
                return_code=-1,
            )

        release = bool(options.get("release")) or "--release" in flags
        cmd = self.build_command(
            sources,
            output,
            flags,
            defines,
            plan.crate_root,
            {"manifest_path": str(plan.manifest_path), "release": release},
        )

        # Resolve (portable-by-default, customisable) RUSTFLAGS to inject.
        from src.build.rustflags import resolve_rustflags

        decision = resolve_rustflags(
            optimization=options.get("optimization"),
            rustflags=options.get("rustflags"),
        )
        env = decision.env()

        # Run cargo from the crate root so subdirectory crates build correctly.
        result = self._run(target_name, cmd, plan.crate_root, env=env or None)
        # Collect artefacts from *this* crate's target/ directory.
        result.output_path = str(plan.profile_dir(release))
        result.details.update(plan.to_dict())
        result.details["language"] = "rust"
        result.details["artifact_dir"] = str(plan.profile_dir(release))
        result.details["release"] = release
        result.details["command"] = cmd
        result.details["rustflags"] = decision.to_dict()
        result.details["env"] = dict(env)
        # The declared dependencies that fed the manifest (for debug + error analysis).
        result.details["declared_dependencies"] = dict(cargo_options.get("dependencies", {}) or {})
        # A preview of the manifest actually used (read back from disk).
        try:
            result.details["manifest"] = plan.manifest_path.read_text(encoding="utf-8")
        except OSError:
            result.details["manifest"] = ""
        return result


# ---------------------------------------------------------------------------
# Python (interpreted -- syntax-check / byte-compile)
# ---------------------------------------------------------------------------

_PYTHON_CANDIDATES = ("python3", "python")


class PythonRuntime(CompilerBackend):
    language = "python"

    def discover(self) -> Optional[str]:
        for name in _PYTHON_CANDIDATES:
            path = shutil.which(name)
            if path:
                return path
        return None

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        binary = self.discover() or "python3"
        cmd: List[str] = [binary, "-m", "py_compile"]
        cmd.extend(sources)
        return cmd


# ---------------------------------------------------------------------------
# Node.js (interpreted -- syntax-check)
# ---------------------------------------------------------------------------


class NodeRuntime(CompilerBackend):
    language = "node"

    def discover(self) -> Optional[str]:
        return shutil.which("node")

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        binary = self.discover() or "node"
        cmd: List[str] = [binary, "--check"]
        cmd.extend(sources)
        return cmd


# ---------------------------------------------------------------------------
# Fortran
# ---------------------------------------------------------------------------

_FORTRAN_CANDIDATES = ("gfortran", "ifort", "flang")


class FortranCompiler(CompilerBackend):
    language = "fortran"

    def discover(self) -> Optional[str]:
        for name in _FORTRAN_CANDIDATES:
            path = shutil.which(name)
            if path:
                return path
        return None

    def build_command(
        self,
        sources: Sequence[str],
        output: Optional[str] = None,
        flags: Sequence[str] = (),
        defines: Sequence[str] = (),
        workdir: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        binary = self.discover() or "gfortran"
        cmd: List[str] = [binary]
        cmd.extend(flags)
        for define in defines:
            cmd.extend(["-D", define])
        cmd.extend(sources)
        if output:
            cmd.extend(["-o", output])
        return cmd


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

BACKENDS: Dict[str, Type[CompilerBackend]] = {
    "c": CCompiler,
    "cpp": CppCompiler,
    "rust": RustCompiler,
    "python": PythonRuntime,
    "node": NodeRuntime,
    "fortran": FortranCompiler,
}


def get_backend(language: str) -> Optional[CompilerBackend]:
    """Look up and instantiate the compiler backend for *language*."""
    cls = BACKENDS.get(language.lower())
    if cls is None:
        return None
    return cls()


def compile_target(
    target_name: str,
    language: str,
    sources: Sequence[str],
    output: Optional[str] = None,
    flags: Sequence[str] = (),
    defines: Sequence[str] = (),
    workdir: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
) -> CompileResult:
    """One-shot: look up the backend and compile a target."""
    backend = get_backend(language)
    if backend is None:
        return CompileResult(
            target_name=target_name,
            success=False,
            command=[],
            stderr=f"unsupported language: {language}",
            return_code=-1,
        )
    return backend.compile(target_name, sources, output, flags, defines, workdir, options)
