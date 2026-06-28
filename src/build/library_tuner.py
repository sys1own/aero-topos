"""
Numerical Library Auto-Tuning.

Probes the host for the numerical libraries a physics simulator typically links
against -- BLAS, LAPACK, MPI and CUDA -- using ``pkg-config``, a CMake-style
``find`` fallback, and well-known environment variables.  For each library it
emits the include/linker flags needed by both ``rustc`` and the C/C++/Fortran
compilers.

The detector is deliberately tolerant: every probe is wrapped so that a missing
tool or library yields a :class:`DetectedLibrary` with ``found=False`` rather
than an exception.  This keeps single-machine builds working when none of the
heavy numerical stack is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.serialization import dataclass_to_dict


# Candidate gene values exposed to the evolutionary engine (feature #2).  Only
# the values that are actually available on the host are offered as genes, so
# the engine never proposes an un-buildable combination.
_BLAS_CANDIDATES = ["auto", "mkl", "openblas", "none"]
_MPI_FLAVORS = ["openmpi", "mpich"]


@dataclass
class DetectedLibrary:
    """The result of probing for a single numerical library."""

    name: str
    found: bool = False
    flavor: str = "none"
    version: str = ""
    include_dirs: List[str] = field(default_factory=list)
    lib_dirs: List[str] = field(default_factory=list)
    libs: List[str] = field(default_factory=list)
    cflags: List[str] = field(default_factory=list)
    linker_flags: List[str] = field(default_factory=list)
    detected_via: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


class LibraryTuner:
    """Detects numerical libraries and emits build flags for them."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        lib_cfg = self.config.get("libraries", {}) or {}
        self.blas_choice = str(lib_cfg.get("blas", "auto")).lower()
        self.lapack_choice = str(lib_cfg.get("lapack", "auto")).lower()
        self.mpi_enabled = bool(lib_cfg.get("mpi", False))
        self.mpi_flavor = lib_cfg.get("mpi_flavor") or None
        self.cuda_choice = str(lib_cfg.get("cuda", "none")).lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self) -> Dict[str, DetectedLibrary]:
        """Probe every configured library and return the detection results."""
        return {
            "blas": self.detect_blas(),
            "lapack": self.detect_lapack(),
            "mpi": self.detect_mpi(),
            "cuda": self.detect_cuda(),
        }

    def linker_flags(self, detected: Optional[Dict[str, DetectedLibrary]] = None) -> List[str]:
        """Aggregate, de-duplicated linker flags across all detected libraries."""
        detected = detected or self.detect_all()
        flags: List[str] = []
        for lib in detected.values():
            for flag in lib.linker_flags:
                if flag not in flags:
                    flags.append(flag)
        return flags

    def compiler_flags(self, detected: Optional[Dict[str, DetectedLibrary]] = None) -> List[str]:
        """Aggregate include/cflags across all detected libraries."""
        detected = detected or self.detect_all()
        flags: List[str] = []
        for lib in detected.values():
            for flag in lib.cflags:
                if flag not in flags:
                    flags.append(flag)
        return flags

    def genome_space(self, detected: Optional[Dict[str, DetectedLibrary]] = None) -> Dict[str, List[str]]:
        """Return the categorical gene values the evolutionary engine may try.

        Only combinations that are buildable on the current host are offered.
        ``"none"`` is always included so the engine can disable a library.
        """
        detected = detected or self.detect_all()
        space: Dict[str, List[str]] = {}

        blas = detected.get("blas")
        blas_values = ["none"]
        if blas and blas.found:
            blas_values.insert(0, blas.flavor)
        space["blas"] = self._dedup(blas_values)

        lapack = detected.get("lapack")
        lapack_values = ["none"]
        if lapack and lapack.found:
            lapack_values.insert(0, lapack.flavor)
        space["lapack"] = self._dedup(lapack_values)

        mpi = detected.get("mpi")
        if mpi and mpi.found:
            space["mpi_flavor"] = self._dedup([mpi.flavor, "none"])
        else:
            space["mpi_flavor"] = ["none"]

        cuda = detected.get("cuda")
        space["cuda"] = ["auto", "none"] if (cuda and cuda.found) else ["none"]

        # Drop trivial single-value genes -- they offer nothing to evolve.
        return {k: v for k, v in space.items() if len(v) > 1}

    # ------------------------------------------------------------------
    # Per-library probes
    # ------------------------------------------------------------------

    def detect_blas(self) -> DetectedLibrary:
        return self._detect_blas_like("blas", self.blas_choice, default_lib="blas")

    def detect_lapack(self) -> DetectedLibrary:
        return self._detect_blas_like("lapack", self.lapack_choice, default_lib="lapack")

    def _detect_blas_like(self, name: str, choice: str, default_lib: str) -> DetectedLibrary:
        result = DetectedLibrary(name=name)
        if choice == "none":
            return result

        # Resolve which concrete implementations to try, in priority order.
        if choice == "auto":
            candidates = ["mkl", "openblas", default_lib]
        else:
            candidates = [choice]

        for candidate in candidates:
            probed = self._probe_blas_candidate(name, candidate)
            if probed.found:
                return probed
        return result

    def _probe_blas_candidate(self, name: str, candidate: str) -> DetectedLibrary:
        result = DetectedLibrary(name=name, flavor=candidate)

        # 1) pkg-config (openblas / mkl-dynamic-lp64-iomp / lapack ...).
        pkg_names = {
            "mkl": ["mkl-dynamic-lp64-iomp", "mkl-dynamic-lp64-seq", "mkl"],
            "openblas": ["openblas"],
            "blas": ["blas"],
            "lapack": ["lapack"],
        }.get(candidate, [candidate])
        for pkg in pkg_names:
            pc = self._pkg_config(pkg)
            if pc is not None:
                result.found = True
                result.detected_via = "pkg-config"
                result.version = pc.get("version", "")
                result.cflags = pc.get("cflags", [])
                result.linker_flags = pc.get("libs", [])
                return result

        # 2) Environment-variable roots (MKLROOT, OPENBLAS_HOME, BLAS_HOME ...).
        env_roots = {
            "mkl": ["MKLROOT", "MKL_HOME"],
            "openblas": ["OPENBLAS_HOME", "OPENBLAS_ROOT"],
            "blas": ["BLAS_HOME", "BLAS_ROOT"],
            "lapack": ["LAPACK_HOME", "LAPACK_ROOT"],
        }.get(candidate, [])
        for env_var in env_roots:
            root = os.environ.get(env_var)
            if root and os.path.isdir(root):
                result.found = True
                result.detected_via = f"env:{env_var}"
                lib_dir = self._first_existing([os.path.join(root, "lib"), os.path.join(root, "lib64"), root])
                inc_dir = os.path.join(root, "include")
                if lib_dir:
                    result.lib_dirs = [lib_dir]
                    result.linker_flags = [f"-L{lib_dir}", f"-l{self._link_name(candidate)}"]
                if os.path.isdir(inc_dir):
                    result.include_dirs = [inc_dir]
                    result.cflags = [f"-I{inc_dir}"]
                return result

        # 3) Bare library on the default linker path.
        if self._find_shared_library(self._link_name(candidate)):
            result.found = True
            result.detected_via = "ldconfig"
            result.linker_flags = [f"-l{self._link_name(candidate)}"]
            return result

        return result

    def detect_mpi(self) -> DetectedLibrary:
        result = DetectedLibrary(name="mpi")
        if not self.mpi_enabled:
            return result

        flavors = [self.mpi_flavor] if self.mpi_flavor else _MPI_FLAVORS

        # Prefer the compiler wrappers -- they encode the correct flags.
        for wrapper in ("mpicc", "mpiicc", "mpicxx"):
            path = shutil.which(wrapper)
            if not path:
                continue
            link = self._mpi_wrapper_flags(wrapper, "link")
            compile_flags = self._mpi_wrapper_flags(wrapper, "compile")
            result.found = True
            result.detected_via = f"wrapper:{wrapper}"
            result.linker_flags = link
            result.cflags = compile_flags
            result.flavor = self._guess_mpi_flavor(wrapper) or (self.mpi_flavor or "openmpi")
            result.version = self._mpi_version()
            return result

        # pkg-config fallback (ompi / mpich).
        pkg_map = {"openmpi": "ompi", "mpich": "mpich"}
        for flavor in flavors:
            pkg = pkg_map.get(flavor or "", flavor or "")
            pc = self._pkg_config(pkg)
            if pc is not None:
                result.found = True
                result.detected_via = "pkg-config"
                result.flavor = flavor or "openmpi"
                result.version = pc.get("version", "")
                result.cflags = pc.get("cflags", [])
                result.linker_flags = pc.get("libs", [])
                return result

        return result

    def detect_cuda(self) -> DetectedLibrary:
        result = DetectedLibrary(name="cuda")
        if self.cuda_choice == "none":
            return result

        cuda_home = (
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or self._cuda_home_from_nvcc()
        )
        nvcc = shutil.which("nvcc")
        if not cuda_home and not nvcc:
            return result

        result.found = True
        result.flavor = "cuda"
        result.detected_via = "nvcc" if nvcc else "env:CUDA_HOME"
        result.version = self._nvcc_version() if nvcc else (
            self.cuda_choice if self.cuda_choice != "auto" else ""
        )
        if cuda_home:
            lib_dir = self._first_existing(
                [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
            )
            inc_dir = os.path.join(cuda_home, "include")
            if lib_dir:
                result.lib_dirs = [lib_dir]
                result.linker_flags = [f"-L{lib_dir}", "-lcudart"]
            if os.path.isdir(inc_dir):
                result.include_dirs = [inc_dir]
                result.cflags = [f"-I{inc_dir}"]
        if not result.linker_flags:
            result.linker_flags = ["-lcudart"]
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup(values: List[str]) -> List[str]:
        seen: List[str] = []
        for v in values:
            if v not in seen:
                seen.append(v)
        return seen

    @staticmethod
    def _link_name(candidate: str) -> str:
        return {"mkl": "mkl_rt"}.get(candidate, candidate)

    @staticmethod
    def _first_existing(paths: List[str]) -> Optional[str]:
        for p in paths:
            if p and os.path.isdir(p):
                return p
        return None

    @classmethod
    def _pkg_config(cls, package: str) -> Optional[Dict[str, Any]]:
        if not package or shutil.which("pkg-config") is None:
            return None
        if cls._run(["pkg-config", "--exists", package]) is None:
            return None
        # ``--exists`` succeeds (returncode 0) only when present.
        exists = cls._run(["pkg-config", "--exists", package], want_rc=True)
        if exists != 0:
            return None
        libs = cls._run(["pkg-config", "--libs", package]) or ""
        cflags = cls._run(["pkg-config", "--cflags", package]) or ""
        version = cls._run(["pkg-config", "--modversion", package]) or ""
        return {
            "libs": libs.split(),
            "cflags": cflags.split(),
            "version": version.strip(),
        }

    @classmethod
    def _mpi_wrapper_flags(cls, wrapper: str, mode: str) -> List[str]:
        # OpenMPI uses ``--showme:link`` / ``--showme:compile``; MPICH uses
        # ``-link_info`` / ``-compile_info``.  Try both.
        flag = "--showme:link" if mode == "link" else "--showme:compile"
        out = cls._run([wrapper, flag])
        if out is None:
            alt = "-link_info" if mode == "link" else "-compile_info"
            out = cls._run([wrapper, alt])
        if not out:
            return []
        # Keep only -L/-l/-Wl flags for link mode, -I/-D for compile mode.
        tokens = out.split()
        if mode == "link":
            return [t for t in tokens if t.startswith(("-L", "-l", "-Wl", "-pthread"))]
        return [t for t in tokens if t.startswith(("-I", "-D", "-pthread"))]

    @classmethod
    def _guess_mpi_flavor(cls, wrapper: str) -> Optional[str]:
        out = cls._run([wrapper, "--version"]) or cls._run([wrapper, "-v"]) or ""
        low = out.lower()
        if "open mpi" in low or "openmpi" in low:
            return "openmpi"
        if "mpich" in low:
            return "mpich"
        return None

    @classmethod
    def _mpi_version(cls) -> str:
        out = cls._run(["mpirun", "--version"]) or ""
        return out.splitlines()[0].strip() if out else ""

    @classmethod
    def _nvcc_version(cls) -> str:
        out = cls._run(["nvcc", "--version"]) or ""
        for line in out.splitlines():
            if "release" in line.lower():
                return line.strip()
        return ""

    @classmethod
    def _cuda_home_from_nvcc(cls) -> Optional[str]:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            return None
        # <cuda>/bin/nvcc -> <cuda>
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))

    @staticmethod
    def _find_shared_library(link_name: str) -> bool:
        # Cheap heuristic: look for lib<name>.so under common library roots.
        roots = [
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/lib",
            "/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ]
        for root in roots:
            try:
                for entry in os.listdir(root):
                    if entry.startswith(f"lib{link_name}.so"):
                        return True
            except OSError:
                continue
        return False

    @staticmethod
    def _run(cmd: List[str], want_rc: bool = False, timeout: float = 5.0):
        """Run a command, returning stdout (str) or, with ``want_rc``, the rc.

        Returns ``None`` when the command is missing or fails to launch.
        """
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                # Inherit the full parent environment so toolchain probes see the
                # same compiler paths and configuration as the real build.
                env=os.environ.copy(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if want_rc:
            return proc.returncode
        if proc.returncode != 0:
            return None
        return proc.stdout.decode("utf-8", "replace")
