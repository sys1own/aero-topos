"""
GPU Offloading Pipeline.

Compiles GPU kernels (CUDA ``.cu`` via ``nvcc``, HIP via ``hipcc``) into object
files and produces a link plan that joins them with the main binary.  OpenCL
kernels are typically compiled at runtime, so for that backend the pipeline
validates and records the sources instead of pre-compiling them.

Like the rest of the build extensions, the pipeline is fail-safe: if the GPU
toolchain is not installed it returns a structured "skipped" result rather than
raising, so a project that *declares* GPU kernels still builds (CPU-only) on a
machine without a GPU SDK.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.serialization import dataclass_to_dict


# backend -> (compiler executable, object-file extension)
_BACKEND_COMPILERS = {
    "cuda": ("nvcc", ".o"),
    "hip": ("hipcc", ".o"),
    "opencl": (None, None),  # runtime-compiled
}


@dataclass
class KernelCompileResult:
    source: str
    output: str
    backend: str
    status: str  # "compiled" | "skipped" | "failed" | "planned"
    command: List[str] = field(default_factory=list)
    returncode: int = 0
    stderr: str = ""
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(
            self, exclude=["stderr"], round_keys={"duration": 6}
        )


class GPUPipeline:
    """Plans and (optionally) executes GPU kernel compilation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        gpu_cfg = self.config.get("gpu", {}) or {}
        self.enabled = bool(gpu_cfg.get("enabled", False))
        self.backend = str(gpu_cfg.get("backend", "cuda")).lower()
        self.kernel_patterns = list(gpu_cfg.get("kernel_sources", []))
        compiler, obj_ext = _BACKEND_COMPILERS.get(self.backend, (None, None))
        self.compiler_name = compiler
        self.object_extension = obj_ext or ".o"

    # ------------------------------------------------------------------
    # Discovery / availability
    # ------------------------------------------------------------------

    def compiler_path(self) -> Optional[str]:
        if not self.compiler_name:
            return None
        return shutil.which(self.compiler_name)

    def available(self) -> bool:
        """Whether kernels can actually be compiled on this host."""
        if not self.enabled:
            return False
        if self.backend == "opencl":
            return True  # runtime compilation, no offline toolchain needed
        return self.compiler_path() is not None

    def discover_kernels(self, project_root: Path) -> List[Path]:
        kernels: List[Path] = []
        for pattern in self.kernel_patterns:
            for path in sorted(project_root.glob(pattern)):
                if path.is_file() and path not in kernels:
                    kernels.append(path)
        return kernels

    # ------------------------------------------------------------------
    # Planning (no side effects) and compilation
    # ------------------------------------------------------------------

    def _compile_command(
        self, source: Path, output: Path, extra_flags: Optional[List[str]] = None
    ) -> List[str]:
        flags = list(extra_flags or [])
        compiler = self.compiler_name or "nvcc"
        if self.backend in ("cuda", "hip"):
            return [compiler, "-c", str(source), "-o", str(output), *flags]
        # OpenCL: no offline object; emit a no-op validation command.
        return [compiler or "cc", "-fsyntax-only", str(source), *flags]

    def plan(
        self,
        project_root: Path,
        output_dir: str = "build/gpu",
        extra_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Describe what *would* be compiled and how it links, without running."""
        kernels = self.discover_kernels(project_root)
        out_dir = project_root / output_dir
        objects: List[str] = []
        steps: List[Dict[str, Any]] = []
        for kernel in kernels:
            obj = out_dir / (kernel.stem + self.object_extension)
            objects.append(str(obj))
            steps.append(
                {
                    "source": str(kernel),
                    "output": str(obj),
                    "command": self._compile_command(kernel, obj, extra_flags),
                }
            )
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "compiler": self.compiler_name,
            "available": self.available(),
            "kernel_count": len(kernels),
            "compile_steps": steps,
            "link_objects": objects,
            "link_flags": self.link_flags(),
        }

    def link_flags(self) -> List[str]:
        """Linker flags needed to join GPU objects with the main binary."""
        if self.backend == "cuda":
            return ["-lcudart"]
        if self.backend == "hip":
            return ["-lamdhip64"]
        if self.backend == "opencl":
            return ["-lOpenCL"]
        return []

    def compile_kernels(
        self,
        project_root: Path,
        output_dir: str = "build/gpu",
        extra_flags: Optional[List[str]] = None,
    ) -> List[KernelCompileResult]:
        """Compile every discovered kernel, skipping gracefully if no toolchain."""
        results: List[KernelCompileResult] = []
        kernels = self.discover_kernels(project_root)
        out_dir = project_root / output_dir

        if not self.enabled:
            return [
                KernelCompileResult(
                    source=str(k), output="", backend=self.backend, status="skipped",
                    stderr="gpu offloading disabled",
                )
                for k in kernels
            ]

        toolchain_ok = self.available()
        if toolchain_ok:
            out_dir.mkdir(parents=True, exist_ok=True)

        for kernel in kernels:
            obj = out_dir / (kernel.stem + self.object_extension)
            command = self._compile_command(kernel, obj, extra_flags)
            if not toolchain_ok:
                results.append(
                    KernelCompileResult(
                        source=str(kernel), output=str(obj), backend=self.backend,
                        status="skipped", command=command,
                        stderr=f"{self.compiler_name or self.backend} not available",
                    )
                )
                continue
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    # Inherit the full parent environment so custom compiler
                    # paths and build limits are never dropped.
                    env=os.environ.copy(),
                )
                results.append(
                    KernelCompileResult(
                        source=str(kernel),
                        output=str(obj),
                        backend=self.backend,
                        status="compiled" if proc.returncode == 0 else "failed",
                        command=command,
                        returncode=proc.returncode,
                        stderr=proc.stderr.decode("utf-8", "replace"),
                        duration=time.monotonic() - start,
                    )
                )
            except (OSError, subprocess.SubprocessError) as exc:
                results.append(
                    KernelCompileResult(
                        source=str(kernel), output=str(obj), backend=self.backend,
                        status="failed", command=command, stderr=str(exc),
                        duration=time.monotonic() - start,
                    )
                )
        return results
