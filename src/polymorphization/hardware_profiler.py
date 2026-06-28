"""
``HardwareProfiler`` -- runtime, flag-free probing of the host system topology.

This is the discovery half of Autonomous Hardware-Polymerization (requirement
#1).  Unlike the benchmark-driven
:class:`src.hardware_profiling.profiler.HardwareProfiler` (which actively times
micro-benchmarks and depends on numpy), this profiler is **pure stdlib** and
purely introspective: it reads ``/proc``/``sysfs`` and shells out to detection
tools that are already present, degrading gracefully to sane defaults when a
source of truth is unavailable.  That keeps it safe to run automatically at the
start of every build, with no flags and no added dependency.

It discovers:

* CPU vector features (AVX2, AVX-512, SSE, ARM NEON),
* the physical-vs-logical core split (for thread-pool sizing),
* the L1/L2/L3 cache hierarchy and line sizes,
* available GPU architectures (CUDA, then Vulkan / WebGPU),
* total memory and a coarse memory-bandwidth class.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.polymorphization.topology import CacheLevel, GpuDevice, HardwareTopology

# Map raw CPUID/cpuinfo flag spellings onto the normalised ISA names the
# topology model and rewriter understand.
_FEATURE_ALIASES = {
    "avx512f": "avx512",
    "avx512": "avx512",
    "avx2": "avx2",
    "avx": "avx",
    "sse4_2": "sse4_2",
    "sse4.2": "sse4_2",
    "sse2": "sse2",
    "neon": "neon",
    "asimd": "neon",  # ARM "Advanced SIMD" == NEON
}


class HardwareProfiler:
    """Introspects the host and returns a :class:`HardwareTopology`."""

    def __init__(self, *, allow_subprocess: bool = True) -> None:
        # ``allow_subprocess`` lets callers (and tests) disable shelling out to
        # nvidia-smi / vulkaninfo entirely for determinism.
        self.allow_subprocess = allow_subprocess

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self) -> HardwareTopology:
        topology = HardwareTopology(arch=platform.machine())
        self._probe_cores(topology)
        self._probe_cpu_features(topology)
        self._probe_cache_hierarchy(topology)
        self._probe_memory(topology)
        self._probe_gpus(topology)
        return topology

    # ------------------------------------------------------------------
    # Cores (physical vs logical)
    # ------------------------------------------------------------------

    def _probe_cores(self, topology: HardwareTopology) -> None:
        logical = os.cpu_count() or 1
        topology.logical_cores = logical
        topology.physical_cores = self._physical_core_count(logical)

    def _physical_core_count(self, logical: int) -> int:
        # Linux: count distinct (physical id, core id) pairs in /proc/cpuinfo.
        cpuinfo = self._read_text("/proc/cpuinfo")
        if cpuinfo:
            cores: Set[tuple] = set()
            physical_id: Optional[str] = None
            core_id: Optional[str] = None
            for line in cpuinfo.splitlines():
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                elif not line.strip():  # processor block boundary
                    if physical_id is not None and core_id is not None:
                        cores.add((physical_id, core_id))
                    physical_id = core_id = None
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            if cores:
                return len(cores)

        # macOS / BSD: sysctl.
        physical = self._sysctl_int("hw.physicalcpu")
        if physical:
            return physical

        # Fallback: assume SMT-2 if logical is even and > 1, else logical.
        if logical > 1 and logical % 2 == 0:
            return logical // 2
        return logical

    # ------------------------------------------------------------------
    # CPU vector features
    # ------------------------------------------------------------------

    def _probe_cpu_features(self, topology: HardwareTopology) -> None:
        raw_flags = self._raw_cpu_flags()
        features: List[str] = []
        seen: Set[str] = set()
        for flag in raw_flags:
            normalised = _FEATURE_ALIASES.get(flag)
            if normalised and normalised not in seen:
                seen.add(normalised)
                features.append(normalised)
        # ARM hosts that expose no flag file still have NEON on aarch64.
        if not features and topology.arch in ("aarch64", "arm64"):
            features.append("neon")
        topology.cpu_features = features

    def _raw_cpu_flags(self) -> Set[str]:
        flags: Set[str] = set()
        cpuinfo = self._read_text("/proc/cpuinfo")
        if cpuinfo:
            for line in cpuinfo.splitlines():
                # x86 uses "flags", ARM uses "Features".
                if line.startswith("flags") or line.startswith("Features"):
                    _, _, value = line.partition(":")
                    flags.update(value.strip().lower().split())
        if flags:
            return flags

        # macOS: sysctl exposes vector features via machdep.cpu.features /
        # leaf7_features.
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            value = self._sysctl_str(key)
            if value:
                flags.update(value.lower().replace(".", "_").split())
        return flags

    # ------------------------------------------------------------------
    # Cache hierarchy
    # ------------------------------------------------------------------

    def _probe_cache_hierarchy(self, topology: HardwareTopology) -> None:
        levels = self._sysfs_cache_levels()
        if not levels:
            levels = self._default_cache_levels()
        topology.cache_levels = levels

    def _sysfs_cache_levels(self) -> List[CacheLevel]:
        base = Path("/sys/devices/system/cpu/cpu0/cache")
        if not base.exists():
            return []
        by_level: Dict[int, CacheLevel] = {}
        for index_dir in sorted(base.glob("index*")):
            try:
                level = int(self._read_text(index_dir / "level").strip())
                cache_type = (self._read_text(index_dir / "type") or "").strip().lower()
                # Skip pure instruction caches; we care about data/unified.
                if cache_type == "instruction":
                    continue
                line_size = int((self._read_text(index_dir / "coherency_line_size") or "0").strip() or 0)
                size = self._parse_size(self._read_text(index_dir / "size"))
            except (ValueError, AttributeError):
                continue
            # Keep the first (largest data/unified) cache seen per level.
            if level not in by_level:
                by_level[level] = CacheLevel(level=level, size_bytes=size, line_size_bytes=line_size)
        return [by_level[level] for level in sorted(by_level)]

    @staticmethod
    def _default_cache_levels() -> List[CacheLevel]:
        # Conservative, widely-correct defaults (64-byte lines on x86/most ARM).
        return [
            CacheLevel(level=1, size_bytes=32 * 1024, line_size_bytes=64),
            CacheLevel(level=2, size_bytes=256 * 1024, line_size_bytes=64),
            CacheLevel(level=3, size_bytes=8 * 1024 * 1024, line_size_bytes=64),
        ]

    @staticmethod
    def _parse_size(raw: Optional[str]) -> int:
        if not raw:
            return 0
        raw = raw.strip().upper()
        match = re.match(r"(\d+)([KMG]?)", raw)
        if not match:
            return 0
        value = int(match.group(1))
        unit = match.group(2)
        return value * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _probe_memory(self, topology: HardwareTopology) -> None:
        total = 0
        meminfo = self._read_text("/proc/meminfo")
        if meminfo:
            for line in meminfo.splitlines():
                if line.startswith("MemTotal"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total = int(parts[1]) * 1024
                    break
        if not total:
            total = (self._sysctl_int("hw.memsize") or 0)
        topology.total_memory_bytes = total
        topology.memory_bandwidth_class = self._bandwidth_class(total, topology)

    @staticmethod
    def _bandwidth_class(total_bytes: int, topology: HardwareTopology) -> str:
        # A coarse, static heuristic: more cores + wider vectors + more RAM
        # generally track higher achievable bandwidth.  This is a *constraint
        # hint* for the rewriter (e.g. blocking aggressiveness), not a measured
        # GB/s, keeping the profiler benchmark-free.
        gib = total_bytes / (1024**3) if total_bytes else 0
        score = 0
        if topology.physical_cores >= 8:
            score += 1
        if topology.best_simd() in ("avx2", "avx512"):
            score += 1
        if gib >= 32:
            score += 1
        return ("low", "medium", "high", "high")[min(score, 3)]

    # ------------------------------------------------------------------
    # GPUs
    # ------------------------------------------------------------------

    def _probe_gpus(self, topology: HardwareTopology) -> None:
        gpus: List[GpuDevice] = []
        gpus.extend(self._probe_cuda())
        if not gpus:
            gpus.extend(self._probe_vulkan())
        topology.gpus = gpus

    def _probe_cuda(self) -> List[GpuDevice]:
        if not self.allow_subprocess or not shutil.which("nvidia-smi"):
            return []
        output = self._run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
        if not output:
            return []
        gpus: List[GpuDevice] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if parts else "NVIDIA GPU"
            arch = "unknown"
            if len(parts) >= 2 and parts[1]:
                # compute_cap "8.6" -> sm_86
                arch = "sm_" + parts[1].replace(".", "")
            gpus.append(GpuDevice(runtime="cuda", name=name, architecture=arch))
        return gpus

    def _probe_vulkan(self) -> List[GpuDevice]:
        if not self.allow_subprocess or not shutil.which("vulkaninfo"):
            return []
        output = self._run(["vulkaninfo", "--summary"])
        if not output:
            return []
        gpus: List[GpuDevice] = []
        for match in re.finditer(r"deviceName\s*=\s*(.+)", output):
            gpus.append(GpuDevice(runtime="vulkan", name=match.group(1).strip()))
        return gpus

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text(path) -> Optional[str]:
        try:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            return None

    def _run(self, cmd: List[str]) -> Optional[str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            return result.stdout if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def _sysctl_str(self, key: str) -> Optional[str]:
        if not self.allow_subprocess or not shutil.which("sysctl"):
            return None
        out = self._run(["sysctl", "-n", key])
        return out.strip() if out else None

    def _sysctl_int(self, key: str) -> Optional[int]:
        value = self._sysctl_str(key)
        if value and value.isdigit():
            return int(value)
        return None
