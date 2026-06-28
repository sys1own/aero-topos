"""
Hardware topology data model for Autonomous Hardware-Polymerization.

Plain, dependency-free dataclasses describing what the :class:`HardwareProfiler`
discovered about the host: CPU vector features, the physical/logical core split,
the L1/L2/L3 cache hierarchy (with line sizes), available GPU architectures, and
a coarse memory-bandwidth class.  Everything is JSON-serialisable so a profile
can be cached in the ephemeral build cache and reused without re-probing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ordered best-to-worst so the rewriter can pick the strongest available target.
SIMD_PRIORITY = ("avx512", "avx2", "avx", "sse4_2", "sse2", "neon", "scalar")

# Width in bytes of one vector register for each ISA -- drives micro-kernel and
# alignment selection.
SIMD_WIDTH_BYTES = {
    "avx512": 64,
    "avx2": 32,
    "avx": 32,
    "sse4_2": 16,
    "sse2": 16,
    "neon": 16,
    "scalar": 8,
}


@dataclass
class CacheLevel:
    """One level of the CPU data-cache hierarchy."""

    level: int  # 1, 2 or 3
    size_bytes: int
    line_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "size_bytes": self.size_bytes, "line_size_bytes": self.line_size_bytes}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheLevel":
        return cls(
            level=int(data["level"]),
            size_bytes=int(data["size_bytes"]),
            line_size_bytes=int(data["line_size_bytes"]),
        )


@dataclass
class GpuDevice:
    """A discovered GPU and its compute architecture."""

    runtime: str  # "cuda" | "vulkan" | "webgpu"
    name: str
    architecture: str = "unknown"  # e.g. "sm_86", "gfx1030", "apple-m"

    def to_dict(self) -> Dict[str, Any]:
        return {"runtime": self.runtime, "name": self.name, "architecture": self.architecture}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GpuDevice":
        return cls(
            runtime=str(data["runtime"]),
            name=str(data["name"]),
            architecture=str(data.get("architecture", "unknown")),
        )


@dataclass
class HardwareTopology:
    """The full probed picture of the host machine."""

    arch: str = ""  # platform.machine(): "x86_64", "aarch64", ...
    physical_cores: int = 1
    logical_cores: int = 1
    cpu_features: List[str] = field(default_factory=list)  # raw ISA flags of interest
    cache_levels: List[CacheLevel] = field(default_factory=list)
    gpus: List[GpuDevice] = field(default_factory=list)
    total_memory_bytes: int = 0
    memory_bandwidth_class: str = "unknown"  # "low" | "medium" | "high"

    # ------------------------------------------------------------------
    # Derived helpers used by the rewriter
    # ------------------------------------------------------------------

    def best_simd(self) -> str:
        """Return the strongest available vector ISA, or ``"scalar"``."""
        features = set(self.cpu_features)
        for isa in SIMD_PRIORITY:
            if isa == "scalar":
                continue
            if isa in features:
                return isa
        return "scalar"

    def vector_width_bytes(self) -> int:
        return SIMD_WIDTH_BYTES.get(self.best_simd(), 8)

    def cache_line_bytes(self) -> int:
        """L1 line size if known, else the conventional 64 bytes."""
        for cache in self.cache_levels:
            if cache.level == 1 and cache.line_size_bytes > 0:
                return cache.line_size_bytes
        for cache in self.cache_levels:
            if cache.line_size_bytes > 0:
                return cache.line_size_bytes
        return 64

    def alignment_bytes(self) -> int:
        """Recommended allocation alignment: the larger of the cache line and
        the vector width, so allocations are both cache- and SIMD-friendly.
        """
        return max(self.cache_line_bytes(), self.vector_width_bytes())

    def has_gpu(self) -> bool:
        return bool(self.gpus)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "cpu_features": list(self.cpu_features),
            "cache_levels": [c.to_dict() for c in self.cache_levels],
            "gpus": [g.to_dict() for g in self.gpus],
            "total_memory_bytes": self.total_memory_bytes,
            "memory_bandwidth_class": self.memory_bandwidth_class,
            "derived": {
                "best_simd": self.best_simd(),
                "vector_width_bytes": self.vector_width_bytes(),
                "cache_line_bytes": self.cache_line_bytes(),
                "alignment_bytes": self.alignment_bytes(),
                "has_gpu": self.has_gpu(),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareTopology":
        return cls(
            arch=str(data.get("arch", "")),
            physical_cores=int(data.get("physical_cores", 1)),
            logical_cores=int(data.get("logical_cores", 1)),
            cpu_features=list(data.get("cpu_features", [])),
            cache_levels=[CacheLevel.from_dict(c) for c in data.get("cache_levels", [])],
            gpus=[GpuDevice.from_dict(g) for g in data.get("gpus", [])],
            total_memory_bytes=int(data.get("total_memory_bytes", 0)),
            memory_bandwidth_class=str(data.get("memory_bandwidth_class", "unknown")),
        )
