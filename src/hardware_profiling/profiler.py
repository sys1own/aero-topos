"""
Hardware Profiling Engine.

Probes the host hardware at compile time using active micro-benchmarks to
build a hardware profile.  The profile drives the dynamic recipe generator
which adjusts loop unrolling, cache blocking, and vectorisation hints.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hardware_profiling.profiler")


@dataclass
class CacheLevel:
    level: int
    size_bytes: int
    line_size_bytes: int
    latency_ns: float


@dataclass
class SIMDCapability:
    instruction_set: str
    available: bool
    throughput_gflops: float = 0.0


@dataclass
class MemoryBandwidth:
    mode: str
    bandwidth_gbps: float
    block_size: int


@dataclass
class InstructionLatency:
    operation: str
    precision: str
    latency_ns: float


@dataclass
class HardwareProfile:
    """Aggregated hardware profile for the current host."""

    cpu_model: str = ""
    cpu_count: int = 1
    cpu_frequency_mhz: float = 0.0
    arch: str = ""
    total_memory_bytes: int = 0
    cache_hierarchy: List[CacheLevel] = field(default_factory=list)
    simd_capabilities: List[SIMDCapability] = field(default_factory=list)
    memory_bandwidths: List[MemoryBandwidth] = field(default_factory=list)
    instruction_latencies: List[InstructionLatency] = field(default_factory=list)
    libraries: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "cpu_count": self.cpu_count,
            "cpu_frequency_mhz": self.cpu_frequency_mhz,
            "arch": self.arch,
            "total_memory_bytes": self.total_memory_bytes,
            "libraries": self.libraries,
            "cache_hierarchy": [
                {
                    "level": c.level,
                    "size_bytes": c.size_bytes,
                    "line_size_bytes": c.line_size_bytes,
                    "latency_ns": c.latency_ns,
                }
                for c in self.cache_hierarchy
            ],
            "simd_capabilities": [
                {
                    "instruction_set": s.instruction_set,
                    "available": s.available,
                    "throughput_gflops": s.throughput_gflops,
                }
                for s in self.simd_capabilities
            ],
            "memory_bandwidths": [
                {
                    "mode": m.mode,
                    "bandwidth_gbps": m.bandwidth_gbps,
                    "block_size": m.block_size,
                }
                for m in self.memory_bandwidths
            ],
            "instruction_latencies": [
                {
                    "operation": il.operation,
                    "precision": il.precision,
                    "latency_ns": il.latency_ns,
                }
                for il in self.instruction_latencies
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareProfile:
        profile = cls(
            cpu_model=data.get("cpu_model", ""),
            cpu_count=data.get("cpu_count", 1),
            cpu_frequency_mhz=data.get("cpu_frequency_mhz", 0.0),
            arch=data.get("arch", ""),
            total_memory_bytes=data.get("total_memory_bytes", 0),
            libraries=data.get("libraries", {}),
        )
        for c in data.get("cache_hierarchy", []):
            profile.cache_hierarchy.append(
                CacheLevel(
                    level=c["level"],
                    size_bytes=c["size_bytes"],
                    line_size_bytes=c["line_size_bytes"],
                    latency_ns=c["latency_ns"],
                )
            )
        for s in data.get("simd_capabilities", []):
            profile.simd_capabilities.append(
                SIMDCapability(
                    instruction_set=s["instruction_set"],
                    available=s["available"],
                    throughput_gflops=s.get("throughput_gflops", 0.0),
                )
            )
        for m in data.get("memory_bandwidths", []):
            profile.memory_bandwidths.append(
                MemoryBandwidth(
                    mode=m["mode"],
                    bandwidth_gbps=m["bandwidth_gbps"],
                    block_size=m["block_size"],
                )
            )
        for il in data.get("instruction_latencies", []):
            profile.instruction_latencies.append(
                InstructionLatency(
                    operation=il["operation"],
                    precision=il["precision"],
                    latency_ns=il["latency_ns"],
                )
            )
        return profile


class HardwareProfiler:
    """
    Probes the host hardware and generates optimisation recipes.

    Benchmarks: cache hierarchy latency, SIMD availability,
    memory bandwidth, and instruction latencies.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        hw_cfg = config.get("hardware_profiling", {})
        self.benchmarks = hw_cfg.get("benchmarks", {})
        self.recipe_cfg = hw_cfg.get("dynamic_recipe_generator", {})
        self.storage_path = Path(hw_cfg.get("profile_storage", ".aero/hardware_profiles"))
        self.retention_days = int(hw_cfg.get("profile_retention_days", 30))
        # Numerical-library probing is enabled whenever a [libraries] section is
        # present (feature #2).  Absent section -> no probing, legacy behaviour.
        self.probe_libraries = bool(config.get("libraries"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self) -> HardwareProfile:
        profile = HardwareProfile(
            cpu_model=platform.processor() or platform.machine(),
            cpu_count=os.cpu_count() or 1,
            arch=platform.machine(),
        )
        self._probe_cpu_frequency(profile)
        self._probe_memory(profile)
        self._probe_cache_hierarchy(profile)
        self._probe_simd_capabilities(profile)
        self._probe_memory_bandwidth(profile)
        self._probe_instruction_latency(profile)
        self._probe_libraries(profile)
        return profile

    def _probe_libraries(self, profile: HardwareProfile) -> None:
        """Probe BLAS/LAPACK/MPI/CUDA and record detection + linker flags."""
        if not self.probe_libraries:
            return
        try:
            from src.build.library_tuner import LibraryTuner

            tuner = LibraryTuner(self.config)
            detected = tuner.detect_all()
            profile.libraries = {
                "detected": {name: lib.to_dict() for name, lib in detected.items()},
                "linker_flags": tuner.linker_flags(detected),
                "compiler_flags": tuner.compiler_flags(detected),
            }
        except ImportError:
            logger.debug("LibraryTuner not available, skipping library detection")
            profile.libraries = {}
        except Exception as exc:
            logger.warning("Library detection failed: %s", exc)
            profile.libraries = {}

    def save_profile(self, profile: HardwareProfile) -> Path:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / "current_profile.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2))
        return path

    def load_profile(self) -> Optional[HardwareProfile]:
        path = self.storage_path / "current_profile.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return HardwareProfile.from_dict(data)

    def generate_recipe(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Generate a dynamic build recipe based on the hardware profile."""
        recipe: Dict[str, Any] = {
            "target_cpu": profile.cpu_model,
            "parallelism": min(profile.cpu_count, 16),
        }

        # Unroll factor based on cache line size
        if profile.cache_hierarchy:
            l1 = profile.cache_hierarchy[0]
            unroll_min = self.recipe_cfg.get("unroll_factor_limits", {}).get("min", 2)
            unroll_max = self.recipe_cfg.get("unroll_factor_limits", {}).get("max", 16)
            optimal_unroll = max(unroll_min, min(unroll_max, l1.line_size_bytes // 8))
            recipe["unroll_factor"] = optimal_unroll
        else:
            recipe["unroll_factor"] = 4

        # Vectorisation hints
        avail_simd = [s for s in profile.simd_capabilities if s.available]
        if avail_simd:
            best = max(avail_simd, key=lambda s: s.throughput_gflops)
            recipe["vectorization_target"] = best.instruction_set
            recipe["vectorization_enabled"] = True
        else:
            recipe["vectorization_enabled"] = False

        # Cache blocking
        if profile.cache_hierarchy and self.recipe_cfg.get("cache_blocking_strategy") == "multi_level":
            recipe["cache_block_sizes"] = [
                c.size_bytes // 4 for c in profile.cache_hierarchy
            ]

        # Polyhedral tiling
        recipe["polyhedral_tiling"] = self.recipe_cfg.get("polyhedral_tiling_adjustment", False)

        # Numerical-library linker/compiler flags (feature #2).
        if profile.libraries:
            recipe["linker_flags"] = profile.libraries.get("linker_flags", [])
            recipe["library_compiler_flags"] = profile.libraries.get("compiler_flags", [])
            recipe["detected_libraries"] = {
                name: info.get("found", False)
                for name, info in profile.libraries.get("detected", {}).items()
            }

        return recipe

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_cpu_frequency(profile: HardwareProfile) -> None:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "cpu MHz" in line:
                        profile.cpu_frequency_mhz = float(line.split(":")[1].strip())
                        return
        except OSError:
            pass
        except (ValueError, IndexError) as exc:
            logger.debug("Failed to parse CPU frequency from /proc/cpuinfo: %s", exc)

    @staticmethod
    def _probe_memory(profile: HardwareProfile) -> None:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        profile.total_memory_bytes = kb * 1024
                        return
        except OSError:
            pass
        except (ValueError, IndexError) as exc:
            logger.debug("Failed to parse memory info from /proc/meminfo: %s", exc)

    def _probe_cache_hierarchy(self, profile: HardwareProfile) -> None:
        cfg = self.benchmarks.get("cache_hierarchy", {})
        stride_sizes = cfg.get("stride_sizes", [64])
        iterations = cfg.get("iterations", 100)

        for level, stride in enumerate(stride_sizes[:3], start=1):
            buf_size = stride * 1024
            buf = np.zeros(buf_size // 8, dtype=np.float64)
            start = time.perf_counter_ns()
            idx = 0
            step = stride // 8 or 1
            for _ in range(iterations):
                idx = (idx + step) % len(buf)
                buf[idx] += 1.0
            elapsed = time.perf_counter_ns() - start
            latency_ns = elapsed / max(iterations, 1)
            profile.cache_hierarchy.append(
                CacheLevel(
                    level=level,
                    size_bytes=buf_size,
                    line_size_bytes=stride,
                    latency_ns=latency_ns,
                )
            )

    def _probe_simd_capabilities(self, profile: HardwareProfile) -> None:
        cfg = self.benchmarks.get("simd_throughput", {})
        test_vectors = cfg.get("test_vectors", ["sse4.2", "avx2"])

        cpuinfo_flags = set()
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        cpuinfo_flags = set(line.split(":")[1].lower().split())
                        break
        except OSError:
            pass
        except (ValueError, IndexError) as exc:
            logger.debug("Failed to parse SIMD flags from /proc/cpuinfo: %s", exc)

        for isa in test_vectors:
            available = isa.lower().replace(".", "_").replace("-", "_") in " ".join(cpuinfo_flags) or isa.lower() in " ".join(cpuinfo_flags)
            throughput = 0.0
            if available:
                size = 1024
                a = np.random.randn(size).astype(np.float32)
                b = np.random.randn(size).astype(np.float32)
                start = time.perf_counter_ns()
                for _ in range(100):
                    np.dot(a, b)
                elapsed_ns = time.perf_counter_ns() - start
                ops = 2 * size * 100
                throughput = ops / (elapsed_ns * 1e-9) / 1e9

            profile.simd_capabilities.append(
                SIMDCapability(
                    instruction_set=isa,
                    available=available,
                    throughput_gflops=round(throughput, 2),
                )
            )

    def _probe_memory_bandwidth(self, profile: HardwareProfile) -> None:
        cfg = self.benchmarks.get("memory_bandwidth", {})
        modes = cfg.get("allocation_modes", ["sequential"])
        block_sizes = cfg.get("block_sizes", [4096])

        for mode in modes:
            for block_size in block_sizes[:2]:
                buf = np.zeros(block_size * 256, dtype=np.float64)
                start = time.perf_counter_ns()
                if mode == "sequential":
                    buf += 1.0
                elif mode == "random_page":
                    indices = np.random.randint(0, len(buf), size=len(buf) // 4)
                    buf[indices] += 1.0
                else:
                    buf[::block_size] += 1.0
                elapsed_ns = max(1, time.perf_counter_ns() - start)
                bytes_moved = buf.nbytes
                bandwidth_gbps = (bytes_moved / elapsed_ns)

                profile.memory_bandwidths.append(
                    MemoryBandwidth(
                        mode=mode,
                        bandwidth_gbps=round(bandwidth_gbps, 4),
                        block_size=block_size,
                    )
                )

    def _probe_instruction_latency(self, profile: HardwareProfile) -> None:
        cfg = self.benchmarks.get("instruction_latency", {})
        operations = cfg.get("operations", ["add", "mul"])
        precisions = cfg.get("precision", ["f64"])

        op_map = {
            "add": lambda a, b: a + b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0)),
            "sqrt": lambda a, _: np.sqrt(np.abs(a)),
            "sin": lambda a, _: np.sin(a),
            "cos": lambda a, _: np.cos(a),
        }

        for op_name in operations:
            fn = op_map.get(op_name)
            if fn is None:
                continue
            for prec in precisions:
                dtype = np.float32 if "32" in prec else np.float64 if "64" in prec else np.int32
                a = np.ones(1024, dtype=dtype) * 1.5
                b = np.ones(1024, dtype=dtype) * 2.5

                start = time.perf_counter_ns()
                for _ in range(100):
                    fn(a, b)
                elapsed = time.perf_counter_ns() - start
                latency_ns = elapsed / (100 * 1024)

                profile.instruction_latencies.append(
                    InstructionLatency(
                        operation=op_name,
                        precision=prec,
                        latency_ns=round(latency_ns, 3),
                    )
                )
