"""
Autonomous Hardware-Polymerization.

Inspects the host machine at runtime and polymorphically rewrites the generated
C/C++/Rust source (or LLVM IR) -- memory alignment, vectorised micro-kernels and
thread-pool sizing -- to fit that exact machine, *before* the compiler is
invoked and **with no flags from the user**.  All rewriting happens in memory or
in an ephemeral build cache, so the user's primary source directory is never
touched.

Typical use::

    from src.polymorphization import PolymorphizationEngine

    engine = PolymorphizationEngine()
    report = engine.polymerize_tree("build_artifacts/", ".aero/polymorph_cache")
    # report["topology"] describes the host; report["rewrite"] lists the edits.
"""

from src.polymorphization.engine import (
    DEFAULT_CACHE_DIR,
    PROFILE_NAME,
    REPORT_NAME,
    PolymorphizationEngine,
)
from src.polymorphization.hardware_profiler import HardwareProfiler
from src.polymorphization.rewriter import PolymorphicRewriter, RewriteResult
from src.polymorphization.topology import (
    SIMD_PRIORITY,
    SIMD_WIDTH_BYTES,
    CacheLevel,
    GpuDevice,
    HardwareTopology,
)

__all__ = [
    "PolymorphizationEngine",
    "HardwareProfiler",
    "PolymorphicRewriter",
    "RewriteResult",
    "HardwareTopology",
    "CacheLevel",
    "GpuDevice",
    "SIMD_PRIORITY",
    "SIMD_WIDTH_BYTES",
    "DEFAULT_CACHE_DIR",
    "PROFILE_NAME",
    "REPORT_NAME",
]

__version__ = "1.0.0"
