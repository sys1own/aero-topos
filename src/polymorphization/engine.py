"""
``PolymorphizationEngine`` -- ties the profiler and rewriter together.

This is the single entry point the build pipeline calls to perform Autonomous
Hardware-Polymerization: probe the host once, then rewrite the freshly generated
sources into an ephemeral build cache, leaving the primary source tree
untouched.  It is deliberately side-effect-light and never raises into the
build -- a failure to polymerise degrades to "use the original sources".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.polymorphization.hardware_profiler import HardwareProfiler
from src.polymorphization.rewriter import PolymorphicRewriter
from src.polymorphization.topology import HardwareTopology

# Default ephemeral cache location (under the workspace's .aero scratch dir).
DEFAULT_CACHE_DIR = Path(".aero") / "polymorph_cache"
PROFILE_NAME = "hardware_topology.json"
REPORT_NAME = "polymorphization_report.json"


class PolymorphizationEngine:
    """Profile the host and polymorphically rewrite generated code for it."""

    def __init__(self, *, profiler: Optional[HardwareProfiler] = None) -> None:
        self.profiler = profiler or HardwareProfiler()
        self.last_topology: Optional[HardwareTopology] = None
        self.last_report: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile_host(self) -> HardwareTopology:
        self.last_topology = self.profiler.probe()
        return self.last_topology

    # ------------------------------------------------------------------
    # Polymerization
    # ------------------------------------------------------------------

    def polymerize_tree(
        self,
        source_dir: Path,
        cache_dir: Optional[Path] = None,
        topology: Optional[HardwareTopology] = None,
    ) -> Dict[str, Any]:
        """Probe (if needed) and rewrite ``source_dir`` into the ephemeral cache.

        Returns a JSON-serialisable report describing the host topology and the
        rewrites that were applied.  The source directory is never modified.
        """
        topology = topology or self.last_topology or self.profile_host()
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        rewriter = PolymorphicRewriter(topology)
        # Start from a clean cache so stale rewrites never leak into a build.
        rewriter.reset_cache(cache_dir)
        rewrite_report = rewriter.rewrite_tree(Path(source_dir), cache_dir)

        report: Dict[str, Any] = {
            "source_dir": str(source_dir),
            "topology": topology.to_dict(),
            "rewrite": rewrite_report,
        }
        self.last_report = report
        return report

    def polymerize_text(self, source: str, language: str, topology: Optional[HardwareTopology] = None) -> str:
        """In-memory convenience: rewrite a single source string for the host."""
        topology = topology or self.last_topology or self.profile_host()
        return PolymorphicRewriter(topology).rewrite_text(source, language).text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_report(self, report: Dict[str, Any], output_dir: Path) -> Path:
        """Persist the topology + rewrite report into the ephemeral cache dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / PROFILE_NAME).write_text(
            json.dumps(report["topology"], indent=2), encoding="utf-8"
        )
        report_path = output_dir / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report_path
