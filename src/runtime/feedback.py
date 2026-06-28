"""
Runtime Feedback Loop.

After a build, runs the configured benchmark command, extracts metrics (wall
time, energy, accuracy error), optionally compares results against reference
data, and blends the metrics into the evolutionary fitness function.

Everything is optional and off by default: ``enable_feedback = false`` (or no
``benchmark_command``) makes :meth:`RuntimeFeedback.run_benchmark` a no-op that
returns an empty, unsuccessful :class:`RuntimeMetrics`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.json_parsing import extract_json
from src.utils.serialization import dataclass_to_dict

logger = logging.getLogger("runtime.feedback")

@dataclass
class RuntimeMetrics:
    success: bool = False
    wall_time: float = 0.0
    cpu_time: float = 0.0
    peak_rss_mb: float = 0.0
    energy: Optional[float] = None
    accuracy_error: Optional[float] = None
    returncode: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self, exclude=["raw"])


_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class RuntimeFeedback:
    """Runs benchmarks and folds runtime metrics into fitness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        rt = self.config.get("runtime", {}) or {}
        self.enable_feedback = bool(rt.get("enable_feedback", False))
        self.benchmark_command = rt.get("benchmark_command", "") or ""
        self.metrics_to_collect = list(rt.get("metrics_to_collect", ["wall_time"]))
        self.accuracy_reference = rt.get("accuracy_reference", "") or ""
        self.feedback_weight = float(rt.get("feedback_weight", 0.3))

    @property
    def enabled(self) -> bool:
        return self.enable_feedback and bool(self.benchmark_command)

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 600.0,
        sandbox: bool = True,
    ) -> RuntimeMetrics:
        """Run the benchmark command and collect metrics.

        With ``sandbox=True`` (and no explicit ``workdir``) the command runs in a
        fresh temporary directory so benchmarks don't interfere with each other.
        """
        if not self.enabled:
            return RuntimeMetrics(success=False, error="runtime feedback disabled")

        temp_dir: Optional[tempfile.TemporaryDirectory] = None
        if workdir is None and sandbox:
            temp_dir = tempfile.TemporaryDirectory(prefix="aero_bench_")
            run_cwd = temp_dir.name
        else:
            run_cwd = workdir or os.getcwd()

        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})

        metrics = RuntimeMetrics()
        try:
            command = shlex.split(self.benchmark_command)
            start = time.monotonic()
            cpu_before = self._cpu_times()
            try:
                proc = subprocess.run(
                    command,
                    cwd=run_cwd,
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                metrics.error = f"benchmark timed out after {timeout}s"
                return metrics
            except (OSError, ValueError) as exc:
                metrics.error = f"benchmark failed to launch: {exc}"
                return metrics

            metrics.wall_time = time.monotonic() - start
            metrics.cpu_time = max(0.0, self._cpu_times() - cpu_before)
            metrics.peak_rss_mb = self._peak_rss_mb()
            metrics.returncode = proc.returncode
            stdout = proc.stdout.decode("utf-8", "replace")
            metrics.success = proc.returncode == 0

            parsed = self.parse_metrics(stdout)
            metrics.raw = parsed
            if "wall_time" in parsed:
                metrics.wall_time = float(parsed["wall_time"])
            if "energy" in parsed:
                metrics.energy = float(parsed["energy"])
            if "accuracy" in parsed:
                # Treat a reported accuracy as an error if it looks like one.
                metrics.accuracy_error = float(parsed["accuracy"])

            if self.accuracy_reference:
                ref_error = self.compare_accuracy(stdout, run_cwd, self.accuracy_reference)
                if ref_error is not None:
                    metrics.accuracy_error = ref_error
            return metrics
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    # ------------------------------------------------------------------
    # Parsing / accuracy
    # ------------------------------------------------------------------

    def parse_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract metrics from benchmark output (JSON first, then regex)."""
        parsed: Dict[str, Any] = {}
        blob = extract_json(stdout)
        if isinstance(blob, dict):
            for key in ("wall_time", "energy", "accuracy", "accuracy_error"):
                if key in blob:
                    parsed["accuracy" if key == "accuracy_error" else key] = blob[key]
        for key in ("wall_time", "energy", "accuracy"):
            if key in parsed:
                continue
            match = re.search(rf"{key}\s*[=:]\s*({_NUMBER_RE.pattern})", stdout, re.IGNORECASE)
            if match:
                parsed[key] = float(match.group(1))
        return parsed

    def compare_accuracy(
        self, produced_text: str, run_cwd: str, reference: str
    ) -> Optional[float]:
        """Return a max relative error between produced and reference numbers."""
        ref_path = Path(reference)
        if not ref_path.is_absolute():
            # Reference may be relative to cwd or the run dir.
            for base in (Path.cwd(), Path(run_cwd)):
                candidate = base / reference
                if candidate.exists():
                    ref_path = candidate
                    break
        if not ref_path.exists():
            return None

        reference_numbers = self._collect_reference_numbers(ref_path)
        produced_numbers = [float(m) for m in _NUMBER_RE.findall(produced_text)]
        if not reference_numbers or not produced_numbers:
            return None

        n = min(len(reference_numbers), len(produced_numbers))
        max_error = 0.0
        for ref, got in zip(reference_numbers[:n], produced_numbers[:n]):
            denom = abs(ref) if abs(ref) > 1e-30 else 1.0
            max_error = max(max_error, abs(got - ref) / denom)
        return max_error

    # ------------------------------------------------------------------
    # Fitness blending
    # ------------------------------------------------------------------

    def to_fitness_objectives(self, metrics: RuntimeMetrics) -> Dict[str, float]:
        """Map collected metrics to (minimised) fitness objectives."""
        objectives: Dict[str, float] = {}
        if "wall_time" in self.metrics_to_collect:
            objectives["runtime_wall_time"] = float(metrics.wall_time)
        if "energy" in self.metrics_to_collect and metrics.energy is not None:
            objectives["runtime_energy"] = float(metrics.energy)
        if "accuracy" in self.metrics_to_collect and metrics.accuracy_error is not None:
            objectives["runtime_accuracy_error"] = float(metrics.accuracy_error)
        return objectives

    def blend_into_fitness(
        self, build_fitness: Dict[str, float], metrics: RuntimeMetrics
    ) -> Dict[str, float]:
        """Combine build-time fitness with weighted runtime objectives."""
        blended = dict(build_fitness)
        for name, value in self.to_fitness_objectives(metrics).items():
            blended[name] = value * self.feedback_weight
        return blended

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_reference_numbers(ref_path: Path) -> List[float]:
        numbers: List[float] = []
        files = [ref_path] if ref_path.is_file() else sorted(ref_path.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            numbers.extend(float(m) for m in _NUMBER_RE.findall(text))
        return numbers

    @staticmethod
    def _cpu_times() -> float:
        try:
            import psutil  # type: ignore

            proc = psutil.Process()
            ct = proc.cpu_times()
            children = proc.children(recursive=True)
            total = ct.user + ct.system
            for child in children:
                try:
                    cct = child.cpu_times()
                    total += cct.user + cct.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("psutil CPU time collection failed: %s", exc)
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            return usage.ru_utime + usage.ru_stime
        except ImportError:
            return 0.0
        except Exception as exc:
            logger.debug("resource CPU time collection failed: %s", exc)
            return 0.0

    @staticmethod
    def _peak_rss_mb() -> float:
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            # Linux reports KB, macOS reports bytes.
            return maxrss / 1024.0 if maxrss > 1_000_000 else maxrss / 1024.0
        except ImportError:
            return 0.0
        except Exception as exc:
            logger.debug("Failed to read peak RSS: %s", exc)
            return 0.0
