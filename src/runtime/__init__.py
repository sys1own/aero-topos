"""Runtime feedback loop.

Runs a benchmark after a build, extracts metrics (wall time, energy, accuracy),
and blends them back into the evolutionary fitness so candidates are tuned for
real workloads, not just build-time statistics.  Disabled by default.
"""

from src.runtime.feedback import RuntimeFeedback, RuntimeMetrics

__all__ = ["RuntimeFeedback", "RuntimeMetrics"]
