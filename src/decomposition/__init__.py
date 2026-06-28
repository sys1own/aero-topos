"""Physical decomposition engine for AeroNova polyglot builds."""

from src.decomposition.splitter import SplitResult, decompose_source, run_decomposition
from src.decomposition.reflux import RefluxResult, run_reflux, scan_anomalies

__all__ = [
    "SplitResult", "decompose_source", "run_decomposition",
    "RefluxResult", "run_reflux", "scan_anomalies",
]
