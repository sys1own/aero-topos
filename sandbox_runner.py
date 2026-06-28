from __future__ import annotations

import importlib
import inspect
import traceback
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


def _microseconds_from_ns(duration_ns: int) -> float:
    """Convert a nanosecond duration into microseconds."""
    return duration_ns / 1000.0


# ---------------------------------------------------------------------------
# Callable resolution
# ---------------------------------------------------------------------------


def _resolve_callable(module: Any, callable_name: Optional[str]) -> Optional[Callable[..., Any]]:
    """Locate the callable to benchmark inside an imported module.

    If *callable_name* is given it is used directly; otherwise the first
    public, module-defined function is chosen so a freshly translated variant
    can still be exercised without an explicit entrypoint.
    """
    if callable_name:
        candidate = getattr(module, callable_name, None)
        return candidate if callable(candidate) else None

    for name, obj in inspect.getmembers(module, callable):
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) == getattr(module, "__name__", None):
            return obj
    return None


def _invoke(func: Callable[..., Any], params: Any) -> Any:
    """Invoke *func* with *params*, supporting mapping, sequence, or scalar."""
    if isinstance(params, Mapping):
        return func(**params)
    if isinstance(params, (list, tuple)):
        return func(*params)
    if params is None:
        return func()
    return func(params)


def _empty_trace(module_name: str, callable_name: Optional[str], error: str) -> Dict[str, Any]:
    return {
        "module": module_name,
        "callable_name": callable_name or "<unresolved>",
        "compile_success": False,
        "compile_error": error,
        "invocation_count": 0,
        "successful_invocations": 0,
        "failed_invocations": 0,
        "total_latency_us": 0.0,
        "average_latency_us": 0.0,
        "min_latency_us": 0.0,
        "max_latency_us": 0.0,
        "traces": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_module(
    module_name: str,
    callable_name: Optional[str] = None,
    sample_params: Optional[Sequence[Any]] = None,
    case_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Import *module_name*, exercise a callable, and collect a latency trace.

    The returned trace mirrors the shape the orchestrator consumes: a
    ``compile_success`` flag (import + invocation health), per-invocation
    timings in microseconds, and aggregate latency statistics used to derive
    a velocity fitness signal.
    """
    sample_params = list(sample_params or [])
    case_names = list(case_names or [])

    # --- Import (the "compile" stage for Python variants) -----------------
    try:
        importlib.invalidate_caches()
        if module_name in tuple(__import__("sys").modules):
            module = importlib.reload(__import__("sys").modules[module_name])
        else:
            module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - any import failure is a compile failure
        return _empty_trace(module_name, callable_name, traceback.format_exc(limit=4))

    func = _resolve_callable(module, callable_name)
    if func is None:
        return _empty_trace(
            module_name,
            callable_name,
            f"No callable {callable_name or '<auto>'} found in module {module_name}",
        )

    resolved_name = getattr(func, "__name__", callable_name or "<callable>")

    # If there is nothing to invoke, the module still imported cleanly.
    if not sample_params:
        return {
            "module": module_name,
            "callable_name": resolved_name,
            "compile_success": True,
            "compile_error": None,
            "invocation_count": 0,
            "successful_invocations": 0,
            "failed_invocations": 0,
            "total_latency_us": 0.0,
            "average_latency_us": 0.0,
            "min_latency_us": 0.0,
            "max_latency_us": 0.0,
            "traces": [],
        }

    traces: List[Dict[str, Any]] = []
    latencies_us: List[float] = []
    successful = 0
    failed = 0

    for index, params in enumerate(sample_params):
        label = case_names[index] if index < len(case_names) else f"case_{index}"
        start = perf_counter_ns()
        try:
            result = _invoke(func, params)
            latency_us = _microseconds_from_ns(perf_counter_ns() - start)
            latencies_us.append(latency_us)
            successful += 1
            traces.append(
                {
                    "case": label,
                    "success": True,
                    "latency_us": latency_us,
                    "result_repr": repr(result)[:200],
                }
            )
        except Exception:  # noqa: BLE001 - record, do not propagate
            latency_us = _microseconds_from_ns(perf_counter_ns() - start)
            failed += 1
            traces.append(
                {
                    "case": label,
                    "success": False,
                    "latency_us": latency_us,
                    "error": traceback.format_exc(limit=3),
                }
            )

    total_latency = sum(latencies_us)
    count = len(sample_params)
    return {
        "module": module_name,
        "callable_name": resolved_name,
        "compile_success": True,
        "compile_error": None,
        "invocation_count": count,
        "successful_invocations": successful,
        "failed_invocations": failed,
        "total_latency_us": total_latency,
        "average_latency_us": (total_latency / len(latencies_us)) if latencies_us else 0.0,
        "min_latency_us": min(latencies_us) if latencies_us else 0.0,
        "max_latency_us": max(latencies_us) if latencies_us else 0.0,
        "traces": traces,
    }