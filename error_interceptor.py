# -*- coding: utf-8 -*-
"""Unified error interceptor for downstream compiler failures.

When a subprocess compiler (``clang++``, ``rustc``, ``gcc``, etc.) exits
non-zero, this module intercepts the raw ``stderr`` stream and re-formats
it into a clean, user-friendly report under an ``Aero Build Failure``
header.  The build tool itself never dumps its own Python stack traces to
the user on a compiler error -- it exits gracefully with a non-zero code.

Usage from the build pipeline::

    from error_interceptor import handle_compile_results

    results = [compile_target(...), ...]
    exit_code = handle_compile_results(results, ui)
"""

from __future__ import annotations

import re
import sys
from typing import IO, List, Optional, Sequence

from aero_ui import AeroUI

# Try to import CompileResult; make it available for type hints.
from src.build.compilers import CompileResult


# ---------------------------------------------------------------------------
# stderr normalisation
# ---------------------------------------------------------------------------

# Strip ANSI escape codes that compilers may emit.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _limit_lines(text: str, max_lines: int = 40) -> str:
    """Trim excessively long compiler output so the terminal stays readable."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    omitted = len(lines) - max_lines
    kept.append(f"  ... ({omitted} more lines omitted)")
    return "\n".join(kept)


def normalise_stderr(stderr: str, max_lines: int = 40) -> str:
    """Clean raw compiler stderr into a presentable form."""
    cleaned = _strip_ansi(stderr).rstrip()
    if not cleaned:
        return "(no output from compiler)"
    return _limit_lines(cleaned, max_lines)


# ---------------------------------------------------------------------------
# Result handler
# ---------------------------------------------------------------------------


def _diagnose(result: CompileResult) -> Optional[List[str]]:
    """Run language-specific root-cause analysis for a failed result.

    Currently Rust-aware: turns a cryptic "method not found" / unresolved-import
    failure into a version-mismatch hypothesis with the actual version in use.
    """
    details = result.details or {}
    if details.get("language") != "rust":
        return None
    try:
        from pathlib import Path

        from src.build.error_analysis import analyze_rust_error

        crate_root = details.get("crate_root")
        diagnosis = analyze_rust_error(
            result.stderr,
            dependencies=details.get("declared_dependencies"),
            crate_root=Path(crate_root) if crate_root else None,
        )
    except Exception:  # noqa: BLE001 - analysis must never break error reporting
        return None
    return diagnosis.render() if diagnosis else None


def handle_compile_results(
    results: Sequence[CompileResult],
    ui: AeroUI,
) -> int:
    """Process a batch of compile results, report errors, return exit code.

    * For each failed result, prints a formatted ``Aero Build Failure``
      block via the UI, enriched with root-cause suggestions where available.
    * Returns 0 if every result succeeded, 1 otherwise.
    """
    failures: List[CompileResult] = []
    for result in results:
        if result.success:
            continue
        failures.append(result)
        cleaned = normalise_stderr(result.stderr)
        ui.build_failure_report(result.target_name, cleaned, suggestions=_diagnose(result))

    if not failures:
        return 0

    ui.failure()
    return 1


# ---------------------------------------------------------------------------
# Top-level exception guard (for wrapping the entire build entry-point)
# ---------------------------------------------------------------------------


def guarded_main(
    entry: "callable",
    ui: Optional[AeroUI] = None,
    stream: Optional[IO[str]] = None,
) -> int:
    """Run *entry* inside an exception guard that never leaks raw tracebacks.

    Any uncaught exception from the build pipeline is caught, formatted
    under the ``Aero Build Failure`` header, and the process exits cleanly.

    Parameters
    ----------
    entry : callable
        A zero-argument callable that returns an ``int`` exit code.
    ui : AeroUI, optional
        Existing UI instance.  One is created if not supplied.
    stream : IO[str], optional
        Output stream (defaults to ``sys.stderr``).
    """
    if ui is None:
        ui = AeroUI(stream=stream or sys.stderr)
    try:
        return entry()
    except KeyboardInterrupt:
        ui.warning("interrupted")
        return 130
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except Exception as exc:
        ui.build_failure_report(
            "internal",
            f"{type(exc).__name__}: {exc}",
        )
        return 1


# ---------------------------------------------------------------------------
# Self-Healing v2: reify a parse/dependency failure as a broken HIN edge
# ---------------------------------------------------------------------------
def reify_parse_failure_as_port(network, message: str):
    """Reify a compiler/parser failure as an un-terminated HIN edge.

    Instead of regex/string lookahead patching, an unresolved error becomes a
    physical broken boundary interface port: a Constructor whose return path
    (``a_2``) is left without a partner.  The argument path is safely
    terminated so exactly one edge is un-terminated -- the fault site that
    :class:`orchestrator.TopologicalSelfHealer` will geometrically re-wire.
    Returns the un-terminated :class:`~core.hin_vm.Port`.
    """
    from core.hin_vm import ConstructorNode, EraserNode

    node = ConstructorNode(network.fresh_id("γ"))
    network.register_node(node)
    node.broken_reason = message

    eraser = EraserNode(network.fresh_id("ε"))
    network.register_node(eraser)
    network._link(eraser.p, node.a_1)

    # a_2 is deliberately left un-terminated: the reified broken edge.
    return node.a_2
