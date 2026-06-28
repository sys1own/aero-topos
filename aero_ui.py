# -*- coding: utf-8 -*-
"""Clean, scannable terminal UI for the Aero build engine.

Inspired by Cargo and Bun: each phase of the pipeline gets a bold,
bracketed tag, coloured where the terminal supports it, so progress is
immediately visible at a glance::

    [Parsing]    blueprint.aero
    [Validating] 3 targets, 0 errors
    [Compiling]  core_engine (cpp)  ........................ ok
    [Compiling]  bindings (python)  ........................ ok
    [Success]    3 targets compiled in 1.2s

All output goes through a single :class:`AeroUI` instance so the rest
of the codebase never writes raw ``print()`` during a managed build.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import IO, List, Optional

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty()

_RESET = "" if _NO_COLOR else "\033[0m"
_BOLD = "" if _NO_COLOR else "\033[1m"
_DIM = "" if _NO_COLOR else "\033[2m"
_GREEN = "" if _NO_COLOR else "\033[32m"
_CYAN = "" if _NO_COLOR else "\033[36m"
_YELLOW = "" if _NO_COLOR else "\033[33m"
_RED = "" if _NO_COLOR else "\033[31m"
_MAGENTA = "" if _NO_COLOR else "\033[35m"
_WHITE = "" if _NO_COLOR else "\033[37m"

_TAG_COLORS = {
    "Parsing": _CYAN,
    "Validating": _MAGENTA,
    "Resolving": _MAGENTA,
    "Compiling": _GREEN,
    "Compiled": _GREEN,
    "Skipped": _YELLOW,
    "Success": _GREEN,
    "Error": _RED,
    "Warning": _YELLOW,
    "Info": _CYAN,
    "Plan": _CYAN,
    "Debug": _DIM,
    "Hint": _YELLOW,
}

_TAG_WIDTH = 13  # pad tag to consistent width


def _format_tag(tag: str) -> str:
    """Render ``tag`` as a bold, colour-coded, fixed-width ``[Tag]`` label."""
    color = _TAG_COLORS.get(tag, _WHITE)
    label = f"[{tag}]"
    padded = label.ljust(_TAG_WIDTH)
    return f"{_BOLD}{color}{padded}{_RESET}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


# ---------------------------------------------------------------------------
# The UI surface
# ---------------------------------------------------------------------------


@dataclass
class AeroUI:
    """A single, stateful sink for all human-facing build output.

    Every phase of the pipeline reports through one instance so the engine
    never scatters raw ``print()`` calls.  The UI also tallies how many
    targets compiled, were skipped, or failed, which drives the final
    ``success``/``failure`` summary line.
    """

    stream: IO[str] = field(default_factory=lambda: sys.stdout)
    _compiled: int = 0
    _skipped: int = 0
    _failed: int = 0
    _start: float = field(default_factory=time.time)

    # -- low-level emit ----------------------------------------------------

    def _emit(self, line: str = "") -> None:
        self.stream.write(line + "\n")
        # Flush so progress is visible immediately on line-buffered terminals.
        try:
            self.stream.flush()
        except Exception:  # noqa: BLE001 - some streams (StringIO) are fine without
            pass

    def _line(self, tag: str, message: str) -> None:
        self._emit(f"{_format_tag(tag)} {message}")

    # -- phase tags --------------------------------------------------------

    def parsing(self, path: str) -> None:
        self._line("Parsing", path)

    def validating(self, target_count: int, error_count: int = 0) -> None:
        msg = f"{target_count} targets"
        if error_count:
            msg += f", {error_count} error(s)"
        else:
            msg += ", 0 errors"
        self._line("Validating", msg)

    def resolving(self, target_count: int, stage_count: int) -> None:
        self._line(
            "Resolving",
            f"{target_count} targets across {stage_count} stages",
        )

    def compiling(self, name: str, language: str) -> None:
        self._line("Compiling", f"{name} {_dim('(' + language + ')')}")

    def compiled(self, name: str, language: str, duration: Optional[str] = None) -> None:
        self._compiled += 1
        suffix = f"  {_dim(duration)}" if duration else ""
        self._line("Compiled", f"{name} {_dim('(' + language + ')')}{suffix}")

    def skipped(self, name: str, reason: str = "") -> None:
        self._skipped += 1
        msg = name
        if reason:
            msg += f"  {_dim(reason)}"
        self._line("Skipped", msg)

    def compile_error(self, name: str, message: str) -> None:
        self._failed += 1
        self._line("Error", f"{name}: {message}")

    def warning(self, message: str) -> None:
        self._line("Warning", message)

    def info(self, message: str) -> None:
        self._line("Info", message)

    def hint(self, message: str) -> None:
        self._line("Hint", message)

    def plan(self, message: str) -> None:
        self._line("Plan", message)

    # -- multi-line diagnostic block --------------------------------------

    def debug_block(self, title: str, lines: List[str]) -> None:
        """Render a titled block of verbatim diagnostic lines (e.g. a manifest)."""
        self._line("Debug", title)
        for raw in lines:
            self._emit(f"  {_dim('|')} {raw}")

    # -- terminal summaries ------------------------------------------------

    def success(self) -> None:
        elapsed = time.time() - self._start
        parts = [f"{self._compiled} compiled"]
        if self._skipped:
            parts.append(f"{self._skipped} skipped")
        self._line("Success", f"{', '.join(parts)} in {elapsed:.1f}s")

    def failure(self) -> None:
        elapsed = time.time() - self._start
        self._line(
            "Error",
            f"build failed: {self._failed} error(s) after {elapsed:.1f}s",
        )

    def build_failure_report(
        self,
        target: str,
        details: str,
        suggestions: Optional[List[str]] = None,
    ) -> None:
        """Print the canonical ``Aero Build Failure`` block for one target."""
        self._emit()
        self._emit(f"{_BOLD}{_RED}Aero Build Failure{_RESET} {_dim('->')} {_BOLD}{target}{_RESET}")
        self._emit(f"{_DIM}{'-' * 60}{_RESET}")
        for raw in details.splitlines() or [details]:
            self._emit(f"  {raw}")
        if suggestions:
            self._emit()
            self._emit(f"{_BOLD}{_YELLOW}Possible cause{_RESET}")
            for suggestion in suggestions:
                self._emit(f"  {_YELLOW}-{_RESET} {suggestion}")
        self._emit()

    # -- introspection -----------------------------------------------------

    @property
    def has_errors(self) -> bool:
        return self._failed > 0

    @property
    def stats(self) -> dict:
        return {
            "compiled": self._compiled,
            "skipped": self._skipped,
            "failed": self._failed,
            "elapsed": time.time() - self._start,
        }