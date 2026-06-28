# -*- coding: utf-8 -*-
"""Errors and the human-friendly diagnostic renderer.

The whole point of this module is requirement #3 of the task: when a blueprint
is broken, the tool must abort and print an *ultra-clear* message that shows the
line, the column, and a visual ``^`` pointer at the exact spot.

The rendering is Rust/Elm-inspired::

    error: unterminated string literal
      --> blueprint.aero:7:15
       |
     7 |     version = "1.0.0
       |               ^ this string is never closed
       |
       = help: strings cannot span lines; add a closing double quote (")

All errors derive from :class:`BlueprintError` and carry an optional
:class:`~blueprint_lang.positions.Span`, so any consumer can render them the
same way via :func:`format_error`.
"""

from __future__ import annotations

from typing import List, Optional

from .positions import Span


class BlueprintError(Exception):
    """Base class for every blueprint diagnostic.

    Parameters
    ----------
    message:
        The primary, one-line description (lower-case, no trailing period --
        compiler convention).
    span:
        Where the problem is.  ``None`` for purely structural problems that have
        no single source location (e.g. "no target blocks were defined").
    label:
        Short text drawn next to the ``^`` carets (e.g. "unclosed block").
    hint:
        An optional ``help:`` line offering a fix.
    """

    def __init__(
        self,
        message: str,
        span: Optional[Span] = None,
        *,
        label: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.span = span
        self.label = label
        self.hint = hint

    @property
    def line(self) -> Optional[int]:
        return self.span.start.line if self.span else None

    @property
    def column(self) -> Optional[int]:
        return self.span.start.column if self.span else None


class BlueprintSyntaxError(BlueprintError):
    """A lexing/parsing problem: bad characters, unclosed strings/brackets, etc."""


class BlueprintValidationError(BlueprintError):
    """A semantic problem in a structurally-valid blueprint.

    Examples: unknown keys, missing required keys, unknown languages, duplicate
    target names, dangling/cyclic ``requires`` dependencies.
    """


def _caret_padding(prefix: str) -> str:
    """Whitespace that aligns a caret under ``prefix``.

    Tabs are preserved so the caret lines up regardless of the reader's tab
    width; every other character becomes a single space.
    """
    return "".join("\t" if ch == "\t" else " " for ch in prefix)


def format_error(
    error: BlueprintError,
    source: str,
    filename: str = "blueprint.aero",
) -> str:
    """Render ``error`` against ``source`` into a clear, multi-line message."""
    lines = source.split("\n")
    span = error.span

    if span is None:
        out = [f"error: {error.message}"]
        out.append(f"  --> {filename}")
        if error.hint:
            out.append(f"  = help: {error.hint}")
        return "\n".join(out)

    line_no = span.start.line
    col = span.start.column

    # Resolve the line of source we will display.  If the span points just past
    # the final line (a common spot for end-of-file errors), fall back to the
    # end of the last real line so there is always something to underline.
    if 1 <= line_no <= len(lines):
        src_line = lines[line_no - 1]
    elif lines:
        line_no = len(lines)
        src_line = lines[-1]
        col = len(src_line) + 1
    else:
        src_line = ""

    gutter = str(line_no)
    margin = " " * len(gutter)

    # Caret run length: for a single-line span underline the whole token, but
    # never run past the end of the displayed text.
    if span.is_single_line:
        run = max(1, span.end.column - span.start.column)
    else:
        run = 1
    remaining = len(src_line) - (col - 1)
    if remaining > 0:
        run = min(run, remaining)
    run = max(1, run)

    prefix = src_line[: col - 1]
    carets = "^" * run
    label = f" {error.label}" if error.label else ""

    out = [
        f"error: {error.message}",
        f"  --> {filename}:{line_no}:{col}",
        f"{margin} |",
        f"{gutter} | {src_line}",
        f"{margin} | {_caret_padding(prefix)}{carets}{label}",
    ]
    if error.hint:
        out.append(f"{margin} |")
        out.append(f"{margin} = help: {error.hint}")
    return "\n".join(out)


def format_errors(
    errors: List[BlueprintError],
    source: str,
    filename: str = "blueprint.aero",
) -> str:
    """Render several diagnostics, separated by blank lines."""
    return "\n\n".join(format_error(e, source, filename) for e in errors)
