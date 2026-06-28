# -*- coding: utf-8 -*-
"""Source positions and spans for the ``blueprint.aero`` language.

Every token, AST node, and error carries a :class:`Span` so the diagnostics
layer can point at the exact ``line:column`` (and underline the offending text
with ``^`` carets).  Positions are 1-based for ``line``/``column`` -- matching
what editors and compilers show users -- and ``offset`` is the 0-based index
into the raw source string, which is convenient for slicing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """A single point in the source text."""

    line: int  # 1-based line number
    column: int  # 1-based column number
    offset: int  # 0-based absolute index into the source string

    def __str__(self) -> str:  # e.g. "7:14"
        return f"{self.line}:{self.column}"


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, end)`` range of source text.

    A *point* span (where ``start == end``) is used for things like an opening
    quote of an unterminated string -- there is no width to underline, so the
    renderer draws a single caret.
    """

    start: Position
    end: Position  # exclusive

    @property
    def length(self) -> int:
        """Number of source characters covered (at least 1, for point spans)."""
        return max(1, self.end.offset - self.start.offset)

    @property
    def is_single_line(self) -> bool:
        return self.start.line == self.end.line

    @classmethod
    def point(cls, pos: Position) -> "Span":
        """A zero-width span anchored at ``pos``."""
        return cls(pos, pos)
