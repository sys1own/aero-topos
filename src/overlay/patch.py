"""Unified-diff computation for the overlay system.

A *patch* captures the difference between a pristine generated file and the
user's hand-edited version.  Patches are stored as standard unified diffs so
they are human-readable and tool-compatible.
"""

from __future__ import annotations

import difflib


def make_patch(original: str, modified: str, fromfile: str = "a", tofile: str = "b") -> str:
    """Return a unified diff turning *original* into *modified*.

    An empty string is returned when the two texts are identical.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=3,
    )
    return "".join(diff)


def is_empty_patch(patch: str) -> bool:
    """True when *patch* contains no actual changes."""
    return not patch.strip()
