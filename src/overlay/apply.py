"""Applying overlay patches to regenerated files.

Patches are unified diffs computed against the *pristine* generated file at the
time the overlay was committed.  When code is regenerated the pristine version
may differ (the blueprint changed), so hunks are located by **content** —
matching each hunk's context + removed lines as a contiguous block — rather than
by the line numbers baked into the diff header.  This lets edits survive as long
as the surrounding code still exists.

If a hunk's pre-image cannot be located, that hunk is a *conflict*: the function
reports it and leaves the regenerated text for that region untouched, so the
caller can keep the freshly generated version and ask the user to re-commit.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def _parse_hunks(patch: str) -> List[List[Tuple[str, str]]]:
    """Split a unified diff into hunks of ``(tag, text)`` where tag is ' ', '-', '+'."""
    hunks: List[List[Tuple[str, str]]] = []
    current: Optional[List[Tuple[str, str]]] = None
    for line in patch.splitlines():
        if line.startswith("@@"):
            current = []
            hunks.append(current)
            continue
        if current is None:
            # Header lines (---/+++) precede the first @@; ignore them.
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        if line and line[0] in " -+":
            current.append((line[0], line[1:]))
        elif line == "":
            # A bare empty line inside a hunk is a context line for "".
            current.append((" ", ""))
    return [h for h in hunks if h]


def _find_block(haystack: List[str], needle: List[str], start: int) -> Optional[int]:
    """Index of the first occurrence of *needle* in *haystack* at/after *start*."""
    if not needle:
        return start
    last = len(haystack) - len(needle)
    for i in range(start, last + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    # Fall back to searching from the top (hunks may be out of order after shifts).
    for i in range(0, start):
        if i + len(needle) <= len(haystack) and haystack[i:i + len(needle)] == needle:
            return i
    return None


def apply_patch(target: str, patch: str) -> Tuple[str, bool]:
    """Apply *patch* to *target*.

    Returns ``(merged_text, conflict)``.  When ``conflict`` is True at least one
    hunk could not be located; merged_text still contains every hunk that *did*
    apply, and the conflicting regions are left as in *target*.
    """
    if not patch.strip():
        return target, False

    result = target.splitlines()
    conflict = False
    search_from = 0

    for hunk in _parse_hunks(patch):
        pre = [text for tag, text in hunk if tag in (" ", "-")]
        post = [text for tag, text in hunk if tag in (" ", "+")]

        idx = _find_block(result, pre, search_from)
        if idx is None:
            conflict = True
            continue
        result[idx:idx + len(pre)] = post
        search_from = idx + len(post)

    merged = "\n".join(result)
    if target.endswith("\n") and not merged.endswith("\n"):
        merged += "\n"
    return merged, conflict
