"""
The ultra-lean "Invisible Configuration Layer" blueprint dialect.

A whole project is described in a handful of lines of *semantic intent* -- no
DAG, no per-target sources, no dependency lists.  Everything else is inferred
(see :class:`~src.invisible_config.dag_inference.DAGInferenceEngine`).

Example (the entire file)::

    project "biophysical_trader"

    ingest   = ["./research/genomics.md", "./research/market_liquidity.txt"]
    targets  = ["cpp_core", "python_dashboard"]
    optimize = "maximum_hardware"

Grammar: one bare ``project "<name>"`` line, then flat ``key = value`` lines
where a value is a quoted string, number, ``true``/``false`` or a ``[list]``.
Comments start with ``#``.  It is deliberately *not* the block DSL (which
requires ``{ ... }``) nor the legacy INI (``[section]``) / JSON (``{``) formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Reuse the project's existing literal coercion so values behave consistently
# with the other blueprint dialects.
from blueprint_parser import parse_literal

_PROJECT_RE = re.compile(r'^\s*project\s+"(?P<name>[^"]*)"\s*$')
_ASSIGN_RE = re.compile(r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*$")
# A bare `key = value` (the assignment style) with NO surrounding section header
# and NO trailing `{` is the signature of the lean dialect.
_LEAN_ASSIGN_SIGNATURE = re.compile(r'^\s*(?:project\s+"|[A-Za-z_][A-Za-z0-9_]*\s*=)', re.MULTILINE)


class LeanBlueprintError(ValueError):
    """Raised when an ultra-lean blueprint cannot be parsed."""


@dataclass
class LeanBlueprint:
    """The parsed semantic intent of a lean blueprint."""

    project: str = ""
    ingest: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    optimize: str = "balanced"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "ingest": list(self.ingest),
            "targets": list(self.targets),
            "optimize": self.optimize,
            "extras": dict(self.extras),
        }


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#") or stripped.startswith("//")


def looks_like_lean_blueprint(content: str) -> bool:
    """Heuristically detect the lean dialect (vs. INI / JSON / block DSL)."""
    first_meaningful = ""
    for raw in content.splitlines():
        if _is_comment_or_blank(raw):
            continue
        first_meaningful = raw.strip()
        break
    if not first_meaningful:
        return False
    # INI starts with '[', JSON with '{'; the block DSL has a `... "name" {`.
    if first_meaningful[0] in "[{":
        return False
    if "{" in content:  # block DSL or JSON braces -> not lean
        return False
    # Must have at least a `project "..."` line or a top-level `key =` assignment.
    return bool(_LEAN_ASSIGN_SIGNATURE.search(content))


def parse_lean_blueprint(content: str) -> LeanBlueprint:
    """Parse the lean dialect into a :class:`LeanBlueprint`.

    Raises :class:`LeanBlueprintError` on a malformed line so misconfiguration
    is surfaced rather than silently ignored.
    """
    blueprint = LeanBlueprint()
    seen_keys: Dict[str, int] = {}

    for lineno, raw in enumerate(content.splitlines(), start=1):
        if _is_comment_or_blank(raw):
            continue
        # Strip trailing inline comments (outside of quotes/lists is good enough
        # for this tiny grammar; we only do it for un-quoted simple values).
        line = raw

        project_match = _PROJECT_RE.match(line)
        if project_match:
            if blueprint.project:
                raise LeanBlueprintError(f"line {lineno}: duplicate 'project' declaration")
            blueprint.project = project_match.group("name")
            continue

        assign_match = _ASSIGN_RE.match(line)
        if not assign_match:
            raise LeanBlueprintError(
                f"line {lineno}: expected 'project \"name\"' or 'key = value', got: {line.strip()!r}"
            )

        key = assign_match.group("key")
        raw_value = assign_match.group("value")
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if seen_keys[key] > 1:
            raise LeanBlueprintError(f"line {lineno}: duplicate key '{key}'")

        value = parse_literal(raw_value)
        if key == "ingest":
            blueprint.ingest = _as_str_list(value)
        elif key == "targets":
            blueprint.targets = _as_str_list(value)
        elif key == "optimize":
            blueprint.optimize = str(value)
        elif key == "project":
            blueprint.project = str(value)
        else:
            blueprint.extras[key] = value

    if not blueprint.project:
        raise LeanBlueprintError("missing required 'project \"name\"' declaration")
    if not blueprint.targets:
        raise LeanBlueprintError("missing required 'targets = [...]' declaration")
    return blueprint


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]
