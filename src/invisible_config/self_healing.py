"""
Self-healing error-correction loops for the inferred build graph.

When a compilation step fails because of a *type mismatch at a language
boundary* (the classic failure mode of auto-generated FFI glue), the
:class:`SelfHealingExecutor` does not give up: it asks a :class:`GlueCodePatcher`
to rewrite the offending glue code at that boundary and retries the step, up to
a bounded number of attempts.  Everything is dependency-free and operates on
glue-code *text*, so it works against any compiler driver the caller supplies
(the step is just a callable returning success + stderr).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.invisible_config.dag_inference import FfiBoundary

# A compile step: takes the (possibly patched) glue source, returns
# (succeeded, diagnostics_text).
CompileStep = Callable[[str], Tuple[bool, str]]

# Heuristics that classify a compiler diagnostic as a healable boundary type
# mismatch (as opposed to a genuine logic error we should not paper over).
_TYPE_MISMATCH_PATTERNS = [
    re.compile(r"cannot convert .* to .*", re.IGNORECASE),
    re.compile(r"no (?:known )?conversion from .* to .*", re.IGNORECASE),
    re.compile(r"incompatible (?:integer-to-pointer |pointer )?type", re.IGNORECASE),
    re.compile(r"mismatched types?", re.IGNORECASE),
    re.compile(r"expected `?([\w:<>* ]+)`?,? *(?:but )?found `?([\w:<>* ]+)`?", re.IGNORECASE),
    re.compile(r"argument of type .* is incompatible with parameter of type", re.IGNORECASE),
    re.compile(r"invalid conversion from .* to .*", re.IGNORECASE),
]

# Map a few well-known cross-language scalar mismatches to an explicit cast the
# patcher can insert into the glue layer.
_CAST_HINTS = {
    "pybind11": {
        ("int", "double"): "static_cast<double>",
        ("double", "int"): "static_cast<int>",
        ("long", "int"): "static_cast<int>",
        ("size_t", "int"): "static_cast<int>",
        ("float", "double"): "static_cast<double>",
    },
    "pyo3": {
        ("i32", "i64"): "i64::from",
        ("i64", "i32"): "i32::try_from",
        ("f32", "f64"): "f64::from",
        ("u32", "usize"): "usize::try_from",
    },
}


@dataclass
class HealingAttempt:
    attempt: int
    succeeded: bool
    diagnostics: str
    patch_applied: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "attempt": self.attempt,
            "succeeded": self.succeeded,
            "patch_applied": self.patch_applied,
            "diagnostics": self.diagnostics[:500],
        }


@dataclass
class HealingResult:
    succeeded: bool
    glue_source: str
    attempts: List[HealingAttempt] = field(default_factory=list)
    boundary: Optional[FfiBoundary] = None

    @property
    def healed(self) -> bool:
        """True if it failed at first but later succeeded after a patch."""
        return self.succeeded and any(a.patch_applied for a in self.attempts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "succeeded": self.succeeded,
            "healed": self.healed,
            "attempts": [a.to_dict() for a in self.attempts],
            "boundary": self.boundary.to_dict() if self.boundary else None,
        }


class GlueCodePatcher:
    """Rewrites FFI glue code to resolve a boundary type mismatch."""

    def is_type_mismatch(self, diagnostics: str) -> bool:
        return any(pattern.search(diagnostics) for pattern in _TYPE_MISMATCH_PATTERNS)

    def extract_type_pair(self, diagnostics: str) -> Optional[Tuple[str, str]]:
        """Best-effort '(expected, found)' extraction from a diagnostic line."""
        for pattern in _TYPE_MISMATCH_PATTERNS:
            match = pattern.search(diagnostics)
            if match and match.re.groups >= 2 and match.lastindex and match.lastindex >= 2:
                return match.group(1).strip(), match.group(2).strip()
        return None

    def patch(self, glue_source: str, diagnostics: str, boundary: FfiBoundary) -> Optional[Tuple[str, str]]:
        """Return ``(patched_source, description)`` or ``None`` if not healable.

        The strategy is deliberately conservative: insert an explicit cast at
        the boundary for a recognised scalar mismatch, otherwise annotate the
        glue with a coercion shim marker that downstream codegen honours.
        """
        if not self.is_type_mismatch(diagnostics):
            return None

        pair = self.extract_type_pair(diagnostics)
        casts = _CAST_HINTS.get(boundary.mechanism, {})

        if pair:
            expected, found = self._normalise(pair[0]), self._normalise(pair[1])
            cast = casts.get((found, expected)) or casts.get((expected, found))
            if cast:
                marker = f"AERO_COERCE({boundary.provider}->{boundary.consumer})"
                if marker in glue_source:
                    return None  # already coerced once; avoid infinite patching
                shim = (
                    f"// {marker}: auto-inserted {cast} to bridge "
                    f"{found} -> {expected} across {boundary.mechanism}\n"
                    f"#define AERO_BRIDGE_CAST {cast}\n"
                )
                return shim + glue_source, f"insert {cast} ({found}->{expected})"

        # Fallback: drop in a generic coercion shim so a regenerated glue layer
        # widens scalars at this boundary on the next attempt.
        marker = f"AERO_AUTO_COERCE({boundary.provider}->{boundary.consumer})"
        if marker in glue_source:
            return None
        shim = f"// {marker}: generic boundary coercion enabled ({boundary.mechanism})\n"
        return shim + glue_source, "enable generic boundary coercion"

    @staticmethod
    def _normalise(type_name: str) -> str:
        cleaned = type_name.strip().strip("`'\"").replace("const ", "").replace("&", "").strip()
        # Collapse "unsigned int" etc. to the leading token for hint lookup.
        return cleaned.split()[0] if cleaned else cleaned


class SelfHealingExecutor:
    """Runs a compile step with a bounded error-correction retry loop."""

    def __init__(self, patcher: Optional[GlueCodePatcher] = None, max_attempts: int = 3) -> None:
        self.patcher = patcher or GlueCodePatcher()
        self.max_attempts = max(1, max_attempts)

    def run(self, step: CompileStep, glue_source: str, boundary: FfiBoundary) -> HealingResult:
        """Attempt ``step`` against ``glue_source``; patch + retry on mismatch."""
        result = HealingResult(succeeded=False, glue_source=glue_source, boundary=boundary)
        current = glue_source

        for attempt in range(1, self.max_attempts + 1):
            succeeded, diagnostics = step(current)
            if succeeded:
                result.attempts.append(HealingAttempt(attempt, True, diagnostics))
                result.succeeded = True
                result.glue_source = current
                return result

            patched = self.patcher.patch(current, diagnostics, boundary)
            if patched is None:
                # Not a healable boundary mismatch (or out of patches) -> stop.
                result.attempts.append(HealingAttempt(attempt, False, diagnostics))
                result.glue_source = current
                return result

            current, description = patched
            result.attempts.append(
                HealingAttempt(attempt, False, diagnostics, patch_applied=description)
            )

        result.glue_source = current
        return result
