# -*- coding: utf-8 -*-
"""
RUSTFLAGS policy for Aero's Rust backend.

Aero can inject ``RUSTFLAGS`` (codegen flags) when invoking ``cargo``, but an
aggressive default such as ``-C target-cpu=native`` is **not portable** -- it
breaks on cross-compiles, heterogeneous CI fleets and older CPUs.  This module
turns a blueprint's intent into a concrete, *explainable* decision and makes it
easy to disable or fully customise:

* ``optimization = "none"``     -> inject nothing (portable; pass the host env through);
* ``optimization = "generic"``  -> ``-C target-cpu=generic`` (portable, still tuned);
* ``optimization = "native"`` / ``"maximum_hardware"`` -> ``-C target-cpu=native``;
* ``optimization = "size"``     -> ``-C opt-level=z``;
* ``rustflags = ["-C", "target-cpu=generic"]`` -> used **verbatim**, overriding all of the above.

The default (no optimization word, no rustflags) injects **nothing**, so builds
are portable unless the user opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# optimization word -> the RUSTFLAGS it implies (empty list == inject nothing).
_OPTIMIZATION_FLAGS = {
    "none": [],
    "off": [],
    "portable": ["-C", "target-cpu=generic"],
    "generic": ["-C", "target-cpu=generic"],
    "balanced": [],
    "default": [],
    "native": ["-C", "target-cpu=native"],
    "maximum_hardware": ["-C", "target-cpu=native"],
    "aggressive": ["-C", "target-cpu=native"],
    "max": ["-C", "target-cpu=native"],
    "size": ["-C", "opt-level=z"],
}

# Words that explicitly mean "do not inject anything".
_DISABLED_WORDS = {"none", "off", "balanced", "default", ""}


@dataclass
class RustFlagsDecision:
    """The resolved RUSTFLAGS, whether to inject them, and why."""

    flags: List[str] = field(default_factory=list)
    inject: bool = False
    reason: str = ""
    source: str = "default"  # "explicit" | "optimization" | "default"

    @property
    def value(self) -> str:
        """The ``RUSTFLAGS`` string (space-joined)."""
        return " ".join(self.flags)

    def env(self, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return an environment dict with ``RUSTFLAGS`` set, if injecting.

        When injection is disabled, the base environment is returned unchanged
        (so any ``RUSTFLAGS`` the user already exported is respected).
        """
        env = dict(base_env) if base_env is not None else {}
        if self.inject:
            env["RUSTFLAGS"] = self.value
        return env

    def to_dict(self) -> Dict[str, object]:
        return {
            "flags": list(self.flags),
            "inject": self.inject,
            "reason": self.reason,
            "source": self.source,
            "RUSTFLAGS": self.value if self.inject else None,
        }


def resolve_rustflags(
    optimization: Optional[str] = None,
    rustflags: Optional[Sequence[str]] = None,
) -> RustFlagsDecision:
    """Resolve the RUSTFLAGS to inject from a target's blueprint options.

    Precedence: an explicit ``rustflags`` list wins (used verbatim); otherwise
    the ``optimization`` word selects a preset; otherwise nothing is injected.
    """
    # 1. Explicit rustflags override everything (an empty list means "none").
    if rustflags is not None:
        flags = [str(f) for f in rustflags]
        if flags:
            return RustFlagsDecision(
                flags=flags,
                inject=True,
                reason="explicit rustflags from blueprint",
                source="explicit",
            )
        return RustFlagsDecision(
            flags=[],
            inject=False,
            reason="explicit empty rustflags (injection disabled)",
            source="explicit",
        )

    # 2. optimization word.
    if optimization is not None:
        word = str(optimization).strip().lower()
        if word in _DISABLED_WORDS:
            return RustFlagsDecision(
                flags=[],
                inject=False,
                reason=f"optimization='{optimization}' (no RUSTFLAGS injected; portable)",
                source="optimization",
            )
        if word in _OPTIMIZATION_FLAGS:
            flags = list(_OPTIMIZATION_FLAGS[word])
            return RustFlagsDecision(
                flags=flags,
                inject=bool(flags),
                reason=f"optimization='{optimization}'",
                source="optimization",
            )
        # Unknown word: stay safe and inject nothing, but say so.
        return RustFlagsDecision(
            flags=[],
            inject=False,
            reason=f"unrecognised optimization='{optimization}'; no RUSTFLAGS injected",
            source="optimization",
        )

    # 3. Default: portable, inject nothing.
    return RustFlagsDecision(
        flags=[],
        inject=False,
        reason="default: no RUSTFLAGS injected (portable build)",
        source="default",
    )
