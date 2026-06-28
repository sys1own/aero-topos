# -*- coding: utf-8 -*-
"""
Robust diagnostic recovery loop.

Wraps the build runner in an exception/return-code loop: if ``cargo build`` exits
with ``101`` pointing at mutability or type-mismatch errors, the failing
``src/lib.rs`` is piped through the auto-correction shield
(:mod:`src.scaffold.rust_shield`) and the build is re-dispatched **once**.  The
build action itself is injected, so this loop is fully decoupled from cargo and
unit-testable with a fake builder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

from src.scaffold.rust_shield import RustSemanticShield

# A build attempt: returns (succeeded, combined_output, return_code).
BuildAction = Callable[[], Tuple[bool, str, int]]

# Failure signatures the recovery pass knows how to repair.
_RECOVERABLE_RE = re.compile(
    r"cannot borrow `\w+` as mutable|mismatched types|expected `?usize`?|"
    r"error\[E0596\]|error\[E0308\]",
    re.IGNORECASE,
)


@dataclass
class RecoveryAttempt:
    attempt: int
    succeeded: bool
    return_code: int
    corrections: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "succeeded": self.succeeded,
            "return_code": self.return_code,
            "corrections": list(self.corrections),
        }


@dataclass
class RecoveryResult:
    succeeded: bool
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    final_output: str = ""

    @property
    def recovered(self) -> bool:
        """True if a later attempt succeeded after corrections were applied."""
        return self.succeeded and any(a.corrections for a in self.attempts)

    def to_dict(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "recovered": self.recovered,
            "attempts": [a.to_dict() for a in self.attempts],
        }


class DiagnosticRecoveryRunner:
    """Runs a build action, auto-correcting and retrying on recoverable errors."""

    def __init__(self, shield: RustSemanticShield | None = None, max_retries: int = 1) -> None:
        self.shield = shield or RustSemanticShield()
        self.max_retries = max(0, max_retries)

    @staticmethod
    def is_recoverable(output: str, return_code: int) -> bool:
        # cargo uses 101 for a compile error; we also accept any non-zero with a
        # signature we recognise.
        return return_code != 0 and bool(_RECOVERABLE_RE.search(output or ""))

    def run(self, crate_root: Path, build: BuildAction) -> RecoveryResult:
        """Build, and on a recoverable failure correct ``src/lib.rs`` and retry."""
        crate_root = Path(crate_root)
        lib = crate_root / "src" / "lib.rs"
        result = RecoveryResult(succeeded=False)

        for attempt in range(1, self.max_retries + 2):  # initial + up to max_retries
            succeeded, output, code = build()
            result.final_output = output
            if succeeded:
                result.attempts.append(RecoveryAttempt(attempt, True, code))
                result.succeeded = True
                return result

            corrections: List[str] = []
            # Only attempt a correction if we have retries left, can recognise the
            # failure, and have a source file to repair.
            if attempt <= self.max_retries and self.is_recoverable(output, code) and lib.is_file():
                source = lib.read_text(encoding="utf-8")
                corrected, applied = self.shield.correct_from_diagnostics(source, output)
                if applied and corrected != source:
                    lib.write_text(corrected, encoding="utf-8")
                    corrections = applied

            result.attempts.append(RecoveryAttempt(attempt, False, code, corrections))
            if not corrections:
                # Nothing more we can safely do -- stop.
                break

        return result
