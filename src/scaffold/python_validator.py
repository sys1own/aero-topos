# -*- coding: utf-8 -*-
"""
Native Python validation dispatch — no Cargo, no Rust recovery loop.

Runs ``py_compile`` / ``compileall`` inside the isolated out-of-tree workspace
and captures ``SyntaxError`` diagnostics when validation fails.
"""

from __future__ import annotations

import compileall
import py_compile
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

BuildAction = Callable[[], tuple]


@dataclass
class PythonValidationAttempt:
    attempt: int
    succeeded: bool
    return_code: int
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "succeeded": self.succeeded,
            "return_code": self.return_code,
            "errors": list(self.errors),
        }


@dataclass
class PythonValidationResult:
    succeeded: bool
    attempts: List[PythonValidationAttempt] = field(default_factory=list)
    final_output: str = ""
    language: str = "python"

    @property
    def recovered(self) -> bool:
        return False  # Python path does not use Rust auto-correction

    def to_dict(self) -> dict:
        return {
            "succeeded": self.succeeded,
            "recovered": self.recovered,
            "attempts": [a.to_dict() for a in self.attempts],
            "language": self.language,
        }


class PythonValidationRunner:
    """Validate Python sources via bytecode compilation — never invokes cargo."""

    def __init__(self, optimize: int = 0) -> None:
        self.optimize = optimize

    def validate_workspace(self, workspace_root: Path) -> PythonValidationResult:
        workspace_root = Path(workspace_root)
        errors: List[str] = []
        output_lines: List[str] = []

        py_files = sorted(workspace_root.rglob("*.py"))
        if not py_files:
            return PythonValidationResult(
                succeeded=False,
                attempts=[PythonValidationAttempt(1, False, 1, ["no .py files found"])],
                final_output="no .py files found",
            )

        for py_file in py_files:
            rel = py_file.relative_to(workspace_root)
            try:
                py_compile.compile(str(py_file), doraise=True)
                output_lines.append(f"[ok] {rel}")
            except SyntaxError as exc:
                msg = f"SyntaxError in {rel}: {exc.msg} (line {exc.lineno})"
                errors.append(msg)
                output_lines.append(f"[fail] {msg}")
            except py_compile.PyCompileError as exc:
                msg = f"SyntaxError in {rel}: {exc.msg}"
                errors.append(msg)
                output_lines.append(f"[fail] {msg}")

        if not errors:
            ok = compileall.compile_dir(
                str(workspace_root),
                quiet=1,
                optimize=self.optimize,
            )
            if not ok:
                errors.append("compileall reported failures")
                output_lines.append("[fail] compileall reported failures")

        succeeded = not errors
        attempt = PythonValidationAttempt(
            attempt=1,
            succeeded=succeeded,
            return_code=0 if succeeded else 1,
            errors=errors,
        )
        return PythonValidationResult(
            succeeded=succeeded,
            attempts=[attempt],
            final_output="\n".join(output_lines),
        )
