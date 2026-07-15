# -*- coding: utf-8 -*-
"""Delegated pre-write validation for generated artifacts.

The engine does not need to be a compiler.  Before any generated files are
committed to the final distribution directory, it writes them to a staging area
and runs a user-defined ``validation_cmd`` from the blueprint.  The command is
executed with the staging directory as its working directory.  Only a zero exit
status causes the files to be promoted.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ValidationError(Exception):
    """Raised when a pre-write validation command fails."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


@dataclass
class ValidationResult:
    """Outcome of a delegated validation run."""

    succeeded: bool
    command: List[str]
    output: str
    return_code: int


class PreWriteValidator:
    """Run a user-defined command against a staged workspace before promotion."""

    def __init__(self, context: Optional[Dict[str, Any]] = None, language: str = "") -> None:
        self.context: Dict[str, Any] = dict(context) if context else {}
        self.language = language

    def _resolve_command(self) -> Optional[List[str]]:
        """Return the parsed validation command, or ``None`` if no command is configured."""
        validation = self.context.get("validation")
        if isinstance(validation, dict):
            cmd = validation.get("validation_cmd") or validation.get("execution_command")
            if cmd:
                return shlex.split(str(cmd))
        return None

    def validate(
        self, workspace_root: str, *, language: Optional[str] = None
    ) -> ValidationResult:
        """Run the configured validation command in *workspace_root*.

        Returns a :class:`ValidationResult` on success.  On failure raises
        :class:`ValidationError` with the captured stdout/stderr so the operator
        sees the external validator's output, not an orchestration traceback.
        """
        saved_language = self.language
        if language is not None:
            self.language = language
        try:
            command = self._resolve_command()
        finally:
            self.language = saved_language
        if not command:
            return ValidationResult(
                succeeded=True, command=[], output="(no validation_cmd configured)", return_code=0
            )

        try:
            result = subprocess.run(
                command,
                cwd=workspace_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise ValidationError(
                f"validation command executable not found: {command[0]}", output=str(exc)
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(
                f"validation command timed out after {exc.timeout}s: {' '.join(command)}",
                output=(exc.stdout or "") + "\n" + (exc.stderr or ""),
            ) from exc

        if result.returncode != 0:
            raise ValidationError(
                f"validation command failed (exit code {result.returncode}): {' '.join(command)}\n"
                f"Captured output:\n{result.stdout}",
                output=result.stdout or "",
            )

        return ValidationResult(
            succeeded=True,
            command=command,
            output=result.stdout or "",
            return_code=result.returncode,
        )
