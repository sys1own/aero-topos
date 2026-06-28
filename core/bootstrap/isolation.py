# -*- coding: utf-8 -*-
"""Self-hosting bootstrap isolation layer.

This module implements the staging workspace isolation needed for AeroNova to
safely target its own codebase without process locks, state corruption, or
infinite recursion.

Architecture
------------
1. **Self-targeting detection** -- compares incoming target context paths against
   the engine's own package directory (`sys.path[0]`, `main.py` location, etc.).
   If overlap is detected, the bootstrap isolation protocol engages automatically.

2. **Shadow staging** -- all file writes (TRS mutations, code-splitting output,
   build artifacts) are redirected to `.aero/bootstrap_stage/` instead of the
   live repository tree.

3. **Atomic swap** -- after the validation suite confirms zero structural
   anomalies and zero compilation errors within the staging directory, a
   filesystem overlay/copy atomically promotes the staged changes into the live
   repository layout.

4. **Recursive loop guard** -- an in-process flag (`_BOOTSTRAP_ACTIVE`) and a
   CLI-visible argument (`--bootstrap-active`) prevent nested bootstrap cycles.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _isolation_token() -> str:
    """A collision-resistant token for per-process/per-thread staging dirs.

    Combines the PID, the current thread id and a short random component so two
    concurrent optimization runs (or parallel candidate evaluations) never share
    a staging subdirectory on the same filesystem.
    """
    return f"p{os.getpid()}_t{threading.get_ident()}_{uuid.uuid4().hex[:8]}"

logger = logging.getLogger("core.bootstrap.isolation")

# ---------------------------------------------------------------------------
# Recursive loop guard (requirement #4)
# ---------------------------------------------------------------------------

_BOOTSTRAP_ACTIVE = threading.local()
_ENV_FLAG = "AERO_BOOTSTRAP_ACTIVE"


def is_bootstrap_active() -> bool:
    """Return True if the current process is already inside a bootstrap pass."""
    if os.environ.get(_ENV_FLAG, "").lower() in ("1", "true", "yes"):
        return True
    return getattr(_BOOTSTRAP_ACTIVE, "active", False)


def set_bootstrap_active(state: bool = True) -> None:
    """Mark the current process as inside an active bootstrap cycle."""
    _BOOTSTRAP_ACTIVE.active = state
    os.environ[_ENV_FLAG] = "1" if state else ""


# ---------------------------------------------------------------------------
# Self-targeting detection (requirement #1)
# ---------------------------------------------------------------------------

def _engine_root() -> Path:
    """Resolve the root directory of the running AeroNova engine package."""
    # Primary heuristic: the directory containing main.py (or __main__.py).
    candidates: List[Path] = []

    # sys.path[0] is typically the script directory or '' for interactive.
    if sys.path and sys.path[0]:
        candidates.append(Path(sys.path[0]).resolve())

    # Walk up from this file to find the repo root (contains main.py).
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "main.py").exists() and (parent / "orchestrator.py").exists():
            candidates.append(parent)
            break

    # Deduplicate and prefer the shortest (most likely repo root).
    seen = set()
    unique: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    unique.sort(key=lambda p: len(str(p)))
    return unique[0] if unique else Path.cwd()


def detect_self_targeting(target_paths: List[str]) -> bool:
    """Check if any target path overlaps with the engine's own directory.

    Parameters
    ----------
    target_paths : list[str]
        Absolute or relative paths from the build context (source entries,
        context registry paths, compilation target sources).

    Returns
    -------
    bool
        True if at least one target resolves to a path inside (or equal to)
        the engine's own installation directory tree.
    """
    engine_root = _engine_root()
    engine_str = str(engine_root)

    for raw_path in target_paths:
        resolved = Path(raw_path).resolve()
        resolved_str = str(resolved)
        # Check bidirectional containment: target inside engine OR engine inside target.
        if resolved_str.startswith(engine_str) or engine_str.startswith(resolved_str):
            logger.info(
                "Self-targeting detected: target path %r overlaps engine root %r",
                raw_path,
                engine_str,
            )
            return True
    return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BootstrapIsolationError(RuntimeError):
    """Raised when bootstrap staging or atomic swap fails."""


# ---------------------------------------------------------------------------
# Shadow staging (requirement #2) + atomic swap (requirement #3)
# ---------------------------------------------------------------------------

_STAGE_DIR_NAME = ".aero/bootstrap_stage"


class BootstrapStage:
    """Manages an isolated staging workspace for self-hosting builds.

    Usage::

        stage = BootstrapStage(workspace_root)
        stage.prepare()
        # ... run TRS mutations, code generation targeting stage.stage_dir ...
        if stage.validate(validator_fn):
            stage.promote()   # atomic swap into live tree
        else:
            stage.discard()   # rollback -- remove staging cache
    """

    def __init__(self, workspace_root: Path, isolation_token: Optional[str] = None) -> None:
        self.workspace_root = workspace_root.resolve()
        # Process-isolated staging (requirement #5): each stage lives in a
        # uniquely-named subdirectory under the shared base so concurrent runs
        # never collide on the filesystem. The base dir is created lazily.
        self.stage_base = self.workspace_root / _STAGE_DIR_NAME
        self.isolation_token = isolation_token or _isolation_token()
        self.stage_dir = self.stage_base / self.isolation_token
        self._prepared = False
        self._promoted = False

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self) -> Path:
        """Create (or reset) the staging directory.

        Returns the absolute path to the staging root so callers can redirect
        their file writes there.
        """
        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self._prepared = True
        logger.info("Bootstrap staging directory prepared: %s", self.stage_dir)
        return self.stage_dir

    def redirect_path(self, original: Path) -> Path:
        """Translate a live-tree path to its staging equivalent.

        Parameters
        ----------
        original : Path
            An absolute path under ``workspace_root`` that would normally be
            written by the build engine.

        Returns
        -------
        Path
            The corresponding path under ``.aero/bootstrap_stage/``.
        """
        try:
            relative = original.resolve().relative_to(self.workspace_root)
        except ValueError:
            # Path is outside workspace -- cannot redirect; return as-is.
            return original
        redirected = self.stage_dir / relative
        redirected.parent.mkdir(parents=True, exist_ok=True)
        return redirected

    def validate(self, validator: Callable[[Path], Dict[str, Any]]) -> bool:
        """Run the validation suite against the staged directory.

        Parameters
        ----------
        validator : callable
            A function ``(stage_dir: Path) -> {"errors": int, "anomalies": int, ...}``
            that runs syntax scanning and structural validation against the
            staged output.

        Returns
        -------
        bool
            True if the validator reports zero errors and zero anomalies.
        """
        if not self._prepared:
            raise BootstrapIsolationError(
                "Cannot validate: staging directory was never prepared."
            )
        result = validator(self.stage_dir)
        errors = int(result.get("errors", 0))
        anomalies = int(result.get("anomalies", 0))

        if errors == 0 and anomalies == 0:
            logger.info(
                "Bootstrap validation passed: 0 errors, 0 anomalies in %s",
                self.stage_dir,
            )
            return True

        logger.warning(
            "Bootstrap validation FAILED: %d error(s), %d anomaly(ies) in %s",
            errors,
            anomalies,
            self.stage_dir,
        )
        return False

    def promote(self) -> List[str]:
        """Atomically copy validated staged files into the live workspace.

        Overwrites existing files in the live tree with the staged versions.
        Returns the list of relative paths that were promoted.

        Raises
        ------
        BootstrapIsolationError
            If the staging directory doesn't exist or was already promoted.
        """
        if self._promoted:
            raise BootstrapIsolationError("Stage has already been promoted.")
        if not self.stage_dir.exists():
            raise BootstrapIsolationError(
                "Staging directory does not exist; nothing to promote."
            )

        promoted: List[str] = []
        for staged_file in self.stage_dir.rglob("*"):
            if not staged_file.is_file():
                continue
            relative = staged_file.relative_to(self.stage_dir)
            live_target = self.workspace_root / relative
            live_target.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: write to a temp file then rename (POSIX atomic rename).
            tmp_path = live_target.with_suffix(live_target.suffix + ".aero_tmp")
            shutil.copy2(staged_file, tmp_path)
            os.replace(tmp_path, live_target)
            promoted.append(str(relative))

        self._promoted = True
        logger.info("Bootstrap promotion complete: %d file(s) promoted.", len(promoted))

        # Clean up the staging directory after successful promotion.
        self.discard()
        return promoted

    def discard(self) -> None:
        """Remove the staging directory (safe rollback).

        Called automatically after a failed validation or after successful
        promotion.  Safe to call multiple times.
        """
        if self.stage_dir.exists():
            shutil.rmtree(self.stage_dir, ignore_errors=True)
            logger.info("Bootstrap staging directory discarded: %s", self.stage_dir)
        self._prepared = False
