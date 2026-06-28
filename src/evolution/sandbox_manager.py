"""Sandbox manager for isolated candidate evaluation.

In addition to local isolated sandboxes, the manager can dispatch compilation
tasks to a pool of remote workers via :class:`DistributedCoordinator`.  Remote
dispatch is entirely optional: with no ``[distributed]`` configuration (or
``enabled = false``) every task runs locally, preserving single-machine
behaviour.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class SandboxManager:
    """Creates and manages isolated sandboxes for candidate evaluation."""

    def __init__(self, workspace: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.workspace = workspace
        self.config = config or {}
        self._active_sandboxes: Dict[str, Path] = {}
        self._coordinator = None  # lazily constructed on first dispatch

    def create_sandbox(self, sandbox_id: str) -> Path:
        sandbox_dir = Path(tempfile.mkdtemp(prefix=f"aero_sandbox_{sandbox_id}_"))
        self._active_sandboxes[sandbox_id] = sandbox_dir
        return sandbox_dir

    def cleanup_sandbox(self, sandbox_id: str) -> None:
        path = self._active_sandboxes.pop(sandbox_id, None)
        if path and path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def cleanup_all(self) -> None:
        for sid in list(self._active_sandboxes):
            self.cleanup_sandbox(sid)

    def get_sandbox_path(self, sandbox_id: str) -> Optional[Path]:
        return self._active_sandboxes.get(sandbox_id)

    # ------------------------------------------------------------------
    # Distributed compilation dispatch (feature #4)
    # ------------------------------------------------------------------

    @property
    def coordinator(self):
        """Lazily build (and cache) the distributed coordinator."""
        if self._coordinator is None:
            from src.build.distributed import DistributedCoordinator

            self._coordinator = DistributedCoordinator(self.config)
        return self._coordinator

    @property
    def distributed_enabled(self) -> bool:
        return bool(self.config.get("distributed", {}).get("enabled", False))

    def dispatch_build_tasks(self, tasks: List[Any]) -> List[Any]:
        """Dispatch :class:`BuildTask` objects across workers and collect results.

        Falls back to local execution when distributed mode is disabled or no
        remote workers are reachable.
        """
        return self.coordinator.dispatch(list(tasks))

    def worker_stats(self) -> Dict[str, Any]:
        return self.coordinator.worker_stats()
