# -*- coding: utf-8 -*-
"""Out-of-tree workspace isolation.

Keeps the ``aero-universal`` repository completely clean: every transient
manifest, scaffolded directory layout, build-cache stream and ``target/`` output
is written **outside** the tool's own tree -- either to a system temp directory
(auto-cleaned) or to a user-supplied ``distribution_directory`` (kept).

A guard refuses to materialise a workspace inside the tool directory, so a
mis-configured path can never clutter the tool again.

Staged workflow:

1. ``create()`` materialises a temporary staging directory.
2. Generated files and any build artifacts are written into the staging dir.
3. A delegated ``validation_cmd`` runs in the staging dir.
4. ``commit()`` atomically moves the staging directory to the final
   ``distribution_directory`` only when validation succeeded.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

# The tool's own repository root (this file is src/scaffold/workspace.py).
TOOL_ROOT = Path(__file__).resolve().parents[2]


class WorkspaceLocationError(ValueError):
    """Raised when a requested workspace would land inside the tool tree."""


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class OutOfTreeWorkspace:
    """A scaffolding/build workspace guaranteed to live outside the tool tree.

    Use as a context manager; a temp workspace is removed on exit, while an
    explicit ``distribution_directory`` is preserved (it is the deliverable).
    """

    def __init__(
        self,
        distribution_directory: Optional[Path] = None,
        prefix: str = "aero-build-",
        keep: Optional[bool] = None,
    ) -> None:
        self._distribution = Path(distribution_directory).expanduser() if distribution_directory else None
        self._prefix = prefix
        # Default: keep an explicit distribution dir, discard a temp one.
        self.keep = keep if keep is not None else (self._distribution is not None)
        self._root: Optional[Path] = None
        self._committed = False

    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("workspace not created; use `with OutOfTreeWorkspace(...) as ws:`")
        return self._root

    @property
    def is_temporary(self) -> bool:
        return self._distribution is None

    @property
    def is_committed(self) -> bool:
        return self._committed

    def create(self) -> Path:
        """Materialise the staging workspace directory (idempotent)."""
        if self._root is not None:
            return self._root
        if self._distribution is not None:
            target = self._distribution.resolve()
            if _is_inside(target, TOOL_ROOT):
                raise WorkspaceLocationError(
                    f"distribution_directory '{target}' is inside the tool tree ({TOOL_ROOT}); "
                    "choose a path outside it to keep aero-universal clean."
                )
        self._root = Path(tempfile.mkdtemp(prefix=self._prefix)).resolve()
        return self._root

    def commit(self) -> Path:
        """Promote the staging directory to the final distribution directory.

        Returns the final path.  No-op when no distribution directory was
        requested.  Raises if the distribution path is inside the tool tree.
        """
        if self._committed or self._distribution is None:
            return self._root or Path()

        final = self._distribution.resolve()
        if _is_inside(final, TOOL_ROOT):
            raise WorkspaceLocationError(
                f"distribution_directory '{final}' is inside the tool tree ({TOOL_ROOT})"
            )

        # Replace any prior deliverable with the validated staging tree.
        if final.exists():
            if final.is_dir():
                shutil.rmtree(final)
            else:
                final.unlink()

        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._root), str(final))
        self._root = final
        self._committed = True
        return final

    def cleanup(self) -> None:
        """Remove the staging workspace."""
        if self._committed or self._root is None:
            return
        if self._distribution is not None and self._root != self._distribution.resolve():
            # Uncommitted staging for a distribution target -- never keep it.
            shutil.rmtree(self._root, ignore_errors=True)
        elif not self.keep:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    # -- context manager --------------------------------------------------

    def __enter__(self) -> "OutOfTreeWorkspace":
        self.create()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def assert_out_of_tree(path: Path) -> None:
    """Raise if ``path`` is inside the tool tree (a safety net for callers)."""
    if _is_inside(Path(path), TOOL_ROOT):
        raise WorkspaceLocationError(
            f"refusing to write build assets to '{path}' inside the tool tree ({TOOL_ROOT})"
        )
