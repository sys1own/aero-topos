"""On-disk storage for overlay patches and pristine build snapshots.

Two directories under the workspace make up the overlay state:

* ``.build_cache/<relpath>``      -- the last *pristine* generated version of a
  file (the baseline a patch is computed against).
* ``.overlays/<relpath>.patch``   -- the user's committed edits as a unified diff.

Both are keyed by the file's path relative to the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

_PathLike = Union[str, Path]

DEFAULT_BUILD_CACHE = ".build_cache"
DEFAULT_OVERLAYS = ".overlays"


class OverlayStore:
    def __init__(
        self,
        workspace: _PathLike,
        build_cache_dir: str = DEFAULT_BUILD_CACHE,
        overlays_dir: str = DEFAULT_OVERLAYS,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.build_cache_dir = self.workspace / build_cache_dir
        self.overlays_dir = self.workspace / overlays_dir

    # -- key derivation -------------------------------------------------------
    def relkey(self, file: _PathLike) -> str:
        """Return the workspace-relative key for *file* (POSIX separators)."""
        resolved = Path(file).resolve()
        try:
            rel = resolved.relative_to(self.workspace)
        except ValueError:
            # Outside the workspace: fall back to a flattened absolute path so
            # the key is still stable and filesystem-safe.
            rel = Path(os.path.splitdrive(str(resolved))[1].lstrip("/\\"))
        return rel.as_posix()

    def cache_path(self, file: _PathLike) -> Path:
        return self.build_cache_dir / self.relkey(file)

    def overlay_path(self, file: _PathLike) -> Path:
        return self.overlays_dir / (self.relkey(file) + ".patch")

    # -- build cache (pristine baselines) ------------------------------------
    def record_generated(self, file: _PathLike, content: str = None) -> None:
        """Snapshot the pristine generated *content* (or the file's current text)."""
        if content is None:
            content = Path(file).read_text(encoding="utf-8")
        dest = self.cache_path(file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def read_cache(self, file: _PathLike):
        path = self.cache_path(file)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    # -- overlays -------------------------------------------------------------
    def save_overlay(self, file: _PathLike, patch: str) -> Path:
        path = self.overlay_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch, encoding="utf-8")
        return path

    def read_overlay(self, file: _PathLike):
        path = self.overlay_path(file)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def has_overlay(self, file: _PathLike) -> bool:
        return self.overlay_path(file).is_file()

    def remove_overlay(self, file: _PathLike) -> bool:
        path = self.overlay_path(file)
        if path.is_file():
            path.unlink()
            return True
        return False

    def list_overlays(self) -> List[str]:
        """Return the relative keys of all files that have a committed overlay."""
        if not self.overlays_dir.is_dir():
            return []
        keys: List[str] = []
        for patch in sorted(self.overlays_dir.rglob("*.patch")):
            rel = patch.relative_to(self.overlays_dir).as_posix()
            keys.append(rel[: -len(".patch")])
        return keys

    def file_for_key(self, key: str) -> Path:
        return self.workspace / key
