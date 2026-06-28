"""High-level overlay orchestration.

:class:`OverlayManager` ties together the patch, apply, and store layers into the
two operations the rest of the system needs:

* ``commit_overlay(file)`` -- capture the user's manual edits as a patch
  (the diff between the file now and its last pristine generated version).
* ``reapply_all()`` -- after regeneration, re-apply every committed overlay so
  manual edits survive, snapshotting the new pristine version as the baseline.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

from src.overlay.apply import apply_patch
from src.overlay.patch import is_empty_patch, make_patch
from src.overlay.store import OverlayStore

_PathLike = Union[str, Path]


class OverlayError(Exception):
    """Raised when an overlay operation cannot proceed."""


class ReapplyStatus(str, Enum):
    APPLIED = "applied"
    CONFLICT = "conflict"
    MISSING = "missing"  # overlay exists but the generated file does not


class OverlayManager:
    def __init__(self, workspace: _PathLike, store: Optional[OverlayStore] = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.store = store or OverlayStore(self.workspace)

    # -- generation hook ------------------------------------------------------
    def record_generated(self, file: _PathLike, content: str = None) -> None:
        """Snapshot a freshly generated file as the pristine baseline."""
        self.store.record_generated(file, content)

    # -- commit ---------------------------------------------------------------
    def commit_overlay(self, file: _PathLike) -> Optional[str]:
        """Persist the diff between *file* and its pristine baseline.

        Returns the patch text, or ``None`` when there are no edits (in which
        case any stale overlay is removed).  Raises :class:`OverlayError` if no
        pristine baseline has been recorded for the file.
        """
        path = Path(file).resolve()
        if not path.is_file():
            raise OverlayError(f"File not found: {path}")

        baseline = self.store.read_cache(path)
        if baseline is None:
            raise OverlayError(
                f"No pristine baseline in {self.store.build_cache_dir} for {path}. "
                "Generate/build the file before committing an overlay."
            )

        key = self.store.relkey(path)
        current = path.read_text(encoding="utf-8")
        patch = make_patch(baseline, current, fromfile=key, tofile=key)
        if is_empty_patch(patch):
            self.store.remove_overlay(path)
            return None
        self.store.save_overlay(path, patch)
        return patch

    # -- reapply --------------------------------------------------------------
    def reapply(self, file: _PathLike) -> ReapplyStatus:
        """Re-apply the committed overlay to a (re)generated *file*."""
        path = Path(file).resolve()
        overlay = self.store.read_overlay(path)
        if overlay is None:
            return ReapplyStatus.APPLIED  # nothing to do
        if not path.is_file():
            return ReapplyStatus.MISSING

        pristine = path.read_text(encoding="utf-8")
        # The current on-disk content is the new pristine baseline.
        self.store.record_generated(path, pristine)

        merged, conflict = apply_patch(pristine, overlay)
        if conflict:
            # Keep the freshly generated version; user must re-commit.
            return ReapplyStatus.CONFLICT
        path.write_text(merged, encoding="utf-8")
        return ReapplyStatus.APPLIED

    def reapply_all(self) -> Dict[str, ReapplyStatus]:
        """Re-apply every committed overlay; returns ``{relkey: status}``."""
        results: Dict[str, ReapplyStatus] = {}
        for key in self.store.list_overlays():
            results[key] = self.reapply(self.store.file_for_key(key))
        return results

    # -- structural (AST) reapply --------------------------------------------
    def structural_reapply(
        self,
        file: _PathLike,
        regenerated_text: str,
        *,
        language=None,
        build_fn=None,
        blueprint_path: _PathLike = None,
    ) -> ReapplyStatus:
        """Re-apply user edits with the structural 3-way AST merge engine.

        This is the AST-based successor to the line-by-line overlay reapply:
        ``base`` is the pristine snapshot in ``.build_cache``, *Left* is the
        user's current on-disk file, and *Right* is ``regenerated_text``.  The
        merge is verified before being written; on conflict/verification failure
        the on-disk file is left untouched and (if given) the collision is
        flagged in ``blueprint.aero``.
        """
        from core.overlay.structural_merger import StructuralMerger

        path = Path(file).resolve()
        if not path.is_file():
            return ReapplyStatus.MISSING
        base = self.store.read_cache(path)
        if base is None:
            base = path.read_text(encoding="utf-8")

        merger = StructuralMerger(self.workspace)
        outcome = merger.merge_file(
            path, base, regenerated_text,
            language=language, build_fn=build_fn, blueprint_path=blueprint_path,
        )
        if not outcome.accepted:
            return ReapplyStatus.CONFLICT
        # The freshly generated text is the pristine baseline for the next cycle.
        self.store.record_generated(path, regenerated_text)
        return ReapplyStatus.APPLIED
