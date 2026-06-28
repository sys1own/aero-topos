"""Build-pipeline stage that re-applies user overlays after code generation.

This is the integration point between the build pipeline and the overlay system:
once code has been (re)generated, committed overlays are re-applied so that
manual edits survive the regeneration.  Conflicts are logged and skipped — the
freshly generated version is kept and the user is asked to re-commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Union

from src.overlay import OverlayManager, ReapplyStatus

_PathLike = Union[str, Path]


def apply_overlays_stage(
    workspace: _PathLike,
    *,
    enabled: bool = True,
    log: Callable[[str], None] = print,
) -> Dict[str, ReapplyStatus]:
    """Re-apply committed overlays under *workspace* to regenerated files.

    Returns ``{relkey: status}``.  When ``enabled`` is False (``build
    --no-overlay``) this is a no-op and returns an empty mapping.
    """
    if not enabled:
        return {}

    manager = OverlayManager(workspace)
    overlays = manager.store.list_overlays()
    if not overlays:
        return {}

    results = manager.reapply_all()
    applied = sum(1 for s in results.values() if s == ReapplyStatus.APPLIED)
    log(f"\n[overlay] re-applying {len(results)} user overlay(s)...")
    for key, status in results.items():
        if status == ReapplyStatus.APPLIED:
            log(f"[overlay] preserved manual edits in {key}")
        elif status == ReapplyStatus.CONFLICT:
            log(
                f"[overlay] CONFLICT for {key}: kept the generated version; "
                "re-run 'commit-overlay' to re-capture your edits"
            )
        elif status == ReapplyStatus.MISSING:
            log(f"[overlay] {key} has an overlay but was not generated; skipped")
    log(f"[overlay] {applied}/{len(results)} overlay(s) applied cleanly")
    return results
