# -*- coding: utf-8 -*-
"""
Active in-tree merger for verified out-of-tree scaffold builds.

After ``scaffold --build`` produces a compiled shared library in a temporary
out-of-tree workspace, ``--merge-active`` copies that artefact into AeroNova's
own live runtime layout (``core/extensions/``) and loads it into the running
process, completing the self-hosting loop: the freshly built native component
(e.g. ``anyon_simulator``) becomes importable immediately, without a restart.

This module is the deterministic controller for that step:

* :func:`find_compiled_library` locates the ``cdylib`` output under the crate's
  ``target/{debug,release}`` directory;
* :func:`merge_active` copies it into the active extension folder under the
  Python-importable module name and (optionally) loads it live.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# cdylib outputs cargo can emit for a pyo3 extension module, by platform.
SHARED_LIB_SUFFIXES = (".so", ".dylib", ".pyd")


@dataclass
class MergeResult:
    """Outcome of an active in-tree merge."""

    merged: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    module_name: Optional[str] = None
    loaded: bool = False
    reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged": self.merged,
            "source": self.source,
            "destination": self.destination,
            "module_name": self.module_name,
            "loaded": self.loaded,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def _candidate_names(crate_name: str) -> List[str]:
    """File names cargo may produce for *crate_name*'s cdylib, most-specific first."""
    names: List[str] = []
    for suffix in SHARED_LIB_SUFFIXES:
        names.append(f"lib{crate_name}{suffix}")  # unix cdylib convention
        names.append(f"{crate_name}{suffix}")     # windows / already-renamed
    return names


def find_compiled_library(workspace_root: Path, crate_name: str) -> Optional[Path]:
    """Locate the compiled shared library for *crate_name* under ``target/``.

    Prefers the ``release`` profile, then ``debug``; within a profile an exact
    ``lib<crate>.<suffix>`` match wins, falling back to any single shared
    library present (covers crates whose lib name differs from the package).
    """
    profiles = []
    for profile in ("release", "debug"):
        directory = Path(workspace_root) / "target" / profile
        if directory.is_dir():
            profiles.append(directory)

    wanted = _candidate_names(crate_name)
    for directory in profiles:
        for name in wanted:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    # Fallback: any shared library directly in a profile dir.
    for directory in profiles:
        libs = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix in SHARED_LIB_SUFFIXES
        )
        if libs:
            return libs[0]
    return None


def active_extensions_dir() -> Path:
    """The live ``core/extensions/`` directory AeroNova loads at runtime."""
    from core.extensions import extensions_dir

    return extensions_dir()


def merge_active(
    workspace_root: Path,
    crate_name: str,
    module_name: Optional[str] = None,
    *,
    dest_dir: Optional[Path] = None,
    load: bool = True,
) -> MergeResult:
    """Copy the crate's compiled library into the active extension layer.

    The artefact is named ``<module_name><platform-suffix>`` (e.g.
    ``anyon_simulator.so``) so CPython's extension loader can import it directly
    — the file stem must match the ``#[pymodule]`` name (and thus the
    ``PyInit_<name>`` symbol).  With ``load=True`` the module is also loaded into
    the running process and registered in :data:`sys.modules` so it is instantly
    importable.
    """
    import_name = module_name or crate_name

    library = find_compiled_library(workspace_root, crate_name)
    if library is None:
        return MergeResult(
            merged=False,
            module_name=import_name,
            reason=(
                f"no compiled shared library found under "
                f"{Path(workspace_root) / 'target'} (build first with --build)"
            ),
        )

    from core.extensions import PRIMARY_SUFFIX, load_extension_file

    target_dir = Path(dest_dir) if dest_dir is not None else active_extensions_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    destination = target_dir / f"{import_name}{PRIMARY_SUFFIX}"
    shutil.copy2(library, destination)

    result = MergeResult(
        merged=True,
        source=str(library),
        destination=str(destination),
        module_name=import_name,
        notes=[f"copied {library.name} -> {destination}"],
    )

    if load:
        module = load_extension_file(destination, import_name)
        result.loaded = module is not None
        if module is not None:
            result.notes.append(f"loaded '{import_name}' into the live process")
        else:
            result.notes.append(
                f"copied but could not load '{import_name}' "
                "(ABI/symbol mismatch?); it will load on next interpreter start"
            )

    return result
