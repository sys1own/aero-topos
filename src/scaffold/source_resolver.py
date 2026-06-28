# -*- coding: utf-8 -*-
"""
Flexible source-entry resolution (zero file relocation).

A ``source_entry`` may live *anywhere* on the filesystem -- ``/content/lib.rs``,
``../data/core.rs``, ``~/work/sim.rs`` or a plain relative path.  This module
resolves such a path from any of those forms and copies the file straight into
the transient compilation workspace, so the user never has to move a file into
the tool's directory and the engine never raises a spurious "source not found"
when the file plainly exists elsewhere.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Extension -> canonical language tag.
_LANGUAGE_BY_EXT = {
    ".rs": "rust",
    ".py": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".f90": "fortran",
}


class SourceEntryNotFound(FileNotFoundError):
    """Raised only when a source entry cannot be found in *any* candidate form."""


@dataclass
class SourceEntry:
    """A resolved source file plus where it came from and what it is."""

    original: str
    path: Path  # absolute, resolved location of the real file
    language: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="ignore")

    def to_dict(self) -> dict:
        return {"original": self.original, "path": str(self.path), "language": self.language}


def infer_language(path: Path) -> str:
    """Best-effort language tag from a file extension."""
    return _LANGUAGE_BY_EXT.get(path.suffix.lower(), "unknown")


def _candidate_paths(raw: str, base_dir: Optional[Path]) -> List[Path]:
    """Every reasonable interpretation of ``raw``, in priority order."""
    expanded = Path(raw).expanduser()
    candidates: List[Path] = []

    # 1. Absolute / user-expanded path, exactly as given.
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        # 2. Relative to an explicit base directory (e.g. the blueprint's dir).
        if base_dir is not None:
            candidates.append((Path(base_dir) / expanded))
        # 3. Relative to the current working directory.
        candidates.append(Path.cwd() / expanded)
    # 4. The literal path, resolved against cwd by Path itself (last resort).
    candidates.append(expanded)

    # De-duplicate while preserving order.
    seen: set = set()
    unique: List[Path] = []
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_source_entry(raw_path: str, base_dir: Optional[Path] = None) -> SourceEntry:
    """Resolve ``raw_path`` from anywhere on the filesystem.

    Tries the path as absolute, relative to ``base_dir``, and relative to the
    current working directory.  Raises :class:`SourceEntryNotFound` only when the
    file exists in none of those forms -- never merely because it sits outside
    the tool's own directory.
    """
    if not raw_path:
        raise SourceEntryNotFound("empty source_entry path")
    for candidate in _candidate_paths(raw_path, base_dir):
        if candidate.is_file():
            return SourceEntry(original=raw_path, path=candidate, language=infer_language(candidate))
    tried = ", ".join(str(p) for p in _candidate_paths(raw_path, base_dir))
    raise SourceEntryNotFound(
        f"source_entry '{raw_path}' not found in any location (tried: {tried})"
    )


def copy_into_workspace(entry: SourceEntry, destination: Path, content: Optional[str] = None) -> Path:
    """Copy (or write transformed ``content`` of) a resolved entry to ``destination``.

    The destination is always inside the transient/out-of-tree workspace; the
    original file is only ever read, never moved or modified.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        destination.write_text(content, encoding="utf-8")
    else:
        shutil.copy2(entry.path, destination)
    return destination
