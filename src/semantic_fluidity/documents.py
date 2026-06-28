"""
Mixed-format document loading for the context ingestion engine.

Walks a directory of ``.txt``/``.md``/``.pdf``/``.json``/``.cpp``/``.py`` (and a
handful of related source extensions) files and turns each into a plain
:class:`IngestedDocument` carrying the raw text and a coarse ``format`` tag.
Per-file errors (unreadable, undecodable, corrupt PDF, ...) are recorded on the
document rather than raised, so one bad file never aborts ingestion of an
entire directory -- mirroring the error-isolation policy already used by
:class:`src.context.ingest.ContextIngestor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.semantic_fluidity.pdf_text import extract_text as extract_pdf_text

_IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".aero", "build_artifacts"}

# Extension -> coarse format tag.  Code formats are kept distinct per language
# so a code file and a prose file are never folded into the same domain bucket.
EXTENSION_FORMATS = {
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".json": "json",
    ".py": "code:python",
    ".pyi": "code:python",
    ".cpp": "code:cpp",
    ".cc": "code:cpp",
    ".cxx": "code:cpp",
    ".hpp": "code:cpp",
    ".hh": "code:cpp",
    ".c": "code:c",
    ".h": "code:c",
    ".pdf": "pdf",
}


@dataclass
class IngestedDocument:
    path: Path
    format: str
    text: str
    error: Optional[str] = None


class DocumentLoader:
    """Reads every supported file under a directory into :class:`IngestedDocument`."""

    def load_directory(self, directory: Path) -> List[IngestedDocument]:
        directory = Path(directory)
        documents: List[IngestedDocument] = []
        if not directory.exists():
            return documents
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            fmt = EXTENSION_FORMATS.get(path.suffix.lower())
            if fmt is None:
                continue
            documents.append(self._load_file(path, fmt))
        return documents

    def load_file(self, path: Path) -> Optional[IngestedDocument]:
        path = Path(path)
        fmt = EXTENSION_FORMATS.get(path.suffix.lower())
        if fmt is None:
            return None
        return self._load_file(path, fmt)

    @staticmethod
    def _load_file(path: Path, fmt: str) -> IngestedDocument:
        try:
            if fmt == "pdf":
                text = extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
            return IngestedDocument(path=path, format=fmt, text=text)
        except Exception as exc:  # noqa: BLE001 - per-file isolation
            return IngestedDocument(path=path, format=fmt, text="", error=f"{type(exc).__name__}: {exc}")
