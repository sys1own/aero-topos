"""
Dependency-free best-effort PDF text extraction.

If a real PDF library (``pypdf`` or ``PyPDF2``) happens to be installed it is
used automatically for higher fidelity.  Otherwise this module falls back to a
minimal, pure-stdlib extractor: it scans ``stream``/``endstream`` objects,
inflates ``FlateDecode`` streams with :mod:`zlib`, and pulls literal strings out
of the ``Tj``/``TJ`` text-showing operators.  This only handles simple,
non-encrypted PDFs with literal (parenthesised) text strings -- enough to ingest
text-based reports and papers without adding a new dependency.  PDFs that only
use hex strings, CID-keyed fonts, or that are encrypted will yield empty text;
callers should treat PDF ingestion as best-effort.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

_STREAM_RE = re.compile(rb"<<(.*?)>>\s*stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_TJ_ARRAY_RE = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
_TJ_LITERAL_RE = re.compile(rb"\((.*?)\)\s*Tj", re.DOTALL)
_TJ_ARRAY_STRING_RE = re.compile(rb"\((.*?)\)")

_ESCAPES = {
    b"\\n": b"\n",
    b"\\r": b"\r",
    b"\\t": b"\t",
    b"\\(": b"(",
    b"\\)": b")",
    b"\\\\": b"\\",
}


def _unescape(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i : i + 1] == b"\\" and i + 1 < len(raw):
            pair = raw[i : i + 2]
            if pair in _ESCAPES:
                out += _ESCAPES[pair]
                i += 2
                continue
            # Octal escape \ddd
            octal = re.match(rb"\\([0-7]{1,3})", raw[i:])
            if octal:
                out.append(int(octal.group(1), 8) & 0xFF)
                i += 1 + len(octal.group(1))
                continue
            i += 1
            continue
        out.append(raw[i])
        i += 1
    return out.decode("latin-1", errors="ignore")


def _extract_with_library(path: Path) -> str:
    try:
        import pypdf  # type: ignore
    except ImportError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError:
            return ""
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _extract_fallback(raw: bytes) -> str:
    chunks = []
    for match in _STREAM_RE.finditer(raw):
        header, body = match.group(1), match.group(2)
        if b"FlateDecode" in header:
            try:
                body = zlib.decompress(body)
            except zlib.error:
                continue
        for tj_match in _TJ_LITERAL_RE.finditer(body):
            chunks.append(_unescape(tj_match.group(1)))
        for array_match in _TJ_ARRAY_RE.finditer(body):
            for string_match in _TJ_ARRAY_STRING_RE.finditer(array_match.group(1)):
                chunks.append(_unescape(string_match.group(1)))
        chunks.append("\n")
    return " ".join(chunks)


def extract_text(path: Path) -> str:
    """Best-effort extraction of the visible text in a PDF file."""
    library_text = _extract_with_library(path)
    if library_text.strip():
        return library_text
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return _extract_fallback(raw)
