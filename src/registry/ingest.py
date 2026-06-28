"""Context ingestion into the AST registry.

``ingest_context`` parses a source file (or directory tree) into a Universal AST
(UAST), computes a deterministic structural *semantic hash*, extracts the
functions/types it declares, stores the UAST + metadata in the
:class:`~src.registry.ast_db.ASTDatabase`, and registers the context in
``blueprint.aero``'s ``[context_registry]`` so the blueprint stays the single
source of truth for which contexts are active.

All parsing is delegated to the language-agnostic Tree-sitter engine in
:mod:`core.parser.universal`.  There are no language-specific parsing paths here:
Python, Rust, C/C++, Fortran, COBOL (and any future grammar) flow through the
same ``Psi: T_L -> U`` transform, and the comment/whitespace-insensitive
semantic hash is computed uniformly over the UAST's flat node array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.parser.universal import (
    LANGUAGE_BY_EXTENSION,
    UniversalParseError,
    detect_language,
    extract_symbols,
    parse_source,
    semantic_hash,
)
from src.registry.ast_db import ASTDatabase, ASTEntry


class IngestError(Exception):
    """Raised when a file cannot be parsed or the language is unsupported."""


@dataclass
class FileResult:
    path: str
    language: str
    semantic_hash: str
    functions: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    fallback: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class IngestResult:
    context_name: str
    files: List[FileResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    blueprint_updated: bool = False


# ---------------------------------------------------------------------------
# blueprint.aero [context_registry] update
# ---------------------------------------------------------------------------
def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update_context_registry(
    blueprint_path: Union[str, Path],
    name: str,
    path: str,
    language: str,
    *,
    fallback: bool = False,
) -> bool:
    """Register ``name`` under ``[context_registry]`` in the blueprint.

    Appends a ``[context_registry.<name>]`` sub-table when the context is not
    already present, preserving the rest of the file (and its comments).  When a
    file had to be bypassed during ingestion a ``fallback = true`` flag is
    recorded on the context entry.  Returns ``True`` if the blueprint was
    modified, ``False`` if the context already existed.
    """
    bp_path = Path(blueprint_path)
    existing_text = bp_path.read_text(encoding="utf-8") if bp_path.is_file() else ""

    # Detect whether the context is already registered.
    try:
        from src.blueprint import load_blueprint

        if bp_path.is_file() and name in load_blueprint(bp_path).context_registry:
            return False
    except Exception:
        # If the existing blueprint can't be parsed, fall through and append
        # anyway rather than silently dropping the registration.
        pass

    block_lines = [
        "",
        f"[context_registry.{name}]",
        f"path = {_toml_str(path)}",
        f"language = {_toml_str(language)}",
    ]
    if fallback:
        block_lines.append("fallback = true")
    block_lines.append("")

    new_text = existing_text
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(block_lines)

    bp_path.parent.mkdir(parents=True, exist_ok=True)
    bp_path.write_text(new_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _iter_source_files(root: Path, language: Optional[str]) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        ext_language = LANGUAGE_BY_EXTENSION.get(candidate.suffix.lower())
        if ext_language is None:
            continue
        if language is not None and ext_language != language:
            continue
        files.append(candidate)
    return files


def ingest_context(
    context_name: str,
    path: Union[str, Path],
    language: Optional[str] = None,
    *,
    db: Optional[ASTDatabase] = None,
    db_path: Optional[Union[str, Path]] = None,
    blueprint_path: Optional[Union[str, Path]] = None,
) -> IngestResult:
    """Ingest a file or directory tree into the registry under *context_name*.

    Parses each source file into a UAST via the universal engine, computes its
    semantic hash, stores the UAST/metadata in the registry DB, and registers the
    context in ``blueprint.aero``.

    Either ``db`` (an open :class:`ASTDatabase`) or ``db_path`` must resolve to a
    database; if neither is given, a default ``.aero/registry.db`` next to the
    target path is used.
    """
    target = Path(path)
    if not target.exists():
        raise IngestError(f"Path does not exist: {target}")

    owns_db = False
    if db is None:
        if db_path is None:
            base = target if target.is_dir() else target.parent
            db_path = base / ".aero" / "registry.db"
        db = ASTDatabase(db_path)
        owns_db = True

    result = IngestResult(context_name=context_name)
    any_fallback = False
    try:
        files = _iter_source_files(target, language)
        if not files:
            result.errors.append(f"No supported source files found under {target}")

        for file_path in files:
            file_language = language or detect_language(file_path)
            if file_language is None:
                result.errors.append(f"Unsupported file type: {file_path}")
                continue
            try:
                source = file_path.read_text(encoding="utf-8")
                uast = parse_source(source, file_language, file_path=str(file_path))
            except (UniversalParseError, OSError, UnicodeDecodeError) as exc:
                result.errors.append(f"{file_path}: {exc}")
                continue

            flags = uast["metadata"]["parser_flags"]
            fallback = bool(flags.get("fallback"))
            any_fallback = any_fallback or fallback
            if fallback:
                result.errors.append(
                    f"{file_path}: bypassed ({flags.get('fallback_reason')})"
                )

            symbols = extract_symbols(uast, source)
            digest = uast["semantic_hash"]
            entry = ASTEntry(
                context_name=context_name,
                semantic_hash=digest,
                language=file_language,
                path=str(file_path),
                ast=uast,  # the full UAST object, ready for the registry DB
                functions=symbols["functions"],
                types=symbols["types"],
            )
            db.upsert(entry)
            result.files.append(
                FileResult(
                    path=str(file_path),
                    language=file_language,
                    semantic_hash=digest,
                    functions=symbols["functions"],
                    types=symbols["types"],
                    fallback=fallback,
                    fallback_reason=flags.get("fallback_reason"),
                )
            )

        # Register the context in the blueprint (path + dominant language).
        if blueprint_path is not None and result.files:
            registry_language = language or result.files[0].language
            result.blueprint_updated = update_context_registry(
                blueprint_path,
                context_name,
                str(target),
                registry_language,
                fallback=any_fallback,
            )
    finally:
        if owns_db:
            db.close()

    return result
