# -*- coding: utf-8 -*-
"""
First-class conditional Language Router for scaffold/build orchestration.

Reads ``blueprint["frameworks"]["language"]`` (with safe fallbacks) and routes
the pipeline to either the Rust Cargo path or the native Python workspace path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.scaffold.source_resolver import SourceEntry, infer_language
from pathlib import Path

SUPPORTED_LANGUAGES = frozenset({"rust", "python"})
DEFAULT_LANGUAGE = "rust"


def frameworks_section(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``frameworks`` section, tolerating missing or non-dict values."""
    section = context.get("frameworks")
    return dict(section) if isinstance(section, dict) else {}


def resolve_target_language(
    context: Dict[str, Any],
    *,
    source_entry: Optional[SourceEntry] = None,
    source_path: Optional[Path] = None,
) -> str:
    """Resolve the blueprint target language for routing.

    Priority:
    1. ``frameworks.language`` from the blueprint (explicit router input).
    2. Resolved :class:`SourceEntry` language tag.
    3. File-extension inference from ``source_path``.
    4. Conservative legacy default (``rust``).
    """
    declared = str(frameworks_section(context).get("language", "")).strip().lower()
    if declared in SUPPORTED_LANGUAGES:
        return declared

    if source_entry is not None and source_entry.language in SUPPORTED_LANGUAGES:
        return source_entry.language

    if source_path is not None:
        inferred = infer_language(Path(source_path))
        if inferred in SUPPORTED_LANGUAGES:
            return inferred

    return DEFAULT_LANGUAGE


def is_rust(language: str) -> bool:
    return language == "rust"


def is_python(language: str) -> bool:
    return language == "python"


def is_native_crate_language(language: str) -> bool:
    """Strict gate: only ``rust`` is a valid low-level cargo (crate) target.

    The scaffold/self-hosting engine restricts its ``cargo build`` path to
    genuine Rust crate components.  Any other resolved language (python is
    handled by its own workspace path; c/cpp/fortran/shell/unknown fall here)
    must bypass the cargo build loop rather than emit spurious Cargo errors.
    """
    return language == "rust"


def cargo_bypass_warning(language: str) -> str:
    """The route warning emitted when a non-Rust target skips the cargo layer."""
    return (
        f"language router -> '{language}' "
        "(Bypassing Rust cargo build layer; target is not a native crate component)"
    )


def layout_description(language: str) -> str:
    """Human-readable summary of the files synthesised for a language."""
    if is_python(language):
        return "pyproject.toml, entry script, .gitignore"
    return "Cargo.toml, src/lib.rs, .gitignore, test_binding.py"


def build_description(language: str) -> str:
    """Human-readable summary of the isolation build step."""
    if is_python(language):
        return "python bytecode validation (compileall / py_compile) in out-of-tree workspace"
    return "cargo build --release (all artifacts stay out-of-tree)"
