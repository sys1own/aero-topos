"""
``DAGInferenceEngine`` -- turns a few lines of semantic intent into a full,
executable build graph by inspecting the project directory.

Given a :class:`~src.invisible_config.lean_parser.LeanBlueprint` and a project
root, it:

1. locates each declared target's source files in the file tree and infers its
   **language** (C++, Rust, Python, ...);
2. infers the **execution DAG** -- compiled "core" targets depend on the text
   *invariants* extracted from the ``ingest`` files (the same Invariant Schema
   the Semantic Fluidity engine produces), and dynamic targets (e.g. a Python
   dashboard) depend on the compiled cores they bind to;
3. maps the **FFI / language boundaries** between those targets (pybind11 /
   ctypes for C++<->Python, PyO3 for Rust<->Python), so glue code can be
   error-corrected automatically later;
4. emits a normalized ``build_context`` that plugs straight into the existing
   core execution system, so ``aero`` needs zero further input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.invisible_config.lean_parser import LeanBlueprint

# Language inference -------------------------------------------------------

# Source extensions per language (used both to locate sources and to infer the
# language of a target when its name is ambiguous).
_LANG_EXTENSIONS = {
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h"},
    "c": {".c", ".h"},
    "rust": {".rs"},
    "python": {".py", ".pyi"},
    "fortran": {".f90", ".f", ".f95"},
}

# Hints in a target *name* that strongly imply a language.
_NAME_LANGUAGE_HINTS = [
    ("cpp", ("cpp", "cxx", "cplus", "cc")),
    ("rust", ("rust", "_rs", "rs_", "cargo")),
    ("python", ("python", "py_", "_py", "dashboard", "ui", "web", "flask", "django", "panel", "notebook")),
    ("fortran", ("fortran", "f90")),
    ("c", ("c_core", "clang", "_c_")),
]

# Languages that compile to a native artifact other targets can FFI-link to.
_COMPILED_LANGUAGES = {"cpp", "c", "rust", "fortran"}
# Languages that act as dynamic "driver"/binding layers over the compiled cores.
_DYNAMIC_LANGUAGES = {"python"}

# FFI mechanism per (provider_language -> consumer_language=python) pair.
_FFI_MECHANISM = {
    ("cpp", "python"): "pybind11",
    ("c", "python"): "ctypes",
    ("rust", "python"): "pyo3",
    ("fortran", "python"): "ctypes",
}

_IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".aero", "build_artifacts"}

# The synthetic node representing the text invariants extracted from `ingest`.
INVARIANTS_NODE = "text_invariants"


@dataclass
class InferredTarget:
    name: str
    language: str
    role: str  # "core" | "binding" | "library"
    sources: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    language_reason: str = ""  # why this language was chosen (for `infer` / debug)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language,
            "role": self.role,
            "sources": list(self.sources),
            "depends_on": list(self.depends_on),
            "language_reason": self.language_reason,
        }


@dataclass
class FfiBoundary:
    """A language boundary where glue code is generated and may need healing."""

    provider: str  # target producing the native symbols
    consumer: str  # target calling across the boundary
    mechanism: str  # pybind11 | ctypes | pyo3 | ...
    provider_language: str = ""
    consumer_language: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "consumer": self.consumer,
            "mechanism": self.mechanism,
            "provider_language": self.provider_language,
            "consumer_language": self.consumer_language,
        }


@dataclass
class InferredDAG:
    project: str
    targets: List[InferredTarget] = field(default_factory=list)
    ffi_boundaries: List[FfiBoundary] = field(default_factory=list)
    ingest: List[str] = field(default_factory=list)
    optimize: str = "balanced"
    has_invariants: bool = False

    # ------------------------------------------------------------------

    def dependency_matrix(self) -> Dict[str, List[str]]:
        return {t.name: list(t.depends_on) for t in self.targets}

    def topological_order(self) -> List[str]:
        """Return target names in dependency order (excludes the invariants node)."""
        deps = {t.name: set(d for d in t.depends_on if d != INVARIANTS_NODE) for t in self.targets}
        order: List[str] = []
        satisfied: Set[str] = set()
        remaining = dict(deps)
        # Kahn's algorithm with deterministic ordering.
        while remaining:
            ready = sorted(name for name, d in remaining.items() if d <= satisfied)
            if not ready:
                # A cycle (or dangling dep): append the rest deterministically so
                # the caller still gets a usable order rather than an exception.
                ready = sorted(remaining)
            for name in ready:
                order.append(name)
                satisfied.add(name)
                remaining.pop(name, None)
        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "optimize": self.optimize,
            "ingest": list(self.ingest),
            "has_invariants": self.has_invariants,
            "invariants_node": INVARIANTS_NODE if self.has_invariants else None,
            "targets": [t.to_dict() for t in self.targets],
            "ffi_boundaries": [b.to_dict() for b in self.ffi_boundaries],
            "dependency_matrix": self.dependency_matrix(),
            "execution_order": self.topological_order(),
        }


class DAGInferenceEngine:
    """Infers a full build graph from a lean blueprint + the project file tree."""

    def __init__(self, blueprint: LeanBlueprint, project_root: Path) -> None:
        self.blueprint = blueprint
        self.project_root = Path(project_root)
        self._file_index: Optional[List[Path]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self) -> InferredDAG:
        has_invariants = bool(self.blueprint.ingest)
        targets = [self._infer_target(name) for name in self.blueprint.targets]

        # 1. Compiled cores depend on the extracted text invariants.
        if has_invariants:
            for target in targets:
                if target.role == "core":
                    self._add_dep(target, INVARIANTS_NODE)

        # 2. FFI boundaries: dynamic targets depend on the compiled cores.
        ffi_boundaries = self._infer_ffi_boundaries(targets)

        return InferredDAG(
            project=self.blueprint.project,
            targets=targets,
            ffi_boundaries=ffi_boundaries,
            ingest=list(self.blueprint.ingest),
            optimize=self.blueprint.optimize,
            has_invariants=has_invariants,
        )

    # ------------------------------------------------------------------
    # Target / language inference
    # ------------------------------------------------------------------

    def _infer_target(self, name: str) -> InferredTarget:
        language = self._infer_language(name)
        reason = ""
        if language != "unknown":
            reason = f"target name '{name}' matches a {language} naming hint"
        sources = self._locate_sources(name, language)
        # If the name implied nothing but sources exist, refine the language.
        if language == "unknown" and sources:
            language = self._language_from_paths(sources)
            if language != "unknown":
                reason = f"dominant source extension under '{name}' is {language}"
        if language == "unknown":
            language = "python"  # safest default: a scripting/driver layer
            reason = "no name hint or sources matched; defaulted to python (driver layer)"
        role = "core" if language in _COMPILED_LANGUAGES else "binding"
        return InferredTarget(
            name=name, language=language, role=role, sources=sources, language_reason=reason
        )

    def _infer_language(self, name: str) -> str:
        lowered = name.lower()
        for language, hints in _NAME_LANGUAGE_HINTS:
            if any(hint in lowered for hint in hints):
                return language
        return "unknown"

    def _locate_sources(self, name: str, language: str) -> List[str]:
        """Find candidate source files for a target by matching name tokens."""
        tokens = [tok for tok in name.lower().replace("-", "_").split("_") if tok]
        # Drop pure language tokens so "cpp_core" matches a "core/" directory.
        meaningful = [tok for tok in tokens if tok not in {"cpp", "rust", "python", "py", "c", "fortran"}]
        search_tokens = meaningful or tokens
        extensions = _LANG_EXTENSIONS.get(language, set())

        matches: List[str] = []
        for path in self._files():
            if extensions and path.suffix.lower() not in extensions:
                continue
            haystack = str(path.relative_to(self.project_root)).lower()
            if any(tok in haystack for tok in search_tokens):
                matches.append(str(path.relative_to(self.project_root)))
        return sorted(matches)

    def _language_from_paths(self, sources: List[str]) -> str:
        counts: Dict[str, int] = {}
        for source in sources:
            suffix = Path(source).suffix.lower()
            for language, extensions in _LANG_EXTENSIONS.items():
                if suffix in extensions:
                    counts[language] = counts.get(language, 0) + 1
        if not counts:
            return "unknown"
        return max(counts, key=counts.get)

    # ------------------------------------------------------------------
    # FFI boundary inference
    # ------------------------------------------------------------------

    def _infer_ffi_boundaries(self, targets: List[InferredTarget]) -> List[FfiBoundary]:
        cores = [t for t in targets if t.role == "core"]
        bindings = [t for t in targets if t.role == "binding"]
        boundaries: List[FfiBoundary] = []
        for consumer in bindings:
            for provider in cores:
                mechanism = _FFI_MECHANISM.get(
                    (provider.language, consumer.language), "cffi"
                )
                self._add_dep(consumer, provider.name)
                boundaries.append(
                    FfiBoundary(
                        provider=provider.name,
                        consumer=consumer.name,
                        mechanism=mechanism,
                        provider_language=provider.language,
                        consumer_language=consumer.language,
                    )
                )
        return boundaries

    @staticmethod
    def _add_dep(target: InferredTarget, dep: str) -> None:
        if dep != target.name and dep not in target.depends_on:
            target.depends_on.append(dep)

    # ------------------------------------------------------------------
    # File-tree scanning
    # ------------------------------------------------------------------

    def _files(self) -> List[Path]:
        if self._file_index is None:
            index: List[Path] = []
            if self.project_root.exists():
                for path in self.project_root.rglob("*"):
                    if not path.is_file():
                        continue
                    if any(part in _IGNORED_DIRS for part in path.parts):
                        continue
                    index.append(path)
            self._file_index = index
        return self._file_index
