"""
``PolymorphicRewriter`` -- the rewrite pass for Autonomous Hardware-Polymerization
(requirements #2, #3, #4).

It runs *after* code generation but *before* linking/execution, intercepting the
generated C/C++/Rust source (or LLVM IR) and rewriting it to fit the host the
:class:`~src.polymorphization.hardware_profiler.HardwareProfiler` discovered:

* **Memory alignment** -- the ``AERO_ALIGN`` placeholder (e.g. in
  ``alignas(AERO_ALIGN)`` or ``#[repr(align(AERO_ALIGN))]``) becomes the host's
  cache-line / vector-width aligned value.
* **Vectorised micro-kernels** -- a generic ``AERO_KERNEL(name)`` call is bound
  to the best available target-specific implementation (``name__avx512`` /
  ``name__avx2`` / ``name__neon`` / ``name__scalar``), and ``AERO_PRAGMA_SIMD``
  marker lines become the language-appropriate vectorisation directive.
* **Thread-pool sizing** -- ``AERO_WORKERS`` becomes the host's *physical* core
  count (not the logical/SMT count), plus vector-lane width placeholders.

The rewrite is purely textual and deterministic, so it can run **in memory**
(:meth:`rewrite_text`) or write rewritten copies into an **ephemeral build
cache** (:meth:`rewrite_tree`) -- it never mutates the user's primary source
directory (requirement #4).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.polymorphization.topology import HardwareTopology

# extension -> language tag
_LANG_BY_EXT = {
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rs": "rust",
    ".ll": "llvm",
}

# Rust target_feature names differ from our normalised ISA names.
_RUST_TARGET_FEATURE = {
    "avx512": "avx512f",
    "avx2": "avx2",
    "avx": "avx",
    "sse4_2": "sse4.2",
    "sse2": "sse2",
    "neon": "neon",
}

_KERNEL_CALL_RE = re.compile(r"\bAERO_KERNEL\s*\(\s*([A-Za-z_]\w*)\s*\)")
_PRAGMA_MARKER_RE = re.compile(r"^([ \t]*).*\bAERO_PRAGMA_SIMD\b.*$", re.MULTILINE)


@dataclass
class RewriteResult:
    """What a single rewrite produced and which substitutions fired."""

    language: str
    text: str
    changes: Dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(self.changes.values())

    def to_dict(self) -> Dict[str, object]:
        return {"language": self.language, "changed": self.changed, "changes": dict(self.changes)}


class PolymorphicRewriter:
    """Rewrites generated sources to match a :class:`HardwareTopology`."""

    def __init__(self, topology: HardwareTopology) -> None:
        self.topology = topology
        self.simd = topology.best_simd()
        self.alignment = topology.alignment_bytes()
        self.workers = max(1, topology.physical_cores)
        self.vector_width = topology.vector_width_bytes()
        # Scalar tokens substituted verbatim (word-boundary matched).
        self._tokens = {
            "AERO_ALIGN": str(self.alignment),
            "AERO_WORKERS": str(self.workers),
            "AERO_VECTOR_WIDTH": str(self.vector_width),
            "AERO_SIMD_LANES_F32": str(max(1, self.vector_width // 4)),
            "AERO_SIMD_LANES_F64": str(max(1, self.vector_width // 8)),
        }
        self._token_re = re.compile(r"\b(" + "|".join(map(re.escape, self._tokens)) + r")\b")

    # ------------------------------------------------------------------
    # In-memory rewriting (requirement #4: transparent, no source mutation)
    # ------------------------------------------------------------------

    def rewrite_text(self, source: str, language: str) -> RewriteResult:
        changes: Dict[str, int] = {
            "alignment": 0,
            "kernels": 0,
            "pragmas": 0,
            "workers": 0,
            "tokens": 0,
        }
        text = source

        # 1. Vectorised micro-kernel binding: AERO_KERNEL(dot) -> dot__avx2
        def _bind_kernel(match: re.Match) -> str:
            changes["kernels"] += 1
            return f"{match.group(1)}__{self.simd}"

        text = _KERNEL_CALL_RE.sub(_bind_kernel, text)

        # 2. Vectorisation pragma markers -> language-specific directive.
        def _emit_pragma(match: re.Match) -> str:
            changes["pragmas"] += 1
            indent = match.group(1)
            return indent + self._pragma_for(language)

        text = _PRAGMA_MARKER_RE.sub(_emit_pragma, text)

        # 3. Scalar placeholder tokens (alignment, workers, lane widths).
        def _sub_token(match: re.Match) -> str:
            token = match.group(1)
            changes["tokens"] += 1
            if token == "AERO_ALIGN":
                changes["alignment"] += 1
            elif token == "AERO_WORKERS":
                changes["workers"] += 1
            return self._tokens[token]

        text = self._token_re.sub(_sub_token, text)

        return RewriteResult(language=language, text=text, changes=changes)

    def _pragma_for(self, language: str) -> str:
        """The vectorisation directive to emit for the host's best ISA."""
        if self.simd == "scalar":
            if language == "rust":
                return "#[inline] // aero: no SIMD target available; scalar path"
            return "// aero: no SIMD target available; scalar path"
        if language == "rust":
            feature = _RUST_TARGET_FEATURE.get(self.simd, self.simd)
            return f'#[target_feature(enable = "{feature}")] // aero: vectorised for {self.simd}'
        if language == "llvm":
            lanes = max(1, self.vector_width // 4)
            return f"; aero: vectorize width {lanes} ({self.simd})"
        # C / C++: OpenMP SIMD is the portable directive; lane count from width.
        lanes = max(1, self.vector_width // 4)
        return f"#pragma omp simd simdlen({lanes}) // aero: vectorised for {self.simd}"

    # ------------------------------------------------------------------
    # Ephemeral build-cache rewriting
    # ------------------------------------------------------------------

    def rewrite_file(self, source_path: Path, cache_path: Path) -> Optional[RewriteResult]:
        """Rewrite one file from ``source_path`` into ``cache_path``.

        Returns ``None`` for unsupported extensions.  ``cache_path`` is always
        somewhere in the ephemeral cache -- the source file is only read.
        """
        language = _LANG_BY_EXT.get(source_path.suffix.lower())
        if language is None:
            return None
        source = source_path.read_text(encoding="utf-8", errors="ignore")
        result = self.rewrite_text(source, language)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result.text, encoding="utf-8")
        return result

    def rewrite_tree(self, source_dir: Path, cache_dir: Path) -> Dict[str, object]:
        """Rewrite every supported file under ``source_dir`` into ``cache_dir``.

        The source tree is treated as read-only; all output goes to the
        ephemeral ``cache_dir`` (requirement #4).  The relative layout is
        preserved so a downstream linker can compile the cache verbatim.
        """
        source_dir = Path(source_dir)
        cache_dir = Path(cache_dir)
        files: List[Dict[str, object]] = []
        rewritten = 0
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _LANG_BY_EXT:
                continue
            relative = path.relative_to(source_dir)
            cache_path = cache_dir / relative
            result = self.rewrite_file(path, cache_path)
            if result is None:
                continue
            if result.changed:
                rewritten += 1
            files.append({"source": str(path), "cache": str(cache_path), **result.to_dict()})
        return {
            "cache_dir": str(cache_dir),
            "files_processed": len(files),
            "files_rewritten": rewritten,
            "files": files,
        }

    @staticmethod
    def reset_cache(cache_dir: Path) -> None:
        """Remove an ephemeral cache directory so a build starts clean."""
        cache_dir = Path(cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
