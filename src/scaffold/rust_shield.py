# -*- coding: utf-8 -*-
"""
Semantic shields & auto-error correction for Rust sources that use ``rug`` /
``pyo3``.

These are the exact API adjustments discovered while compiling the legacy
high-precision Anyon Simulator against modern crate versions, codified as
deterministic, idempotent source transforms applied during the scaffolding
(parsing) phase when ``rug`` or ``pyo3`` anchors are present:

* **Hygienic extension-trait injection** -- ``neg_mut`` / ``nth_root`` are no
  longer inherent methods on ``rug::Float`` / ``rug::Complex``; compatibility
  traits are prepended *after* any crate-level inner attributes so they never
  collide with downstream ``use`` imports.
* **Type-cascading alignment** -- ``let q_dim = match sec { ... }`` indexing
  arrays are annotated ``let q_dim: usize = match sec { ... }`` to stop a single
  inferred ``i32`` cascading into architecture mismatches downstream.
* **Mutability recovery** -- a ``cannot borrow `x` as mutable`` failure is
  repaired by upgrading ``let x`` to ``let mut x`` (used by the diagnostic
  recovery loop, see :mod:`src.scaffold.recovery`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

# Crate anchors that trigger Rust shielding.
RUST_ANCHORS = ("rug", "pyo3")

# Named compatibility shims referenced from blueprint [scaffold] configuration.
COMPATIBILITY_SHIMS = {
    "rug_v1_30_patch": "extension-traits(neg_mut,nth_root)",
    "pyo3_usize_alignment": "type-cascade-alignment(usize)",
}

# A sentinel so trait injection is idempotent across repeated passes.
_TRAIT_SENTINEL = "AeroNegMutExt"

# The exact compatibility traits codified from live compile testing.
EXTENSION_TRAITS = """\
// --- Aero compatibility shims (auto-injected for rug/pyo3) ---
trait AeroNegMutExt { fn neg_mut(&mut self); }
impl AeroNegMutExt for rug::Float {
    #[inline] fn neg_mut(&mut self) { let c = -self.clone(); <rug::Float as rug::Assign>::assign(self, c); }
}
impl AeroNegMutExt for rug::Complex {
    #[inline] fn neg_mut(&mut self) { let c = -self.clone(); <rug::Complex as rug::Assign>::assign(self, c); }
}
trait AeroNthRootExt { fn nth_root(&self, n: u32) -> rug::Float; }
impl AeroNthRootExt for rug::Float {
    #[inline] fn nth_root(&self, n: u32) -> rug::Float { rug::Float::with_val(self.prec(), self.clone().root(n)) }
}
// --- end Aero compatibility shims ---
"""

# `let <name> = match <expr> {` with no existing type annotation.
_MATCH_ASSIGN_RE = re.compile(
    r"(?P<indent>[ \t]*)let[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*=[ \t]*match\b",
)
# `cannot borrow `x` as mutable` / `cannot borrow `x` as mutable, as it is not declared as mutable`
_BORROW_MUT_RE = re.compile(r"cannot borrow `(?P<name>[A-Za-z_]\w*)` as mutable")
# `x` does not need to be mutable -> we leave those alone (warnings, not errors).


@dataclass
class ShieldReport:
    """The result of shielding a Rust source."""

    source: str
    anchors: Set[str] = field(default_factory=set)
    applied: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def to_dict(self) -> dict:
        return {"anchors": sorted(self.anchors), "applied": list(self.applied), "changed": self.changed}


class RustSemanticShield:
    """Applies the codified rug/pyo3 compatibility fixes to Rust source."""

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_anchors(self, source: str) -> Set[str]:
        found: Set[str] = set()
        for anchor in RUST_ANCHORS:
            # match `use rug`, `rug::`, `extern crate rug`, etc.
            if re.search(rf"\b{re.escape(anchor)}\b\s*(::|;)", source) or re.search(
                rf"\b(?:use|extern crate)\s+{re.escape(anchor)}\b", source
            ):
                found.add(anchor)
        return found

    def needs_shielding(self, source: str) -> bool:
        return bool(self.detect_anchors(source))

    # ------------------------------------------------------------------
    # Pre-emptive shields (applied during the parsing/scaffold phase)
    # ------------------------------------------------------------------

    def apply(
        self,
        source: str,
        compatibility_shims: Optional[List[str]] = None,
    ) -> ShieldReport:
        """Apply pre-emptive shields, idempotently.

        When ``compatibility_shims`` is provided (from a blueprint ``[scaffold]``
        section), only the named shims are applied.  Otherwise the legacy
        auto-detection path applies all relevant fixes when ``rug`` / ``pyo3``
        anchors are present.
        """
        anchors = self.detect_anchors(source)
        report = ShieldReport(source=source, anchors=anchors)

        if compatibility_shims is not None:
            if not compatibility_shims:
                return report
            if "rug_v1_30_patch" in compatibility_shims:
                source, injected = self.inject_extension_traits(source)
                if injected:
                    report.applied.append(COMPATIBILITY_SHIMS["rug_v1_30_patch"])
            if "pyo3_usize_alignment" in compatibility_shims:
                source, aligned = self.align_match_types(source)
                if aligned:
                    report.applied.append(
                        f"{COMPATIBILITY_SHIMS['pyo3_usize_alignment']} x{aligned}"
                    )
            report.source = source
            return report

        if not anchors:
            return report

        if "rug" in anchors:
            source, injected = self.inject_extension_traits(source)
            if injected:
                report.applied.append("extension-traits(neg_mut,nth_root)")

        source, aligned = self.align_match_types(source)
        if aligned:
            report.applied.append(f"type-cascade-alignment(usize x{aligned})")

        if "pyo3" in anchors:
            source, expanded = self.expand_pymodule_globs(source)
            if expanded:
                report.applied.append("pymodule-glob-expansion")

        report.source = source
        return report

    def expand_pymodule_globs(self, source: str) -> Tuple[str, bool]:
        """Rewrite ``use super::*;`` inside a ``#[pymodule]`` into named imports.

        PyO3's declarative ``#[pymodule]`` macro rejects glob re-exports, so the
        wildcard is expanded pre-emptively into an explicit named import map of
        every top-level structure / enum / function.  Delegates to the
        structural self-healing matrix so there is one source of truth for the
        Tree-sitter rewrite.  Idempotent and a no-op when Tree-sitter is
        unavailable or no glob is present.
        """
        try:
            from core.toolchain.self_healing import Diagnostic, heal_pyo3_glob_import

            data = source.encode("utf-8")
            diag = Diagnostic("#[pymodule] cannot import glob statements", "lib.rs")
            edits = heal_pyo3_glob_import(data, diag, "rust")
            if not edits:
                return source, False
            for edit in sorted(edits, key=lambda e: e.start, reverse=True):
                data = data[: edit.start] + edit.text.encode("utf-8") + data[edit.end :]
            return data.decode("utf-8"), True
        except Exception:
            # Tree-sitter not installed / parse failure -> leave source untouched.
            return source, False

    def inject_extension_traits(self, source: str) -> Tuple[str, bool]:
        """Prepend the compatibility traits, after crate-level inner attributes.

        Idempotent: if the traits are already present the source is returned
        unchanged.  Inner attributes (``#![...]``) and inner doc comments
        (``//!``) must stay at the very top of the file, so the shims are
        inserted *after* that leading block to remain hygienic.
        """
        if _TRAIT_SENTINEL in source:
            return source, False

        lines = source.splitlines(keepends=True)
        insert_at = 0
        for index, raw in enumerate(lines):
            stripped = raw.strip()
            # Leading block that must remain first: blanks, inner attrs/docs,
            # and ordinary comments at the very top of the file.
            if (
                stripped == ""
                or stripped.startswith("#![")
                or stripped.startswith("//!")
                or stripped.startswith("//")
            ):
                insert_at = index + 1
                continue
            break

        block = ("\n" if insert_at > 0 else "") + EXTENSION_TRAITS + "\n"
        new_source = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
        return new_source, True

    def align_match_types(self, source: str) -> Tuple[str, int]:
        """Annotate ``let x = match … {`` index assignments as ``: usize``.

        Only applies when the match arms are predominantly integer literals (the
        signature of a dimension/index table), so non-integer matches are left
        untouched.
        """
        count = 0
        out: List[str] = []
        pos = 0
        for match in _MATCH_ASSIGN_RE.finditer(source):
            block_end = _find_block_end(source, match.end())
            arms = source[match.end():block_end] if block_end > match.end() else ""
            if not _arms_are_integer_like(arms):
                continue
            # Rewrite `let name =` -> `let name: usize =` for this occurrence.
            out.append(source[pos:match.start()])
            out.append(f"{match.group('indent')}let {match.group('name')}: usize = match")
            pos = match.end()
            count += 1
        out.append(source[pos:])
        return ("".join(out), count) if count else (source, 0)

    # ------------------------------------------------------------------
    # Recovery shields (applied by the diagnostic recovery loop on failure)
    # ------------------------------------------------------------------

    def fix_mutability(self, source: str, diagnostics: str) -> Tuple[str, List[str]]:
        """Upgrade ``let x`` to ``let mut x`` for every borrow-as-mutable error."""
        applied: List[str] = []
        names = {m.group("name") for m in _BORROW_MUT_RE.finditer(diagnostics)}
        for name in sorted(names):
            pattern = re.compile(rf"\blet[ \t]+({re.escape(name)})\b(?![ \t]+mut\b)")
            new_source, n = pattern.subn(rf"let mut \1", source)
            if n:
                source = new_source
                applied.append(f"mut({name})")
        return source, applied

    def correct_from_diagnostics(self, source: str, diagnostics: str) -> Tuple[str, List[str]]:
        """Run the recovery corrections implied by a compiler diagnostic."""
        applied: List[str] = []
        source, mut_fixes = self.fix_mutability(source, diagnostics)
        applied.extend(mut_fixes)
        # A leftover type mismatch on an index variable -> force the usize cascade.
        if re.search(r"expected `?usize`?|mismatched types", diagnostics):
            source, aligned = self.align_match_types(source)
            if aligned:
                applied.append(f"type-cascade-alignment(usize x{aligned})")
        return source, applied


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_block_end(source: str, open_search_from: int) -> int:
    """Return the index just after the ``{ … }`` block that follows a match.

    ``open_search_from`` points just past the ``match`` keyword; we find the
    first ``{`` and scan to its balanced ``}``.
    """
    brace = source.find("{", open_search_from)
    if brace == -1:
        return -1
    depth = 0
    for i in range(brace, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _arms_are_integer_like(arms: str) -> bool:
    """True if the match arms look like an integer/index table."""
    results = re.findall(r"=>\s*([^,\n}]+)", arms)
    if not results:
        return False
    integer_like = 0
    for result in results:
        token = result.strip().rstrip(",").strip()
        if re.fullmatch(r"-?\d+(usize|u32|u64|i32|i64)?", token) or token.endswith("as usize"):
            integer_like += 1
    return integer_like >= max(1, len(results) // 2 + len(results) % 2)
