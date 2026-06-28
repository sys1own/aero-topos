"""
Enhanced-Precision Selector.

Decides which floating-point precision a code zone should use and emits the
matching compiler flags + type mappings for Rust, C/C++, and Python.  An
optional heuristic pass scans the Unified AST for operations that typically
need more precision (iterative solvers, transcendental functions, large
condition numbers) and recommends promoting them.

Precision tiers
---------------
* ``double``     -> IEEE binary64 (``f64`` / ``double`` / Python ``float``).
* ``quad``       -> IEEE binary128 (``__float128`` + libquadmath / ``f128`` crate
                    / Python ``mpmath`` at 113 bits).
* ``arbitrary``  -> GMP/MPFR-backed arbitrary precision at ``N`` bits
                    (``rug``/``num-bigint`` for Rust, ``gmpy2``/``mpmath`` for
                    Python).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.serialization import dataclass_to_dict


@dataclass
class PrecisionRecommendation:
    location: str
    zone: str
    current: str
    recommended: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


# Heuristic signals that an operation benefits from higher precision.
_TRANSCENDENTAL = {"exp", "log", "log10", "log2", "pow", "sqrt", "sin", "cos",
                   "tan", "sinh", "cosh", "tanh", "expm1", "erf", "gamma"}
_ITERATIVE_HINTS = ("solve", "solver", "iterate", "newton", "gmres", "bicg",
                    "conjugate_gradient", "_cg", "jacobi", "gauss_seidel",
                    "relax", "converge", "fixed_point", "krylov")
_CONDITIONING_HINTS = ("inverse", "inv(", "matmul", "matrix_mul", "determinant",
                       "det(", "eig", "eigen", "lu_decomp", "cholesky", "svd",
                       "condition_number", "linsolve")


class PrecisionSelector:
    """Selects floating-point precision per zone and emits build settings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        ps = self.config.get("precision_shield", {}) or {}
        self.default_float = str(ps.get("default_float", "double")).lower()
        self.arbitrary_bits = int(ps.get("arbitrary_precision_bits", 128))
        raw_overrides = ps.get("per_zone_overrides", {}) or {}
        self.per_zone_overrides: Dict[str, str] = {}
        for zone, spec in raw_overrides.items():
            if isinstance(spec, dict):
                self.per_zone_overrides[zone] = str(spec.get("float", self.default_float)).lower()
            else:
                self.per_zone_overrides[zone] = str(spec).lower()
        self.auto_detect_need = bool(ps.get("auto_detect_need", False))

    # ------------------------------------------------------------------
    # Zone selection
    # ------------------------------------------------------------------

    def float_for_zone(self, zone_id: str) -> str:
        return self.per_zone_overrides.get(zone_id, self.default_float)

    def recommend(self, mapper_or_graph: Any = None) -> Dict[str, str]:
        """Return a {zone-or-file -> float kind} map.

        Combines explicit per-zone overrides with auto-detected promotions.
        """
        decisions: Dict[str, str] = dict(self.per_zone_overrides)
        if self.auto_detect_need and mapper_or_graph is not None:
            for rec in self.analyze_uast(mapper_or_graph):
                # Don't downgrade an explicit override.
                decisions.setdefault(rec.location, rec.recommended)
        return decisions

    # ------------------------------------------------------------------
    # Heuristic UAST analysis
    # ------------------------------------------------------------------

    def analyze_uast(self, mapper_or_graph: Any) -> List[PrecisionRecommendation]:
        """Flag UAST regions that likely need higher precision."""
        graph = getattr(mapper_or_graph, "uast", mapper_or_graph)
        promote_to = "quad" if self.default_float == "double" else self.default_float
        seen: Dict[str, PrecisionRecommendation] = {}

        for _, data in graph.nodes(data=True):
            meta = data.get("metadata", {})
            name = (meta.get("name") or "").lower()
            payload = data.get("data", {})
            text = (payload.get("source") or payload.get("text") or "").lower()
            loc = data.get("source_location") or ["<unknown>"]
            file_key = str(loc[0])

            reason = None
            if meta.get("uast_kind") == "uast_call" and name in _TRANSCENDENTAL:
                reason = f"transcendental call '{name}' accumulates rounding error"
            elif any(hint in name or hint in text for hint in _ITERATIVE_HINTS):
                reason = "iterative solver pattern is sensitive to precision"
            elif any(hint in name or hint in text for hint in _CONDITIONING_HINTS):
                reason = "linear-algebra op may be ill-conditioned"

            if reason and file_key not in seen:
                seen[file_key] = PrecisionRecommendation(
                    location=file_key,
                    zone=file_key,
                    current=self.default_float,
                    recommended=promote_to,
                    reason=reason,
                )
        return list(seen.values())

    # ------------------------------------------------------------------
    # Compiler flags / type mappings
    # ------------------------------------------------------------------

    def compiler_flags(self, language: str, float_kind: Optional[str] = None) -> List[str]:
        kind = (float_kind or self.default_float).lower()
        lang = language.lower()

        if lang in ("c", "cpp", "c++"):
            if kind == "double":
                return []
            if kind == "quad":
                return ["-DAERO_FLOAT=__float128", "-lquadmath"]
            if kind == "arbitrary":
                return [f"-DAERO_PREC_BITS={self.arbitrary_bits}", "-lgmp", "-lgmpxx", "-lmpfr"]
        if lang == "rust":
            if kind == "double":
                return []
            if kind == "quad":
                return ["--cfg", "aero_quad"]            # gated on the `f128` crate
            if kind == "arbitrary":
                return ["--cfg", f"aero_arbitrary_bits=\"{self.arbitrary_bits}\""]
        if lang == "python":
            return []  # precision handled via libraries, not flags
        return []

    def type_mapping(self, language: str, float_kind: Optional[str] = None) -> Dict[str, Any]:
        kind = (float_kind or self.default_float).lower()
        lang = language.lower()
        table = {
            ("c", "double"): {"type": "double", "headers": []},
            ("c", "quad"): {"type": "__float128", "headers": ["quadmath.h"]},
            ("c", "arbitrary"): {"type": "mpf_t", "headers": ["gmp.h"]},
            ("cpp", "double"): {"type": "double", "headers": []},
            ("cpp", "quad"): {"type": "__float128", "headers": ["quadmath.h"]},
            ("cpp", "arbitrary"): {"type": "mpf_class", "headers": ["gmpxx.h"]},
            ("rust", "double"): {"type": "f64", "crates": []},
            ("rust", "quad"): {"type": "f128::f128", "crates": ["f128"]},
            ("rust", "arbitrary"): {"type": "rug::Float", "crates": ["rug"]},
            ("python", "double"): {"type": "float", "modules": []},
            ("python", "quad"): {"type": "mpmath.mpf", "modules": ["mpmath"]},
            ("python", "arbitrary"): {"type": "gmpy2.mpfr", "modules": ["gmpy2"]},
        }
        lang_key = "cpp" if lang in ("cpp", "c++") else lang
        return table.get((lang_key, kind), {"type": "double"})

    def required_dependencies(self, float_kind: Optional[str] = None) -> Dict[str, List[str]]:
        """External packages needed for the chosen precision tier."""
        kind = (float_kind or self.default_float).lower()
        if kind == "double":
            return {}
        if kind == "quad":
            return {"system": ["libquadmath"], "rust": ["f128"], "python": ["mpmath"]}
        return {
            "system": ["libgmp", "libmpfr"],
            "rust": ["rug"],
            "python": ["gmpy2"],
        }

    # ------------------------------------------------------------------
    # Precision-aware SMT tolerance
    # ------------------------------------------------------------------

    def epsilon_for(self, float_kind: Optional[str] = None) -> float:
        """Machine epsilon used to make SMT validation precision-aware."""
        kind = (float_kind or self.default_float).lower()
        if kind == "double":
            return 2.0 ** -52
        if kind == "quad":
            return 2.0 ** -112
        return 2.0 ** -(self.arbitrary_bits - 1)
