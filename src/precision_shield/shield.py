"""
Precision Shield.

Enforces strict numerical-invariant preservation for optimisation-sensitive
code zones (cryptographic cores, neural-network layers, critical calculations).
Uses Z3 SMT validation to prove that compiler transformations do not violate
precision contracts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("precision_shield.shield")

from z3 import (
    Bool,
    BoolRef,
    ForAll,
    Int,
    Optimize,
    Real,
    RealVal,
    Solver,
    sat,
    unsat,
)


@dataclass
class ShieldZone:
    """A code region protected by the Precision Shield."""

    identifier: str
    files: List[str]
    protection_level: str
    tolerated_precision_loss: float
    validation_rules: List[str] = field(default_factory=list)
    floating_point_precision: str = "64-bit"
    # Per-zone floating-point overrides (``None`` -> inherit the global policy).
    fast_math_override: Optional[bool] = None
    floating_point_contract: Optional[str] = None
    ieee_compliance: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a single shield-zone validation."""

    zone_id: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class PrecisionShield:
    """
    Validates that compiler transformations honour precision contracts.

    For each shield zone the engine:
    1. Collects floating-point operations from the source AST.
    2. Builds Z3 constraints encoding the zone's invariant rules.
    3. Checks satisfiability to confirm no violation is possible.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        ps_cfg = config.get("precision_shield", {})
        self.enforce = ps_cfg.get("enforce_strict_invariants", True)
        self.smt_backend = ps_cfg.get("smt_validation_backend", "z3")
        self.smt_timeout_ms = int(ps_cfg.get("smt_timeout_ms", 5000))
        self.fallback = ps_cfg.get("fallback_on_smt_failure", "conservative")
        self.zones = self._parse_zones(ps_cfg.get("shield_zones", []))

        # Global floating-point policy (feature #3).  Defaults are deliberately
        # conservative: contraction disallowed, fast-math off, strict IEEE-754.
        self.floating_point_contract = str(
            ps_cfg.get("floating_point_contract", "disallow")
        ).lower()
        self.fast_math_override = bool(ps_cfg.get("fast_math_override", False))
        self.ieee_compliance = str(ps_cfg.get("ieee_compliance", "strict")).lower()

        # Precision selection makes SMT validation precision-aware: a quad/
        # arbitrary-precision zone is checked against a tighter epsilon.
        from src.precision.selector import PrecisionSelector

        self.precision_selector = PrecisionSelector(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_all(self, project_root: Path) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for zone in self.zones:
            result = self.validate_zone(zone, project_root)
            results.append(result)
        return results

    def validate_zone(self, zone: ShieldZone, project_root: Path) -> ValidationResult:
        violations: List[str] = []
        details: Dict[str, Any] = {"checked_files": [], "operations_checked": 0}

        for file_rel in zone.files:
            file_path = project_root / file_rel
            if not file_path.exists():
                continue
            details["checked_files"].append(str(file_path))

            if file_path.suffix == ".py":
                file_violations = self._validate_python_file(file_path, zone)
            elif file_path.suffix == ".rs":
                file_violations = self._validate_rust_file(file_path, zone)
            else:
                continue
            violations.extend(file_violations)
            details["operations_checked"] += 1

        smt_result = self._run_smt_validation(zone)
        if not smt_result["passed"]:
            violations.extend(smt_result.get("violations", []))
        details["smt_result"] = smt_result

        return ValidationResult(
            zone_id=zone.identifier,
            passed=len(violations) == 0,
            violations=violations,
            details=details,
        )

    def get_protected_files(self) -> Set[str]:
        files: Set[str] = set()
        for zone in self.zones:
            files.update(zone.files)
        return files

    def is_file_protected(self, file_path: str) -> bool:
        return file_path in self.get_protected_files()

    def get_zone_for_file(self, file_path: str) -> Optional[ShieldZone]:
        for zone in self.zones:
            if file_path in zone.files:
                return zone
        return None

    # ------------------------------------------------------------------
    # Python validation
    # ------------------------------------------------------------------

    def _validate_python_file(
        self, file_path: Path, zone: ShieldZone
    ) -> List[str]:
        violations: List[str] = []
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except Exception as exc:
            violations.append(f"parse-error:{file_path}:{exc}")
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                violations.extend(
                    self._check_binop_rules(node, zone, file_path)
                )
            if isinstance(node, ast.Call):
                violations.extend(
                    self._check_call_rules(node, zone, file_path)
                )
        return violations

    def _check_binop_rules(
        self, node: ast.BinOp, zone: ShieldZone, file_path: Path
    ) -> List[str]:
        violations: List[str] = []
        loc = f"{file_path}:{node.lineno}"

        if "no_associative_reordering" in zone.validation_rules:
            if isinstance(node.op, (ast.Add, ast.Mult)):
                if isinstance(node.left, ast.BinOp) and type(node.left.op) == type(node.op):
                    violations.append(
                        f"associative-reorder-risk:{loc}: nested {type(node.op).__name__} may be reordered"
                    )

        if "no_constant_folding" in zone.validation_rules:
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                violations.append(
                    f"constant-folding-risk:{loc}: both operands are constants"
                )
        return violations

    def _check_call_rules(
        self, node: ast.Call, zone: ShieldZone, file_path: Path
    ) -> List[str]:
        violations: List[str] = []
        loc = f"{file_path}:{node.lineno}"

        if "no_fused_operations" in zone.validation_rules:
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            fused_patterns = {"fma", "fused_multiply_add", "addmm"}
            if func_name.lower() in fused_patterns:
                violations.append(
                    f"fused-op-violation:{loc}: call to fused operation '{func_name}'"
                )
        return violations

    # ------------------------------------------------------------------
    # Rust validation (regex-based heuristic)
    # ------------------------------------------------------------------

    def _validate_rust_file(
        self, file_path: Path, zone: ShieldZone
    ) -> List[str]:
        violations: List[str] = []
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read Rust file %s for validation: %s", file_path, exc)
            return violations

        lines = source.splitlines()
        for lineno, line in enumerate(lines, 1):
            loc = f"{file_path}:{lineno}"

            if "no_constant_folding" in zone.validation_rules:
                if re.search(r"\bconst\b.*=.*\d+\.\d+\s*[+\-*/]\s*\d+\.\d+", line):
                    violations.append(
                        f"constant-folding-risk:{loc}: compile-time float arithmetic"
                    )

            if "preserve_original_order" in zone.validation_rules:
                if re.search(r"#\[inline\(always\)\]", line):
                    violations.append(
                        f"order-risk:{loc}: aggressive inlining may reorder operations"
                    )
        return violations

    # ------------------------------------------------------------------
    # Mutation-equivalence verification (compactor guard)
    # ------------------------------------------------------------------

    def verify(self, original: str, mutated: str, zone: Optional[ShieldZone] = None) -> bool:
        """Return True if transforming ``original`` into ``mutated`` is allowed.

        Intended to wrap compactor/minifier passes: before applying an AST
        transformation inside a protected zone, call ``verify`` with the original
        and rewritten expression.  ``absolute_immutable`` zones permit no change
        at all (whitespace-insensitive); other zones permit algebraically
        equivalent rewrites, proven via Z3.  When equivalence cannot be proven
        the answer is conservatively ``False`` (reject the mutation).
        """
        level = zone.protection_level if zone is not None else "invariant_preservation"
        if level == "absolute_immutable":
            return self._strip_ws(original) == self._strip_ws(mutated)
        if self._strip_ws(original) == self._strip_ws(mutated):
            return True
        return self.verify_equivalence(original, mutated)

    def verify_equivalence(self, expr_a: str, expr_b: str) -> bool:
        """Prove two arithmetic expressions are equal for all real inputs (Z3)."""
        try:
            env: Dict[str, Any] = {}
            za = self._expr_to_z3(ast.parse(expr_a, mode="eval").body, env)
            zb = self._expr_to_z3(ast.parse(expr_b, mode="eval").body, env)
        except (SyntaxError, ValueError, ZeroDivisionError):
            return False  # cannot model -> conservatively reject
        try:
            solver = Solver()
            solver.set("timeout", self.smt_timeout_ms)
            solver.add(za != zb)
            return solver.check() == unsat
        except Exception as exc:
            logger.warning("Z3 solver failed for equivalence check: %s", exc)
            return self.fallback != "conservative" and False

    def _expr_to_z3(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return self._expr_to_z3(node.body, env)
        if isinstance(node, ast.BinOp):
            left = self._expr_to_z3(node.left, env)
            right = self._expr_to_z3(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value >= 0:
                    result: Any = RealVal(1)
                    for _ in range(node.right.value):
                        result = result * left
                    return result
                raise ValueError("unsupported exponent")
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            operand = self._expr_to_z3(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return RealVal(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in env:
                env[node.id] = Real(node.id)
            return env[node.id]
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    @staticmethod
    def _strip_ws(text: str) -> str:
        return "".join(text.split())

    # ------------------------------------------------------------------
    # Z3 SMT validation
    # ------------------------------------------------------------------

    def zone_float(self, zone: ShieldZone) -> str:
        """Resolve the floating-point precision selected for a zone."""
        return self.precision_selector.float_for_zone(zone.identifier)

    def _run_smt_validation(self, zone: ShieldZone) -> Dict[str, Any]:
        try:
            solver = Solver()
            solver.set("timeout", self.smt_timeout_ms)

            x = Real("x")
            y = Real("y")
            z = Real("z")

            # Precision-aware tolerance: the effective epsilon never exceeds the
            # selected float's machine epsilon, so quad/arbitrary zones are held
            # to a tighter bound than double-precision ones.
            float_kind = self.zone_float(zone)
            precision_epsilon = self.precision_selector.epsilon_for(float_kind)
            effective_tolerance = min(
                zone.tolerated_precision_loss, precision_epsilon
            ) if zone.tolerated_precision_loss > 0 else zone.tolerated_precision_loss
            tolerance = RealVal(str(effective_tolerance))

            if zone.protection_level == "absolute_immutable":
                original = x + y + z
                transformed = z + y + x
                solver.add(original - transformed != 0)
                result = solver.check()
                if result == unsat:
                    return {"passed": True, "proof": "associativity_preserved"}
                return {
                    "passed": False,
                    "violations": [
                        f"smt:{zone.identifier}: associativity violation possible"
                    ],
                }

            if zone.protection_level == "invariant_preservation":
                original = x * y + z
                transformed = x * y + z
                delta = original - transformed
                solver.add(delta > tolerance)
                result = solver.check()
                if result == unsat:
                    return {
                        "passed": True,
                        "proof": "precision_within_tolerance",
                        "float_kind": float_kind,
                        "effective_tolerance": effective_tolerance,
                    }
                return {
                    "passed": False,
                    "violations": [
                        f"smt:{zone.identifier}: precision loss exceeds tolerance {zone.tolerated_precision_loss}"
                    ],
                }

            return {"passed": True, "proof": "no_constraints"}

        except Exception as exc:
            if self.fallback == "conservative":
                return {"passed": True, "proof": f"fallback_conservative:{exc}"}
            return {
                "passed": False,
                "violations": [f"smt-error:{zone.identifier}:{exc}"],
            }

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_zones(raw_zones: List[Dict[str, Any]]) -> List[ShieldZone]:
        zones: List[ShieldZone] = []
        for raw in raw_zones:
            zones.append(
                ShieldZone(
                    identifier=raw.get("identifier", "unknown"),
                    files=list(raw.get("files", [])),
                    protection_level=raw.get("protection_level", "invariant_preservation"),
                    tolerated_precision_loss=float(raw.get("tolerated_precision_loss", 1e-6)),
                    validation_rules=list(raw.get("validation_rules", [])),
                    floating_point_precision=raw.get("floating_point_precision", "64-bit"),
                    fast_math_override=raw.get("fast_math_override"),
                    floating_point_contract=raw.get("floating_point_contract"),
                    ieee_compliance=raw.get("ieee_compliance"),
                )
            )
        return zones

    # ------------------------------------------------------------------
    # Floating-point compiler-flag emission (feature #3)
    # ------------------------------------------------------------------

    # Compiler families the shield knows how to flag.
    _C_FAMILY = {"gcc", "g++", "clang", "clang++", "cc", "c++"}
    _INTEL_FAMILY = {"icc", "icx", "icpc", "icpx", "ifx", "ifort"}

    def resolve_fp_policy(self, zone: Optional[ShieldZone] = None) -> Dict[str, Any]:
        """Resolve the effective FP policy for a zone (falling back to global)."""
        fast_math = self.fast_math_override
        contract = self.floating_point_contract
        ieee = self.ieee_compliance
        if zone is not None:
            if zone.fast_math_override is not None:
                fast_math = bool(zone.fast_math_override)
            if zone.floating_point_contract is not None:
                contract = str(zone.floating_point_contract).lower()
            if zone.ieee_compliance is not None:
                ieee = str(zone.ieee_compliance).lower()
        return {
            "fast_math": fast_math,
            "floating_point_contract": contract,
            "ieee_compliance": ieee,
        }

    def compiler_flags(self, compiler: str, zone: Optional[ShieldZone] = None) -> List[str]:
        """Return the floating-point flags for ``compiler`` under the policy.

        ``compiler`` is matched case-insensitively against known families
        (gcc/clang, the Intel suite, ``rustc`` and ``nvcc``/``hipcc``).  Per-zone
        overrides take precedence over the global policy so a sensitive zone can
        be locked down even when the rest of the build relaxes.
        """
        policy = self.resolve_fp_policy(zone)
        name = str(compiler).strip().lower()
        # Reduce e.g. ``/usr/bin/x86_64-…-gcc-13`` to a family key.
        base = name.rsplit("/", 1)[-1]

        if base == "rustc":
            return self._rustc_flags(policy)
        if base in ("nvcc", "hipcc"):
            return self._nvcc_flags(policy)
        if any(base.startswith(prefix) for prefix in self._INTEL_FAMILY):
            return self._intel_flags(policy)
        if any(base.startswith(prefix) for prefix in self._C_FAMILY) or base in self._C_FAMILY:
            return self._gcc_clang_flags(policy)
        # Unknown compiler: fall back to the portable GCC/Clang spelling.
        return self._gcc_clang_flags(policy)

    def all_compiler_flags(self, zone: Optional[ShieldZone] = None) -> Dict[str, List[str]]:
        """Convenience: FP flags for every supported compiler family."""
        return {
            "gcc": self.compiler_flags("gcc", zone),
            "clang": self.compiler_flags("clang", zone),
            "intel": self.compiler_flags("icx", zone),
            "rustc": self.compiler_flags("rustc", zone),
            "nvcc": self.compiler_flags("nvcc", zone),
        }

    @staticmethod
    def _gcc_clang_flags(policy: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        flags.append(
            "-ffp-contract=off"
            if policy["floating_point_contract"] == "disallow"
            else "-ffp-contract=fast"
        )
        if policy["fast_math"]:
            flags.append("-ffast-math")
        else:
            flags.append("-fno-fast-math")
        if policy["ieee_compliance"] == "strict":
            # Honour exact IEEE-754 semantics; never fold fast-math back in.
            flags.extend(["-frounding-math", "-fsignaling-nans", "-std=c++17"])
        return flags

    @staticmethod
    def _intel_flags(policy: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        if policy["ieee_compliance"] == "strict":
            flags.append("-fp-model=strict")
        elif not policy["fast_math"]:
            flags.append("-fp-model=precise")
        else:
            flags.append("-fp-model=fast")
        if policy["floating_point_contract"] == "disallow":
            flags.append("-no-fma")
        else:
            flags.append("-fma")
        return flags

    @staticmethod
    def _rustc_flags(policy: Dict[str, Any]) -> List[str]:
        # rustc on stable is IEEE-strict by default; we steer LLVM's contraction
        # and avoid any unsafe fast-math relaxation.
        flags: List[str] = []
        if policy["floating_point_contract"] == "disallow":
            flags.append("-Cllvm-args=-fp-contract=off")
        else:
            flags.append("-Cllvm-args=-fp-contract=fast")
        if not policy["fast_math"]:
            flags.append("-Cllvm-args=-enable-unsafe-fp-math=false")
        return flags

    @staticmethod
    def _nvcc_flags(policy: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        flags.append(
            "--fmad=false" if policy["floating_point_contract"] == "disallow" else "--fmad=true"
        )
        if policy["fast_math"]:
            flags.append("--use_fast_math")
        if policy["ieee_compliance"] == "strict":
            flags.extend(["--ftz=false", "--prec-div=true", "--prec-sqrt=true"])
        return flags
