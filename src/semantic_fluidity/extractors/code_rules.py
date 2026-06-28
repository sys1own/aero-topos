"""
AST-based (Python) and regex-based (C/C++) semantic extraction from source code.

This is the offline fallback requirement #2 explicitly allows ("if offline,
implement an AST-based semantic extractor"): module-level constants become
state variables, ``assert``/``#define``-guarded conditions become algorithmic
boundaries, and simple single-expression functions/assignments become
equations.  Python is parsed with the stdlib :mod:`ast` module (matching the
style already used by :mod:`src.context.analyser`); C/C++ uses a best-effort
regex scan rather than requiring tree-sitter, so this module has no extra
dependency beyond the stdlib.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, List

from src.semantic_fluidity.extractors.base import RuleExtractor
from src.semantic_fluidity.schema import AlgorithmicBoundary, Equation, InvariantSchema, SourceRef, StateVariable

if TYPE_CHECKING:
    from src.semantic_fluidity.documents import IngestedDocument

_PY_TYPE_HINTS = {bool: "boolean", int: "integer", float: "real", str: "string"}


class CodeRuleExtractor(RuleExtractor):
    """Dispatches to a per-language AST/regex extractor based on ``document.format``."""

    def extract(self, document: "IngestedDocument", domain: str) -> InvariantSchema:
        if document.format == "code:python":
            return self._extract_python(document.text, domain, str(document.path))
        return self._extract_c_like(document.text, domain, str(document.path))

    # ------------------------------------------------------------------
    # Python (AST-based)
    # ------------------------------------------------------------------

    def _extract_python(self, source: str, domain: str, path: str) -> InvariantSchema:
        schema = InvariantSchema()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return schema

        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self._maybe_constant(node.targets[0].id, node.value, node.lineno, domain, path, schema)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self._maybe_constant(node.target.id, node.value, node.lineno, domain, path, schema)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
                try:
                    expression = ast.unparse(node.test)
                except Exception:
                    continue
                variables = [n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)]
                schema.boundaries.append(
                    AlgorithmicBoundary(
                        domain=domain,
                        symbol=variables[0] if variables else f"assert_{node.lineno}",
                        description=f"assert {expression}",
                        expression=expression,
                        variables=variables,
                        source=SourceRef(path, node.lineno),
                        confidence=0.9,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._maybe_equation_from_function(node, domain, path, schema)
        return schema

    @staticmethod
    def _maybe_constant(name: str, value: ast.AST, lineno: int, domain: str, path: str, schema: InvariantSchema) -> None:
        if not isinstance(value, ast.Constant) or isinstance(value.value, (type(None),)):
            return
        py_type = type(value.value)
        type_hint = _PY_TYPE_HINTS.get(py_type, "unknown")
        bounds = None
        if isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
            bounds = (float(value.value), float(value.value))
        schema.state_variables.append(
            StateVariable(
                domain=domain,
                symbol=name,
                description=f"module-level constant '{name}'",
                type_hint=type_hint,
                bounds=bounds,
                source=SourceRef(path, lineno),
                confidence=0.9,
            )
        )

    @staticmethod
    def _maybe_equation_from_function(
        node: "ast.FunctionDef | ast.AsyncFunctionDef", domain: str, path: str, schema: InvariantSchema
    ) -> None:
        body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            return
        if isinstance(body[0].value, (ast.Constant, ast.Name)):
            return  # trivial return -- not interesting as an equation
        try:
            rhs = ast.unparse(body[0].value)
        except Exception:
            return
        variables = list(dict.fromkeys(n.id for n in ast.walk(body[0].value) if isinstance(n, ast.Name)))
        schema.equations.append(
            Equation(
                domain=domain,
                symbol=node.name,
                expression=f"{node.name} = {rhs}",
                lhs=node.name,
                rhs=rhs,
                variables=variables,
                source=SourceRef(path, node.lineno),
                confidence=0.8,
            )
        )

    # ------------------------------------------------------------------
    # C / C++ (regex-based)
    # ------------------------------------------------------------------

    _DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(\S.*)$", re.MULTILINE)
    _CONST_RE = re.compile(
        r"\bconst\s+[\w:<>]+\s+([A-Za-z_]\w*)\s*=\s*([^;]+);",
    )
    _ASSERT_RE = re.compile(r"\bassert\s*\(\s*([^;]+?)\s*\)\s*;")
    _IDENT_RE = re.compile(r"[A-Za-z_]\w*")
    _NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?[fFlL]?$")
    _STRING_LITERAL_RE = re.compile(r'^"(.*)"$')

    def _extract_c_like(self, source: str, domain: str, path: str) -> InvariantSchema:
        schema = InvariantSchema()

        for match in self._DEFINE_RE.finditer(source):
            name, value = match.group(1), match.group(2).strip()
            line = source.count("\n", 0, match.start()) + 1
            self._record_value(name, value, domain, path, line, schema, confidence=0.85)

        for match in self._CONST_RE.finditer(source):
            name, value = match.group(1), match.group(2).strip()
            line = source.count("\n", 0, match.start()) + 1
            self._record_value(name, value, domain, path, line, schema, confidence=0.85)

        for match in self._ASSERT_RE.finditer(source):
            expression = match.group(1).strip()
            line = source.count("\n", 0, match.start()) + 1
            variables = self._IDENT_RE.findall(expression)
            schema.boundaries.append(
                AlgorithmicBoundary(
                    domain=domain,
                    symbol=variables[0] if variables else f"assert_{line}",
                    description=f"assert({expression})",
                    expression=expression,
                    variables=variables,
                    source=SourceRef(path, line),
                    confidence=0.85,
                )
            )
        return schema

    @classmethod
    def _record_value(
        cls, name: str, value: str, domain: str, path: str, line: int, schema: InvariantSchema, confidence: float
    ) -> None:
        string_match = cls._STRING_LITERAL_RE.match(value)
        if string_match:
            schema.state_variables.append(
                StateVariable(
                    domain=domain,
                    symbol=name,
                    description=f"constant '{name}'",
                    type_hint="string",
                    source=SourceRef(path, line),
                    confidence=confidence,
                )
            )
        elif cls._NUMERIC_RE.match(value):
            numeric = float(value.rstrip("fFlL"))
            schema.state_variables.append(
                StateVariable(
                    domain=domain,
                    symbol=name,
                    description=f"constant '{name}'",
                    type_hint="real" if "." in value else "integer",
                    bounds=(numeric, numeric),
                    source=SourceRef(path, line),
                    confidence=confidence,
                )
            )
        else:
            variables = list(dict.fromkeys(v for v in cls._IDENT_RE.findall(value) if v != name))
            schema.equations.append(
                Equation(
                    domain=domain,
                    symbol=name,
                    expression=f"{name} = {value}",
                    lhs=name,
                    rhs=value,
                    variables=variables,
                    source=SourceRef(path, line),
                    confidence=confidence - 0.15,
                )
            )
