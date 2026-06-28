"""
Regex/NLP-lite rule extraction for unstructured prose (``.txt``/``.md``, and the
best-effort text pulled out of PDFs).

This is intentionally a set of small, composable cue-phrase and symbol-pattern
regexes rather than a full NLP pipeline -- offline, dependency-free, and good
enough to pull "X = Y" equations, "let X be ..." / "where X denotes ..." state
variable declarations, and "must not exceed" / "must remain between" boundary
constraints out of plain-language text such as medical papers, economics prose
or mathematical write-ups.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

from src.semantic_fluidity.extractors.base import RuleExtractor
from src.semantic_fluidity.schema import AlgorithmicBoundary, Equation, InvariantSchema, SourceRef, StateVariable

if TYPE_CHECKING:
    from src.semantic_fluidity.documents import IngestedDocument

_IDENT = r"[A-Za-z][A-Za-z0-9_]*"

# The rhs alternation prefers a whole "digits.digits" decimal literal over a
# bare "[^.]" character so that a decimal point inside a number (e.g. "0.5")
# is consumed as part of the number rather than being mistaken for the
# sentence-ending period that should terminate the match.
_EQUATION_RE = re.compile(rf"(?P<lhs>{_IDENT})\s*=\s*(?P<rhs>(?:\d+\.\d+|[^.\n;])+)")

_LET_BE_RE = re.compile(rf"\blet\s+(?P<symbol>{_IDENT})\s+be\s+(?P<desc>[^.,;\n]+)", re.IGNORECASE)
_WHERE_RE = re.compile(
    rf"\bwhere\s+(?P<symbol>{_IDENT})\s+(?:denotes|represents|is)\s+(?P<desc>[^.,;\n]+)",
    re.IGNORECASE,
)

# A bare "[\d.]+" would swallow a trailing sentence-ending period (e.g. "90."),
# so bounds match an optional decimal fraction explicitly instead.
_NUMBER = r"\d+(?:\.\d+)?"
_BETWEEN_RE = re.compile(
    rf"(?P<symbol>{_IDENT})\s+must\s+(?:remain\s+|be\s+)?between\s+(?P<low>{_NUMBER})\s+and\s+(?P<high>{_NUMBER})",
    re.IGNORECASE,
)
_NOT_EXCEED_RE = re.compile(
    rf"(?P<symbol>{_IDENT})\s+must\s+not\s+exceed\s+(?P<bound>{_NUMBER})",
    re.IGNORECASE,
)
_UPPER_BOUND_RE = re.compile(
    rf"(?P<symbol>{_IDENT})\s+must\s+(?:remain\s+|be\s+)?(?:below|under)\s+(?P<bound>{_NUMBER})",
    re.IGNORECASE,
)
_LOWER_BOUND_RE = re.compile(
    rf"(?P<symbol>{_IDENT})\s+must\s+(?:remain\s+|be\s+)?(?:above|over|at\s+least)\s+(?P<bound>{_NUMBER})",
    re.IGNORECASE,
)

_VARIABLE_TOKEN_RE = re.compile(_IDENT)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


class TextRuleExtractor(RuleExtractor):
    """Cue-phrase + regex extraction for free-form prose."""

    def extract(self, document: "IngestedDocument", domain: str) -> InvariantSchema:
        text = document.text
        schema = InvariantSchema()
        if not text or not text.strip():
            return schema

        path = str(document.path)
        self._extract_state_variables(text, domain, path, schema)
        self._extract_boundaries(text, domain, path, schema)
        self._extract_equations(text, domain, path, schema)
        return schema

    @staticmethod
    def _extract_state_variables(text: str, domain: str, path: str, schema: InvariantSchema) -> None:
        for pattern in (_LET_BE_RE, _WHERE_RE):
            for match in pattern.finditer(text):
                schema.state_variables.append(
                    StateVariable(
                        domain=domain,
                        symbol=match.group("symbol"),
                        description=match.group("desc").strip(),
                        source=SourceRef(path, _line_number(text, match.start())),
                        confidence=0.7,
                    )
                )

    @staticmethod
    def _extract_boundaries(text: str, domain: str, path: str, schema: InvariantSchema) -> None:
        for match in _BETWEEN_RE.finditer(text):
            symbol = match.group("symbol")
            low, high = match.group("low"), match.group("high")
            schema.boundaries.append(
                AlgorithmicBoundary(
                    domain=domain,
                    symbol=symbol,
                    description=match.group(0).strip(),
                    expression=f"{low} <= {symbol} <= {high}",
                    variables=[symbol],
                    source=SourceRef(path, _line_number(text, match.start())),
                    confidence=0.7,
                )
            )
        for pattern, comparator in (
            (_NOT_EXCEED_RE, "<="),
            (_UPPER_BOUND_RE, "<"),
            (_LOWER_BOUND_RE, ">"),
        ):
            for match in pattern.finditer(text):
                symbol = match.group("symbol")
                bound = match.group("bound")
                schema.boundaries.append(
                    AlgorithmicBoundary(
                        domain=domain,
                        symbol=symbol,
                        description=match.group(0).strip(),
                        expression=f"{symbol} {comparator} {bound}",
                        variables=[symbol],
                        source=SourceRef(path, _line_number(text, match.start())),
                        confidence=0.7,
                    )
                )

    @staticmethod
    def _extract_equations(text: str, domain: str, path: str, schema: InvariantSchema) -> None:
        for match in _EQUATION_RE.finditer(text):
            lhs = match.group("lhs")
            rhs = match.group("rhs").strip()
            if not rhs:
                continue
            variables = list(dict.fromkeys(v for v in _VARIABLE_TOKEN_RE.findall(rhs) if v != lhs))
            schema.equations.append(
                Equation(
                    domain=domain,
                    symbol=lhs,
                    expression=f"{lhs} = {rhs}",
                    lhs=lhs,
                    rhs=rhs,
                    variables=variables,
                    source=SourceRef(path, _line_number(text, match.start())),
                    confidence=0.6,
                )
            )
