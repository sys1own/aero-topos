"""
The Invariant Schema.

A small, domain-agnostic intermediate representation for facts extracted from
unstructured context (papers, prose, JSON, source code, ...): state variables,
algorithmic boundaries (constraints/limits) and mathematical equations.  Every
invariant carries a ``domain`` and is given an id namespaced as
``"<domain>::<symbol>"`` so that two unrelated domains (say ``genomics`` and
``game_engine``) can both define a variable named ``rate`` without colliding --
see :mod:`src.semantic_fluidity.domain` for how the domain is inferred and
:mod:`src.semantic_fluidity.graph` for how same-named symbols across domains are
linked (without being merged) in the system graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SourceRef:
    """Where an invariant was extracted from."""

    path: str
    line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "line": self.line}


def make_id(domain: str, symbol: str) -> str:
    """Namespace a symbol under its domain so cross-domain collisions can't happen."""
    return f"{domain}::{symbol}"


@dataclass
class StateVariable:
    domain: str
    symbol: str
    description: str = ""
    type_hint: str = "unknown"  # "real" | "integer" | "boolean" | "string" | "unknown"
    unit: Optional[str] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    source: Optional[SourceRef] = None
    confidence: float = 1.0

    @property
    def id(self) -> str:
        return make_id(self.domain, self.symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": "state_variable",
            "domain": self.domain,
            "symbol": self.symbol,
            "description": self.description,
            "type_hint": self.type_hint,
            "unit": self.unit,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "source": self.source.to_dict() if self.source else None,
            "confidence": self.confidence,
        }


@dataclass
class AlgorithmicBoundary:
    domain: str
    symbol: str
    description: str
    expression: str
    variables: List[str] = field(default_factory=list)
    source: Optional[SourceRef] = None
    confidence: float = 1.0

    @property
    def id(self) -> str:
        return make_id(self.domain, self.symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": "boundary",
            "domain": self.domain,
            "symbol": self.symbol,
            "description": self.description,
            "expression": self.expression,
            "variables": list(self.variables),
            "source": self.source.to_dict() if self.source else None,
            "confidence": self.confidence,
        }


@dataclass
class Equation:
    domain: str
    symbol: str
    expression: str
    lhs: str
    rhs: str
    variables: List[str] = field(default_factory=list)
    source: Optional[SourceRef] = None
    confidence: float = 1.0

    @property
    def id(self) -> str:
        return make_id(self.domain, self.symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": "equation",
            "domain": self.domain,
            "symbol": self.symbol,
            "expression": self.expression,
            "lhs": self.lhs,
            "rhs": self.rhs,
            "variables": list(self.variables),
            "source": self.source.to_dict() if self.source else None,
            "confidence": self.confidence,
        }


@dataclass
class InvariantSchema:
    """A collection of invariants, grouped implicitly by ``domain``."""

    state_variables: List[StateVariable] = field(default_factory=list)
    boundaries: List[AlgorithmicBoundary] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)

    @property
    def domains(self) -> List[str]:
        seen: Dict[str, None] = {}
        for item in (*self.state_variables, *self.boundaries, *self.equations):
            seen.setdefault(item.domain, None)
        return list(seen)

    def merge(self, other: "InvariantSchema") -> "InvariantSchema":
        self.state_variables.extend(other.state_variables)
        self.boundaries.extend(other.boundaries)
        self.equations.extend(other.equations)
        return self

    def finalize(self) -> "InvariantSchema":
        """Disambiguate symbols that collide within the same (domain, kind) bucket.

        Extractors are free to reuse an obvious symbol name (e.g. ``x``) per
        document; once everything is merged this renames the 2nd, 3rd, ...
        occurrence to ``x#2``, ``x#3`` so every invariant's namespaced ``id``
        stays unique without extractors having to coordinate.
        """
        for items in (self.state_variables, self.boundaries, self.equations):
            seen: Dict[Tuple[str, str], int] = {}
            for item in items:
                key = (item.domain, item.symbol)
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1:
                    item.symbol = f"{item.symbol}#{seen[key]}"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": self.domains,
            "state_variables": [v.to_dict() for v in self.state_variables],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "equations": [e.to_dict() for e in self.equations],
        }


# A hand-written JSON-schema-style document describing the wire format above.
# Kept dependency-free (no ``jsonschema`` package) -- ``validate_invariant_document``
# below performs the equivalent structural checks by hand.
JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Aero Invariant Schema",
    "type": "object",
    "required": ["domains", "state_variables", "boundaries", "equations"],
    "properties": {
        "domains": {"type": "array", "items": {"type": "string"}},
        "state_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "kind", "domain", "symbol"],
                "properties": {"kind": {"const": "state_variable"}},
            },
        },
        "boundaries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "kind", "domain", "symbol", "expression"],
                "properties": {"kind": {"const": "boundary"}},
            },
        },
        "equations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "kind", "domain", "symbol", "lhs", "rhs"],
                "properties": {"kind": {"const": "equation"}},
            },
        },
    },
}


def validate_invariant_document(doc: Dict[str, Any]) -> List[str]:
    """Structurally validate a serialized :class:`InvariantSchema`.

    Returns a list of human-readable problems; an empty list means ``doc``
    conforms to :data:`JSON_SCHEMA`.
    """
    errors: List[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]

    for key in ("domains", "state_variables", "boundaries", "equations"):
        if key not in doc:
            errors.append(f"missing required key '{key}'")
        elif not isinstance(doc[key], list):
            errors.append(f"'{key}' must be a list")

    _required_by_kind = {
        "state_variables": ("state_variable", ("id", "kind", "domain", "symbol")),
        "boundaries": ("boundary", ("id", "kind", "domain", "symbol", "expression")),
        "equations": ("equation", ("id", "kind", "domain", "symbol", "lhs", "rhs")),
    }
    for key, (expected_kind, required_fields) in _required_by_kind.items():
        for index, item in enumerate(doc.get(key, []) or []):
            if not isinstance(item, dict):
                errors.append(f"{key}[{index}] must be an object")
                continue
            for required_field in required_fields:
                if required_field not in item:
                    errors.append(f"{key}[{index}] missing required field '{required_field}'")
            if item.get("kind") not in (None, expected_kind):
                errors.append(
                    f"{key}[{index}] has kind '{item.get('kind')}', expected '{expected_kind}'"
                )
    return errors
