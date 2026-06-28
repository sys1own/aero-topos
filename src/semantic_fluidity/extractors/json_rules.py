"""
Structured-JSON ingestion.

Two modes: if a JSON document already looks like an invariant document (has
``state_variables``/``variables``, ``boundaries``/``constraints`` or
``equations`` keys), each entry is mapped directly into the schema with light
normalization.  Otherwise the JSON is treated as plain configuration data and
its scalar leaves are flattened into state variables (e.g.
``{"max_retries": 3}`` becomes a state variable named ``max_retries`` with
``type_hint="integer"`` and bounds ``(3, 3)``), so arbitrary structured context
is never silently dropped.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from src.semantic_fluidity.extractors.base import RuleExtractor
from src.semantic_fluidity.schema import AlgorithmicBoundary, Equation, InvariantSchema, SourceRef, StateVariable

if TYPE_CHECKING:
    from src.semantic_fluidity.documents import IngestedDocument

_SCALAR_TYPE_HINTS = {bool: "boolean", int: "integer", float: "real", str: "string"}
_INVARIANT_KEYS = {"state_variables", "variables", "boundaries", "constraints", "equations"}


class JsonRuleExtractor(RuleExtractor):
    def extract(self, document: "IngestedDocument", domain: str) -> InvariantSchema:
        schema = InvariantSchema()
        try:
            payload = json.loads(document.text)
        except (json.JSONDecodeError, ValueError):
            return schema

        path = str(document.path)
        if isinstance(payload, dict) and _INVARIANT_KEYS & payload.keys():
            self._extract_structured(payload, domain, path, schema)
        elif isinstance(payload, dict):
            self._flatten_generic(payload, domain, path, schema)
        return schema

    @staticmethod
    def _extract_structured(payload: Dict[str, Any], domain: str, path: str, schema: InvariantSchema) -> None:
        for entry in payload.get("state_variables", payload.get("variables", [])) or []:
            if not isinstance(entry, dict) or "symbol" not in entry and "name" not in entry:
                continue
            symbol = str(entry.get("symbol", entry.get("name")))
            bounds = entry.get("bounds")
            schema.state_variables.append(
                StateVariable(
                    domain=domain,
                    symbol=symbol,
                    description=str(entry.get("description", "")),
                    type_hint=str(entry.get("type_hint", entry.get("type", "unknown"))),
                    unit=entry.get("unit"),
                    bounds=tuple(bounds) if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else None,
                    source=SourceRef(path),
                    confidence=float(entry.get("confidence", 0.95)),
                )
            )

        for entry in payload.get("boundaries", payload.get("constraints", [])) or []:
            if not isinstance(entry, dict) or "expression" not in entry:
                continue
            symbol = str(entry.get("symbol", entry.get("name", "boundary")))
            schema.boundaries.append(
                AlgorithmicBoundary(
                    domain=domain,
                    symbol=symbol,
                    description=str(entry.get("description", entry["expression"])),
                    expression=str(entry["expression"]),
                    variables=list(entry.get("variables", [])),
                    source=SourceRef(path),
                    confidence=float(entry.get("confidence", 0.95)),
                )
            )

        for entry in payload.get("equations", []) or []:
            if not isinstance(entry, dict) or "lhs" not in entry or "rhs" not in entry:
                continue
            symbol = str(entry.get("symbol", entry["lhs"]))
            schema.equations.append(
                Equation(
                    domain=domain,
                    symbol=symbol,
                    expression=str(entry.get("expression", f"{entry['lhs']} = {entry['rhs']}")),
                    lhs=str(entry["lhs"]),
                    rhs=str(entry["rhs"]),
                    variables=list(entry.get("variables", [])),
                    source=SourceRef(path),
                    confidence=float(entry.get("confidence", 0.95)),
                )
            )

    @staticmethod
    def _flatten_generic(payload: Dict[str, Any], domain: str, path: str, schema: InvariantSchema) -> None:
        for key, value in payload.items():
            if not isinstance(value, (bool, int, float, str)):
                continue
            schema.state_variables.append(
                StateVariable(
                    domain=domain,
                    symbol=key,
                    description=f"configuration value '{key}'",
                    type_hint=_SCALAR_TYPE_HINTS.get(type(value), "unknown"),
                    bounds=(float(value), float(value))
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else None,
                    source=SourceRef(path),
                    confidence=0.5,
                )
            )
