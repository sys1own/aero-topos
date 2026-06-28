# -*- coding: utf-8 -*-
"""Abstract syntax tree for the ``blueprint.aero`` language.

The grammar is intentionally small and declarative::

    blueprint   := block*
    block       := IDENT STRING '{' field* '}'
    field       := IDENT '=' value
    value       := STRING | NUMBER | BOOL | list
    list        := '[' (value (',' value)* ','?)? ']'

So a document is a flat sequence of *blocks* (``project "name" { ... }``),
each holding ``key = value`` *fields*.  Every node keeps its source
:class:`~blueprint_lang.positions.Span` so validation errors can point at the
exact offending value.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import ClassVar, Dict, List, Optional, Union

from .positions import Span

# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@dataclass
class StringValue:
    value: str
    span: Span
    kind: ClassVar[str] = "string"


@dataclass
class NumberValue:
    value: Union[int, float]
    span: Span
    kind: ClassVar[str] = "number"


@dataclass
class BoolValue:
    value: bool
    span: Span
    kind: ClassVar[str] = "boolean"


@dataclass
class ListValue:
    items: List["AnyValue"]
    span: Span
    kind: ClassVar[str] = "list"

    @property
    def value(self) -> List[object]:
        """The list lowered to plain Python values."""
        return [item.value for item in self.items]


AnyValue = Union[StringValue, NumberValue, BoolValue, ListValue]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@dataclass
class Field:
    """A single ``key = value`` entry inside a block."""

    key: str
    key_span: Span
    value: AnyValue
    span: Span  # key start .. value end


@dataclass
class Block:
    """A ``<type> "<name>" { ... }`` block."""

    type: str
    type_span: Span
    name: str
    name_span: Span
    fields: List[Field] = dataclass_field(default_factory=list)
    span: Optional[Span] = None  # type start .. closing brace end

    def field_map(self) -> Dict[str, Field]:
        return {f.key: f for f in self.fields}

    def get(self, key: str) -> Optional[Field]:
        for f in self.fields:
            if f.key == key:
                return f
        return None


@dataclass
class Blueprint:
    """A parsed blueprint document plus the source it came from."""

    blocks: List[Block]
    source: str
    filename: str = "blueprint.aero"

    @property
    def projects(self) -> List[Block]:
        return [b for b in self.blocks if b.type == "project"]

    @property
    def targets(self) -> List[Block]:
        return [b for b in self.blocks if b.type == "target"]

    def to_config(self) -> Dict[str, object]:
        """Lower the AST into a plain, build-friendly dictionary.

        Shape::

            {
              "project": {"name": ..., "version": ..., ...} | None,
              "targets": {name: {"name": ..., "language": ..., ...}, ...},
            }
        """
        project: Optional[Dict[str, object]] = None
        targets: Dict[str, Dict[str, object]] = {}
        for block in self.blocks:
            lowered: Dict[str, object] = {"name": block.name}
            for fld in block.fields:
                lowered[fld.key] = fld.value.value
            if block.type == "project" and project is None:
                project = lowered
            elif block.type == "target":
                targets[block.name] = lowered
        return {"project": project, "targets": targets}
