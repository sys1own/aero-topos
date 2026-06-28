"""Shared dataclass serialization helpers."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence


def dataclass_to_dict(
    obj: Any,
    *,
    exclude: Optional[Sequence[str]] = None,
    round_keys: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Convert a dataclass instance to a plain dict.

    Parameters
    ----------
    obj:
        A dataclass instance.
    exclude:
        Field names to omit from the result.
    round_keys:
        Mapping of ``field_name -> decimal_places`` for rounding float values.
    """
    result: Dict[str, Any] = {}
    excluded = set(exclude or ())
    rounding = round_keys or {}
    for f in dataclasses.fields(obj):
        if f.name in excluded:
            continue
        value = getattr(obj, f.name)
        if f.name in rounding and isinstance(value, float):
            value = round(value, rounding[f.name])
        result[f.name] = value
    return result
