"""Shared JSON extraction helpers."""

from __future__ import annotations

import json
from typing import Any, Optional


def extract_json(text: str) -> Optional[Any]:
    """Try to parse *text* as JSON; fall back to the first ``{…}`` span."""
    text = text.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
    return None
