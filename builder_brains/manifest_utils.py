"""Shared utilities for loading and querying the build manifest.

Every ``builder_brains`` module needs the same boilerplate to locate and parse
``build_manifest.json`` and extract sub-dictionaries from it.  This module
centralises that logic.
"""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger("builder_brains.manifest_utils")

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "build_manifest.json")


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, Any]:
    """Load the build manifest JSON from *path*, returning ``{}`` on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load build_manifest.json: %s — using defaults", exc)
        return {}


def get_module_params(manifest: Dict[str, Any], module_key: str) -> Dict[str, Any]:
    """Return the hyperparameter sub-dict for *module_key* (e.g. ``"scanner"``)."""
    return manifest.get("hyperparameter_weights", {}).get(module_key, {})


def get_thresholds(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("thresholds", {})


def get_cost_ceilings(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return manifest.get("execution_cost_ceilings", {})
