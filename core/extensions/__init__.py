# -*- coding: utf-8 -*-
"""AeroNova active native-extension layer.

Compiled shared libraries (``.so`` / ``.dylib`` / ``.pyd``) that the
``scaffold --merge-active`` flow merges back into the live runtime are dropped
into *this* directory.  On import the package scans the folder, loads every
shared library as a Python extension module, and exposes each one both as an
attribute of this package **and** as a top-level entry in :data:`sys.modules`,
so a merged component (e.g. ``anyon_simulator``) is immediately importable
globally::

    import anyon_simulator            # works once merged + loaded
    from core.extensions import anyon_simulator

The scan is idempotent: re-importing the package (or calling
:func:`load_extensions` again) only loads libraries that are not already live.
Loading failures are swallowed per-library so one bad artefact never breaks the
whole runtime.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

# Platform extension-module suffixes (e.g. ``.cpython-311-...-gnu.so``,
# ``.abi3.so``, ``.so``, ``.pyd``).  Sorted longest-first so ABI-tagged names
# are matched before the bare ``.so`` fallback.
_EXT_SUFFIXES = tuple(
    sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
)

# The platform's canonical (untagged) suffix used when naming merged artefacts.
PRIMARY_SUFFIX = importlib.machinery.EXTENSION_SUFFIXES[-1]

#: module name -> loaded extension module, for everything this package brought up.
loaded_extensions: Dict[str, ModuleType] = {}

__all__: List[str] = ["load_extensions", "load_extension_file", "loaded_extensions"]


def extensions_dir() -> Path:
    """Absolute path to this active-extension directory."""
    return Path(__file__).resolve().parent


def _module_name_for(filename: str) -> Optional[str]:
    """Strip a recognised extension suffix to recover the import name."""
    for suffix in _EXT_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return None


def load_extension_file(path: Path, module_name: Optional[str] = None) -> Optional[ModuleType]:
    """Load a single shared library as an extension module and register it.

    The module is inserted into :data:`sys.modules`, recorded in
    :data:`loaded_extensions`, and bound as an attribute of this package so it
    is importable both as a top-level name and via ``core.extensions``.
    Returns the loaded module, or ``None`` on failure.
    """
    path = Path(path)
    name = module_name or _module_name_for(path.name)
    if not name:
        return None
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        loaded_extensions[name] = existing
        return existing
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    sys.modules[name] = module
    loaded_extensions[name] = module
    setattr(sys.modules[__name__], name, module)
    globals()[name] = module
    if name not in __all__:
        __all__.append(name)
    return module


def load_extensions(directory: Optional[Path] = None) -> Dict[str, ModuleType]:
    """Scan *directory* (default: this folder) and load every shared library."""
    base = Path(directory) if directory is not None else extensions_dir()
    if not base.is_dir():
        return loaded_extensions
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        name = _module_name_for(path.name)
        if not name or name in loaded_extensions:
            continue
        load_extension_file(path, name)
    return loaded_extensions


# Auto-expose any extensions already present when the package is first imported.
load_extensions()
