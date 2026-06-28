# -*- coding: utf-8 -*-
"""Staging workspace isolation for self-hosting bootstrap operations.

When AeroNova targets its own codebase, this module ensures safe execution by:

1. Detecting self-targeting scopes (engine directory overlap).
2. Routing all writes to an isolated shadow stage (`.aero/bootstrap_stage/`).
3. Gating promotion via atomic swap only after validation passes.
4. Preventing recursive bootstrap loops via an execution flag.
"""

from .isolation import (
    BootstrapIsolationError,
    BootstrapStage,
    detect_self_targeting,
    is_bootstrap_active,
    set_bootstrap_active,
)

__all__ = [
    "BootstrapIsolationError",
    "BootstrapStage",
    "detect_self_targeting",
    "is_bootstrap_active",
    "set_bootstrap_active",
]
