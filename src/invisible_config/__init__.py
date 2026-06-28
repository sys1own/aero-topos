"""
The Invisible Configuration Layer.

Shrinks ``blueprint.aero`` to a few lines of pure semantic intent and infers
everything else -- the execution DAG, language/FFI boundaries and self-healing
error-correction loops -- from the files in the project directory, so running
``aero`` requires zero further input.

Typical use::

    from pathlib import Path
    from src.invisible_config import InvisibleConfigEngine

    engine = InvisibleConfigEngine(Path("."))
    context = engine.build_context_from_source(open("blueprint.aero").read())
    # `context` plugs straight into the core execution system.
"""

from src.invisible_config.dag_inference import (
    INVARIANTS_NODE,
    DAGInferenceEngine,
    FfiBoundary,
    InferredDAG,
    InferredTarget,
)
from src.invisible_config.engine import InvisibleConfigEngine
from src.invisible_config.lean_parser import (
    LeanBlueprint,
    LeanBlueprintError,
    looks_like_lean_blueprint,
    parse_lean_blueprint,
)
from src.invisible_config.self_healing import (
    GlueCodePatcher,
    HealingAttempt,
    HealingResult,
    SelfHealingExecutor,
)

__all__ = [
    "InvisibleConfigEngine",
    "DAGInferenceEngine",
    "InferredDAG",
    "InferredTarget",
    "FfiBoundary",
    "INVARIANTS_NODE",
    "LeanBlueprint",
    "LeanBlueprintError",
    "looks_like_lean_blueprint",
    "parse_lean_blueprint",
    "SelfHealingExecutor",
    "GlueCodePatcher",
    "HealingResult",
    "HealingAttempt",
]

__version__ = "1.0.0"
