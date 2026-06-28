# -*- coding: utf-8 -*-
"""
Zero-config, out-of-tree standalone repository generator.

Resolves a ``source_entry`` from anywhere on the filesystem, shields rug/pyo3
sources with the codified compatibility fixes, and emits a complete, turn-key
Rust/pyo3 project (``Cargo.toml`` / ``src/lib.rs`` / ``.gitignore`` /
``README.md`` / ``test_binding.py``) into an out-of-tree workspace -- keeping the
``aero-universal`` directory completely clean.

Typical use::

    from src.scaffold import ScaffoldEngine

    engine = ScaffoldEngine(logger=print, verbose=True)
    result = engine.scaffold("/content/lib.rs", distribution_directory="~/out/anyon")
"""

from src.scaffold.decomposition import (
    DecomposedModule,
    DecompositionError,
    DecompositionResult,
    ImportCollisionError,
    MergedSource,
    MissingASTNodeError,
    ModularDecomposer,
    merge_source_asts,
    resolve_cross_imports,
)
from src.scaffold.test_matrix import TestMatrixResult, generate_test_matrix
from src.scaffold.engine import ScaffoldEngine, ScaffoldResult
from src.scaffold.import_pruner import PruneOutcome, prune_dead_imports, render_imports
from src.scaffold.language_router import resolve_target_language
from src.scaffold.pipeline import (
    PipelineResult,
    ScaffoldBuildPipeline,
    scaffold_config_from_context,
    should_run_scaffold_pipeline,
)
from src.scaffold.recovery import DiagnosticRecoveryRunner, RecoveryResult
from src.scaffold.repo_generator import (
    GeneratedRepo,
    RepoSpec,
    build_spec,
    generate_repo,
    infer_dependencies,
)
from src.scaffold.rust_shield import COMPATIBILITY_SHIMS, EXTENSION_TRAITS, RUST_ANCHORS, RustSemanticShield, ShieldReport
from src.scaffold.source_resolver import (
    SourceEntry,
    SourceEntryNotFound,
    resolve_source_entry,
)
from src.scaffold.workspace import OutOfTreeWorkspace, WorkspaceLocationError

__all__ = [
    "ScaffoldEngine",
    "ScaffoldResult",
    "ScaffoldBuildPipeline",
    "PipelineResult",
    "scaffold_config_from_context",
    "should_run_scaffold_pipeline",
    "resolve_target_language",
    "RustSemanticShield",
    "ShieldReport",
    "COMPATIBILITY_SHIMS",
    "EXTENSION_TRAITS",
    "RUST_ANCHORS",
    "OutOfTreeWorkspace",
    "WorkspaceLocationError",
    "RepoSpec",
    "GeneratedRepo",
    "build_spec",
    "generate_repo",
    "infer_dependencies",
    "SourceEntry",
    "SourceEntryNotFound",
    "resolve_source_entry",
    "DiagnosticRecoveryRunner",
    "RecoveryResult",
    "ModularDecomposer",
    "DecompositionResult",
    "DecomposedModule",
    "DecompositionError",
    "MissingASTNodeError",
    "ImportCollisionError",
    "prune_dead_imports",
    "render_imports",
    "PruneOutcome",
    "merge_source_asts",
    "resolve_cross_imports",
    "MergedSource",
    "generate_test_matrix",
    "TestMatrixResult",
]

__version__ = "1.0.0"
