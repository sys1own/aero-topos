"""
``InvisibleConfigEngine`` -- the facade that makes ``aero`` run from a few lines
of intent with zero further input.

It parses the lean blueprint, runs the :class:`DAGInferenceEngine`, and emits a
normalized ``build_context`` shaped exactly like the one
:func:`blueprint_parser.parse_blueprint` produces for the other dialects -- so
the inferred graph plugs straight into the existing core execution system.

The ``optimize`` intent word is mapped onto the concrete optimizer flags and the
optional build subsystems (hardware-polymerization, GPU, numerical libraries,
semantic-fluidity ingestion of the ``ingest`` files into text invariants).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from blueprint_parser import _default_optional_sections
from src.invisible_config.dag_inference import INVARIANTS_NODE, DAGInferenceEngine, InferredDAG
from src.invisible_config.lean_parser import LeanBlueprint, parse_lean_blueprint

# How each `optimize` intent word maps onto optimizer aggressiveness.
_OPTIMIZE_PROFILES = {
    "maximum_hardware": {
        "optimization_level": "O3",
        "hotspot_loop_unroll_depth": 64,
        "vector_intrinsics_auto_generation": True,
        "polymorphization": True,
        "gpu": True,
        "libraries": "auto",
    },
    "balanced": {
        "optimization_level": "O2",
        "hotspot_loop_unroll_depth": 32,
        "vector_intrinsics_auto_generation": True,
        "polymorphization": True,
        "gpu": False,
        "libraries": "auto",
    },
    "size": {
        "optimization_level": "Os",
        "hotspot_loop_unroll_depth": 8,
        "vector_intrinsics_auto_generation": False,
        "polymorphization": False,
        "gpu": False,
        "libraries": "none",
    },
    "debug": {
        "optimization_level": "O0",
        "hotspot_loop_unroll_depth": 1,
        "vector_intrinsics_auto_generation": False,
        "polymorphization": False,
        "gpu": False,
        "libraries": "none",
    },
}


class InvisibleConfigEngine:
    """Parse + infer + emit an executable build context from lean intent."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    # ------------------------------------------------------------------

    def infer_from_source(self, content: str) -> InferredDAG:
        blueprint = parse_lean_blueprint(content)
        return self.infer(blueprint)

    def infer(self, blueprint: LeanBlueprint) -> InferredDAG:
        return DAGInferenceEngine(blueprint, self.project_root).infer()

    # ------------------------------------------------------------------
    # Connection to the core execution system
    # ------------------------------------------------------------------

    def build_context_from_source(self, content: str) -> Dict[str, Any]:
        blueprint = parse_lean_blueprint(content)
        dag = self.infer(blueprint)
        return self.to_build_context(blueprint, dag)

    def to_build_context(self, blueprint: LeanBlueprint, dag: InferredDAG) -> Dict[str, Any]:
        profile = _OPTIMIZE_PROFILES.get(blueprint.optimize, _OPTIMIZE_PROFILES["balanced"])
        target_names = [t.name for t in dag.targets]
        dependencies = dag.dependency_matrix()
        # The synthetic invariants node is part of the inferred graph but is not
        # a compilable target -- expose it as a producer dependency only.
        target_metadata = [
            {"name": t.name, "language": t.language, "role": t.role, "sources": t.sources}
            for t in dag.targets
        ]

        context: Dict[str, Any] = {
            "workspace_status": "inferred_active",
            "config_layer": "invisible",
            "timestamp": time.time(),
            "project_name": blueprint.project,
            "compilation_targets": target_names,
            "dependency_matrix": dependencies,
            "active_optimizer_flags": {
                "profile_guided_optimization": "enabled_strict",
                "tier_shifting_hotness_threshold": 100,
                "hotspot_loop_unroll_depth": profile["hotspot_loop_unroll_depth"],
                "aot_boundary_check_elimination": True,
                "vector_intrinsics_auto_generation": profile["vector_intrinsics_auto_generation"],
                "consensus_protocol": "raft_driven_mutation_lock",
                "mutation_entropy_clamp_threshold": 0.05,
                "optimization_level": profile["optimization_level"],
            },
            "environment_targets": {
                "execution_mode": "lock_free_polling_wheel_realtime",
                "core_affinity_mask": "0xFFFF",
                "numa_node_locality_binding": True,
                "inter_core_ring_buffer_capacity": 262144,
            },
            "resource_metrics": {
                "pipeline_budget_seconds": 120.0,
                "max_memory_mb": 2048,
                "elapsed_seconds": {name: 0.0 for name in target_names},
            },
            "node_configurations": {},
            "graph": {
                "entrypoint": "orchestrator",
                "targets": target_names,
                "target_metadata": target_metadata,
                "dependencies": dependencies,
                "workspace_mode": "incremental",
                "allow_partial_graph": True,
            },
            # The full inferred picture, for downstream consumers / introspection.
            "inferred_dag": dag.to_dict(),
            "ffi_boundaries": [b.to_dict() for b in dag.ffi_boundaries],
            "self_healing": {
                "enabled": True,
                "max_attempts": 3,
                "boundaries": [b.to_dict() for b in dag.ffi_boundaries],
            },
        }

        # Optional subsystems, driven by the `optimize` intent + `ingest`.
        optional = _default_optional_sections()
        if profile["polymorphization"]:
            context["polymorphization"] = {"enabled": True, "source_dir": "build_artifacts"}
        if profile["gpu"]:
            optional["gpu"] = {**optional["gpu"], "enabled": True, "backend": "cuda"}
        if profile["libraries"] == "auto":
            optional["libraries"] = {
                "blas": "auto", "lapack": "auto", "mpi": False, "mpi_flavor": "openmpi", "cuda": "none"
            }
        # `ingest` -> semantic-fluidity invariants feeding the compiled cores.
        if dag.has_invariants:
            optional["context"] = {"sources": self._ingest_sources(blueprint)}
            context["semantic_fluidity"] = {
                "enabled": True,
                "ingest": list(blueprint.ingest),
                "invariants_node": INVARIANTS_NODE,
            }

        context.update(optional)
        return context

    def _ingest_sources(self, blueprint: LeanBlueprint) -> List[Dict[str, Any]]:
        """Map each ingest path into a [context].sources-style entry."""
        sources: List[Dict[str, Any]] = []
        for path in blueprint.ingest:
            sources.append(
                {
                    "path": path,
                    "language": "text",
                    "purpose": "text_invariants",
                    "repair_rules": [],
                    "target_mapping": INVARIANTS_NODE,
                }
            )
        return sources
