# -*- coding: utf-8 -*-
"""Map ``blueprint.aero`` target blocks to an internal DAG and resolve build order.

The module bridges two worlds:

* **blueprint_lang** -- the block-DSL parser that yields a validated
  :class:`~blueprint_lang.nodes.Blueprint` AST with ``target`` blocks and
  ``requires`` dependency arrays.
* **The orchestrator's DAG** -- a flat ``{name: [deps]}`` dependency matrix
  consumed by the existing build engine.

Public API
----------
* :func:`blueprint_to_dag` -- convert a :class:`Blueprint` into a
  :class:`BuildGraph`.
* :class:`BuildGraph` -- holds the resolved topological order and can
  render a clean, minimalist visual tree of the planned build steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from blueprint_lang.nodes import Blueprint, ListValue, StringValue


@dataclass
class TargetNode:
    """A single build target extracted from the blueprint AST."""

    name: str
    language: str
    sources: List[str]
    requires: List[str]
    flags: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)
    output: Optional[str] = None
    optional: bool = False
    # Rust/Cargo: point at a crate in a subdirectory and/or an explicit manifest.
    manifest_path: Optional[str] = None
    root: Optional[str] = None
    # Pin dependency versions for a synthesised manifest ("name=version" entries).
    cargo_dependencies: List[str] = field(default_factory=list)
    # RUSTFLAGS control: an optimization word and/or explicit rustflags.
    optimization: Optional[str] = None
    rustflags: List[str] = field(default_factory=list)
    # Polyglot emitter: translate UAST into this language instead of compiling.
    target_output_language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flatten this node into the metadata dict the engine/compilers read."""
        meta: Dict[str, Any] = {
            "name": self.name,
            "language": self.language,
            "sources": list(self.sources),
            "requires": list(self.requires),
            "flags": list(self.flags),
            "defines": list(self.defines),
            "optional": self.optional,
        }
        if self.output is not None:
            meta["output"] = self.output
        if self.manifest_path is not None:
            meta["manifest_path"] = self.manifest_path
        if self.root is not None:
            meta["root"] = self.root
        if self.cargo_dependencies:
            meta["cargo_dependencies"] = list(self.cargo_dependencies)
        if self.optimization is not None:
            meta["optimization"] = self.optimization
        if self.rustflags:
            meta["rustflags"] = list(self.rustflags)
        if self.target_output_language is not None:
            meta["target_output_language"] = self.target_output_language
        return meta


# ---------------------------------------------------------------------------
# Field coercion helpers
# ---------------------------------------------------------------------------


def _as_str_list(value: Any) -> List[str]:
    """Coerce a lowered blueprint value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _node_from_lowered(name: str, fields: Dict[str, Any]) -> "TargetNode":
    """Build a :class:`TargetNode` from a single lowered ``target`` block."""
    return TargetNode(
        name=name,
        language=str(fields.get("language", "")),
        sources=_as_str_list(fields.get("sources")),
        requires=_as_str_list(fields.get("requires")),
        flags=_as_str_list(fields.get("flags")),
        defines=_as_str_list(fields.get("defines")),
        output=_as_str(fields.get("output")),
        optional=bool(fields.get("optional", False)),
        manifest_path=_as_str(fields.get("manifest_path")),
        root=_as_str(fields.get("root")),
        cargo_dependencies=_as_str_list(fields.get("cargo_dependencies")),
        optimization=_as_str(fields.get("optimization")),
        rustflags=_as_str_list(fields.get("rustflags")),
        target_output_language=_as_str(fields.get("target_output_language")),
    )


# ---------------------------------------------------------------------------
# BuildGraph
# ---------------------------------------------------------------------------


@dataclass
class BuildGraph:
    """A resolved build DAG: topological order plus the original target nodes."""

    targets: Dict[str, TargetNode]
    build_order: List[str]
    dependency_map: Dict[str, List[str]]
    project_name: str = "project"
    project_version: str = "0.0.0"

    # -- parallel stage grouping ------------------------------------------

    @property
    def levels(self) -> List[List[str]]:
        """Group targets into stages that can be built in parallel.

        A target belongs to the earliest stage in which every one of its
        dependencies has already been scheduled.  Within a stage the original
        :attr:`build_order` ordering is preserved for determinism.
        """
        stage_of: Dict[str, int] = {}
        for name in self.build_order:
            deps = self.dependency_map.get(name, [])
            stage = 0
            for dep in deps:
                if dep in stage_of:
                    stage = max(stage, stage_of[dep] + 1)
            stage_of[name] = stage

        if not stage_of:
            return []
        max_stage = max(stage_of.values())
        levels: List[List[str]] = [[] for _ in range(max_stage + 1)]
        for name in self.build_order:
            levels[stage_of[name]].append(name)
        return [stage for stage in levels if stage]

    # -- visual rendering --------------------------------------------------

    def render_tree(self) -> str:
        """Render a clean, minimalist tree of the planned build."""
        lines: List[str] = []
        lines.append(f"Build Plan: {self.project_name} v{self.project_version}")
        lines.append("")

        ordered = self.build_order
        for index, name in enumerate(ordered):
            node = self.targets[name]
            last = index == len(ordered) - 1
            connector = "└──" if last else "├──"

            descriptor = f"{connector} {name} [{node.language}]"
            if node.optional:
                descriptor += " (optional)"
            lines.append(descriptor)

            child_indent = "    " if last else "│   "
            n_sources = len(node.sources)
            lines.append(
                f"{child_indent}└── {n_sources} source pattern"
                + ("s" if n_sources != 1 else "")
            )
            if node.requires:
                lines.append(f"{child_indent}    requires: {', '.join(node.requires)}")

        stage_count = len(self.levels)
        lines.append("")
        lines.append(f"{len(ordered)} targets, {stage_count} stages")
        return "\n".join(lines)

    # -- engine hand-off ---------------------------------------------------

    def to_build_context(self) -> Dict[str, Any]:
        """Lower the graph into the flat ``build_context`` the engine consumes."""
        metadata = [self.targets[name].to_dict() for name in self.build_order]
        return {
            "compilation_targets": list(self.build_order),
            "dependency_matrix": {k: list(v) for k, v in self.dependency_map.items()},
            "graph": {
                "entrypoint": "orchestrator",
                "targets": list(self.build_order),
                "target_metadata": metadata,
                "dependencies": {k: list(v) for k, v in self.dependency_map.items()},
                "workspace_mode": "incremental",
                "allow_partial_graph": False,
            },
        }


# ---------------------------------------------------------------------------
# Topological resolution
# ---------------------------------------------------------------------------


class CyclicDependencyError(ValueError):
    """Raised when the target dependency graph contains a cycle."""


def _topological_order(
    order_hint: List[str], dependency_map: Dict[str, List[str]]
) -> List[str]:
    """Deterministic Kahn topological sort.

    *order_hint* fixes the tie-break order (declaration order in the
    blueprint), so independent targets keep their original sequence and the
    output is stable across runs.
    """
    rank = {name: i for i, name in enumerate(order_hint)}
    indegree = {name: 0 for name in order_hint}
    dependents: Dict[str, List[str]] = {name: [] for name in order_hint}
    for name in order_hint:
        for dep in dependency_map.get(name, []):
            indegree[name] += 1
            dependents[dep].append(name)

    ready = sorted((n for n in order_hint if indegree[n] == 0), key=lambda n: rank[n])
    resolved: List[str] = []
    while ready:
        current = ready.pop(0)
        resolved.append(current)
        newly_ready: List[str] = []
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                newly_ready.append(child)
        if newly_ready:
            ready.extend(newly_ready)
            ready.sort(key=lambda n: rank[n])

    if len(resolved) != len(order_hint):
        unresolved = [n for n in order_hint if n not in resolved]
        raise CyclicDependencyError(
            f"Cyclic or unsatisfiable target dependencies among: {unresolved}"
        )
    return resolved


def blueprint_to_dag(blueprint: Blueprint) -> BuildGraph:
    """Convert a validated :class:`Blueprint` AST into a :class:`BuildGraph`."""
    config = blueprint.to_config()
    project = config.get("project") or {}
    project_name = str(project.get("name", "project")) if project else "project"
    project_version = str(project.get("version", "0.0.0")) if project else "0.0.0"

    lowered_targets: Dict[str, Dict[str, Any]] = config.get("targets") or {}

    targets: Dict[str, TargetNode] = {}
    order_hint: List[str] = []
    for name, fields in lowered_targets.items():
        targets[name] = _node_from_lowered(name, fields)
        order_hint.append(name)

    # A target may only require declared targets; unknown requires keep their
    # edge dropped here (the validator surfaces hard errors upstream).
    dependency_map: Dict[str, List[str]] = {}
    for name in order_hint:
        deps = [dep for dep in targets[name].requires if dep in targets]
        dependency_map[name] = deps

    build_order = _topological_order(order_hint, dependency_map)

    return BuildGraph(
        targets=targets,
        build_order=build_order,
        dependency_map=dependency_map,
        project_name=project_name,
        project_version=project_version,
    )