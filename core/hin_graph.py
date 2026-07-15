"""Graph model for the Phase 5 Boundary-Aware Topological Execution Engine.

``HINGraph`` is a lean, directed view over an underlying interaction net (or any
DAG-like graph).  It exposes the adjacency, in/out degrees, expected arities and
boundary interface signatures that :mod:`core.invariants` and
:mod:`core.wavefront_scheduler` need to verify and schedule HIN topologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class GraphMutationKind(Enum):
    ADD_NODE = auto()
    REMOVE_NODE = auto()
    ADD_EDGE = auto()
    REMOVE_EDGE = auto()
    UPDATE_NODE = auto()


@dataclass
class GraphMutation:
    """A single mutation to be applied incrementally to a ``HINGraph``."""

    kind: GraphMutationKind
    node_id: Optional[str] = None
    edge: Optional[Tuple[str, str]] = None
    payload: Optional[Any] = None

    @staticmethod
    def add_node(node_id: str, payload: Any = None) -> "GraphMutation":
        return GraphMutation(GraphMutationKind.ADD_NODE, node_id=node_id, payload=payload)

    @staticmethod
    def remove_node(node_id: str) -> "GraphMutation":
        return GraphMutation(GraphMutationKind.REMOVE_NODE, node_id=node_id)

    @staticmethod
    def add_edge(src: str, dst: str, payload: Any = None) -> "GraphMutation":
        return GraphMutation(
            GraphMutationKind.ADD_EDGE, edge=(src, dst), payload=payload
        )

    @staticmethod
    def remove_edge(src: str, dst: str) -> "GraphMutation":
        return GraphMutation(GraphMutationKind.REMOVE_EDGE, edge=(src, dst))

    @staticmethod
    def update_node(node_id: str, payload: Any = None) -> "GraphMutation":
        return GraphMutation(
            GraphMutationKind.UPDATE_NODE, node_id=node_id, payload=payload
        )


@dataclass
class InterfaceSignature:
    """Boundary port signature for a topological execution wave."""

    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterfaceSignature):
            return NotImplemented
        return self.inputs == other.inputs and self.outputs == other.outputs

    def matches(self, other: "InterfaceSignature") -> bool:
        return self == other

    def __repr__(self) -> str:
        return f"InterfaceSignature(inputs={sorted(self.inputs)}, outputs={sorted(self.outputs)})"


@dataclass
class HINGraph:
    """Directed graph view used by the invariants and wavefront scheduler."""

    nodes: Set[str]
    edges: List[Tuple[str, str, str]]
    adj_list: Dict[str, List[str]]
    in_degrees: Dict[str, int]
    out_degrees: Dict[str, int]
    expected_arity: Dict[str, int]
    node_types: Dict[str, str]
    reverse_adj: Dict[str, List[str]]
    undirected_adj: Dict[str, List[str]]
    edge_types: List[Tuple[str, Any, Any]]
    edge_records: List[Dict[str, Any]] = field(default_factory=list)
    interface_signatures: Dict[int, InterfaceSignature] = field(default_factory=dict)

    @classmethod
    def from_hin_network(cls, network: Any) -> "HINGraph":
        """Project an interaction net onto a directed ``HINGraph``.

        Port bindings are treated as undirected edges for spectral checks, but are
        oriented deterministically by sorted ``node_id`` for the wavefront
        scheduler so the result is a DAG.
        """
        nodes: Set[str] = set(getattr(network, "nodes", {}).keys())
        adj: Dict[str, List[str]] = {n: [] for n in nodes}
        rev: Dict[str, List[str]] = {n: [] for n in nodes}
        in_deg: Dict[str, int] = {n: 0 for n in nodes}
        out_deg: Dict[str, int] = {n: 0 for n in nodes}
        undirected: Dict[str, List[str]] = {n: [] for n in nodes}

        expected_arity: Dict[str, int] = {}
        node_types: Dict[str, str] = {}
        edges: List[Tuple[str, str, str]] = []
        edge_types: List[Tuple[str, Any, Any]] = []
        edge_records: List[Dict[str, Any]] = []

        for node_id, node in getattr(network, "nodes", {}).items():
            expected_arity[node_id] = len(node.ports())
            node_types[node_id] = type(node).__name__

            for port in node.ports():
                target = getattr(port, "target", None)
                if target is None:
                    continue
                other = getattr(getattr(target, "owner", None), "node_id", None)
                if other is None or other == node_id:
                    continue

                # Canonical orientation by sorted node_id.  Each physical binding
                # is added exactly once (when we visit the smaller-id endpoint),
                # so parallel edges between the same node pair are preserved.
                if other < node_id:
                    continue
                a, b = node_id, other

                if b not in undirected[a]:
                    undirected[a].append(b)
                    undirected[b].append(a)

                label = (
                    f"{a}:{getattr(port, 'name', '?')}"
                    f"->{b}:{getattr(target, 'name', '?')}"
                )
                edges.append((a, b, label))
                adj[a].append(b)
                rev[b].append(a)
                out_deg[a] += 1
                in_deg[b] += 1
                src_type = getattr(port, "type", None)
                dst_type = getattr(target, "type", None)
                edge_types.append((label, src_type, dst_type))
                edge_records.append(
                    {
                        "src": a,
                        "dst": b,
                        "label": label,
                        "src_type": src_type,
                        "dst_type": dst_type,
                    }
                )

        return cls(
            nodes=nodes,
            edges=edges,
            adj_list=adj,
            in_degrees=in_deg,
            out_degrees=out_deg,
            expected_arity=expected_arity,
            node_types=node_types,
            reverse_adj=rev,
            undirected_adj=undirected,
            edge_types=edge_types,
            edge_records=edge_records,
        )
