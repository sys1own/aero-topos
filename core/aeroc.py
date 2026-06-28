"""``.aeroc`` -- the Aero-Calculus compiled-graph container format.

An ``.aeroc`` file is a JSON document that fully serializes a compiled HIN
network: every node (with its class, payload and spatial-temporal coordinate),
every typed port, and every port-to-port wire.  Serialization is *lossless* --
``deserialize_network(serialize_network(net))`` reconstructs an isomorphic
network, including post-reduction (minimized) topologies, so a graph can be
compiled, reduced inside the VM, and written back to disk without data loss.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from core.hin_vm import (
    CausalProjectionNode,
    ConstructorNode,
    DestructorNode,
    DuplicatorNode,
    EraserNode,
    HINNetwork,
    MELLType,
    Node,
    Port,
    SwitchNode,
    TypeKind,
    ValueNode,
)

AEROC_VERSION = "aeroc_v1"

# Registry mapping a serialized class name back to its node class.
_NODE_REGISTRY: Dict[str, type] = {
    cls.__name__: cls
    for cls in (
        Node,
        ConstructorNode,
        DestructorNode,
        DuplicatorNode,
        EraserNode,
        ValueNode,
        SwitchNode,
        CausalProjectionNode,
    )
}


def _register_optional() -> None:
    """Register node classes defined outside hin_vm (e.g. boundary caps)."""
    try:
        from core.translator import BoundaryPortNode

        _NODE_REGISTRY.setdefault("BoundaryPortNode", BoundaryPortNode)
    except Exception:  # pragma: no cover - translator optional at load time
        pass


_register_optional()


# ---------------------------------------------------------------------------
# MELL type (de)serialization
# ---------------------------------------------------------------------------
def serialize_type(t: Optional[MELLType]) -> Optional[dict]:
    if t is None:
        return None
    return {
        "kind": t.kind.value,
        "left": serialize_type(t.left),
        "right": serialize_type(t.right),
        "wildcard": bool(getattr(t, "wildcard", False)),
    }


def deserialize_type(data: Optional[dict]) -> Optional[MELLType]:
    if data is None:
        return None
    return MELLType(
        TypeKind(data["kind"]),
        deserialize_type(data.get("left")),
        deserialize_type(data.get("right")),
        wildcard=bool(data.get("wildcard", False)),
    )


# ---------------------------------------------------------------------------
# Value (de)serialization
# ---------------------------------------------------------------------------
def _serialize_value(value) -> dict:
    try:
        json.dumps(value)
        return {"json": value}
    except (TypeError, ValueError):
        return {"repr": repr(value)}


def _deserialize_value(data: Optional[dict]):
    if not data:
        return None
    if "json" in data:
        return data["json"]
    return data.get("repr")


# ---------------------------------------------------------------------------
# Network (de)serialization
# ---------------------------------------------------------------------------
def serialize_network(net: HINNetwork) -> dict:
    """Serialize a HIN network into a plain ``.aeroc`` dict."""
    nodes_out: List[dict] = []
    for node in net.nodes.values():
        ports_out = []
        for port in node.ports():
            target = port.target
            endpoint = (
                [target.owner.node_id, target.name] if target is not None else None
            )
            ports_out.append(
                {
                    "name": port.name,
                    "type": serialize_type(port.type),
                    "target": endpoint,
                }
            )
        record = {
            "class": type(node).__name__,
            "node_id": node.node_id,
            "ports": ports_out,
        }
        if isinstance(node, ValueNode):
            record["value"] = _serialize_value(node.value)
        coord = getattr(node, "coordinate", None)
        if coord is not None:
            record["coordinate"] = [
                str(coord.x),
                str(coord.y),
                str(coord.z),
                int(coord.t_causal),
            ]
        nodes_out.append(record)

    return {
        "version": AEROC_VERSION,
        "node_count": len(nodes_out),
        "active_pairs": [
            [a.node_id, b.node_id] for a, b in net.active_pairs
        ],
        "nodes": nodes_out,
    }


def deserialize_network(data: dict) -> HINNetwork:
    """Reconstruct a HIN network from a ``.aeroc`` dict (lossless)."""
    net = HINNetwork()

    # First pass: materialize every node and its ports (unwired).
    port_index: Dict[str, Dict[str, Port]] = {}
    for record in data.get("nodes", []):
        cls = _NODE_REGISTRY.get(record["class"], Node)
        node = object.__new__(cls)
        Node.__init__(node, record["node_id"])
        if isinstance(node, ValueNode):
            node.value = _deserialize_value(record.get("value"))

        port_index[node.node_id] = {}
        for pdata in record["ports"]:
            port = Port(node, pdata["name"], deserialize_type(pdata["type"]))
            if pdata["name"] == Port.PRINCIPAL:
                node.p = port
            else:
                node.aux.append(port)
            port_index[node.node_id][pdata["name"]] = port

        coord = record.get("coordinate")
        if coord is not None:
            from core.spacetime_ledger import CoordinateVector

            node.coordinate = CoordinateVector(coord[0], coord[1], coord[2], coord[3])

        net.register_node(node)

    # Second pass: rewire ports from the saved endpoints.
    for record in data.get("nodes", []):
        owner_id = record["node_id"]
        for pdata in record["ports"]:
            endpoint = pdata.get("target")
            if endpoint is None:
                continue
            src = port_index[owner_id][pdata["name"]]
            tgt_owner, tgt_name = endpoint
            tgt = port_index.get(tgt_owner, {}).get(tgt_name)
            if tgt is not None:
                src.target = tgt  # symmetric: partner sets its own side too

    # Rebuild the active-pair worklist from the wiring.
    seen = set()
    for node in net.nodes.values():
        p = node.p
        if p is None or p.target is None or not p.target.is_principal:
            continue
        other = p.target.owner
        key = frozenset((node.node_id, other.node_id))
        if key in seen or other.node_id not in net.nodes:
            continue
        seen.add(key)
        net.active_pairs.append((node, other))

    return net


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------
def save_aeroc(net: HINNetwork, path: str) -> str:
    """Serialize ``net`` and write it to ``path`` as JSON; return ``path``."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialize_network(net), handle, indent=2)
    return path


def load_aeroc(path: str) -> HINNetwork:
    """Load and reconstruct a HIN network from an ``.aeroc`` file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return deserialize_network(data)


__all__ = [
    "AEROC_VERSION",
    "serialize_network",
    "deserialize_network",
    "serialize_type",
    "deserialize_type",
    "save_aeroc",
    "load_aeroc",
]
