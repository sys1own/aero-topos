"""Aero-Calculus compiler frontend: UAST -> HIN translation + module mitosis.

This module implements :class:`UASTToHINTranslator`, the compiler frontend of
the Aero-Calculus.  It performs two jobs:

1. **Homomorphic translation.**  A normalized UAST (Universal Abstract Syntax
   Tree) is walked recursively and mapped onto a *topologically homomorphic*
   HIN network (see :mod:`core.hin_vm`).  The mapping preserves semantic
   dependencies **without name lookups** -- every variable scope is resolved
   to a direct port connection (or a :class:`DuplicatorNode` fork for a
   multi-reference variable), so the resulting net carries no global symbol
   table.

2. **Module mitosis.**  When local graph complexity crosses
   ``auto_split_threshold`` the module is bisected.  The split is computed by
   *spectral graph partitioning*: the Fiedler vector (the eigenvector of the
   second-smallest eigenvalue of the graph Laplacian ``L = D - A``) yields a
   near-minimum cut.  Edges crossing the cut are reified as explicit boundary
   interface ports -- a clean API contract, with no runtime global tables --
   embodying the Holographic Boundary Principle ``Int(Ω) ↔ ∂Ω``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment dependent fallback
    np = None  # type: ignore[assignment]

from core.hin_vm import (
    ConstructorNode,
    DestructorNode,
    DuplicatorNode,
    EraserNode,
    HINNetwork,
    MELLType,
    Node,
    Port,
    SwitchNode,
    ValueNode,
)


# ---------------------------------------------------------------------------
# UAST kind normalisation
# ---------------------------------------------------------------------------
def _kind(node: dict) -> str:
    """Return the canonical syntactic kind of a UAST node dict."""
    return (
        node.get("canonical_kind")
        or node.get("kind")
        or node.get("type")
        or "unknown"
    )


_FUNCTION_KINDS = {
    "function_declaration",
    "function_definition",
    "function",
    "lambda",
}
_BINDING_KINDS = {"binding", "assignment", "let", "variable_declaration"}
_REFERENCE_KINDS = {"reference", "identifier", "name", "var"}
_LITERAL_KINDS = {"literal", "constant", "number", "string", "value"}
_IF_KINDS = {"if", "if_statement", "conditional", "if_else"}
_CALL_KINDS = {"call", "application", "apply", "call_expression"}
_CONTAINER_KINDS = {
    "module",
    "translation_unit",
    "block",
    "lexical_block",
    "body",
    "sequence",
    "program",
}


# ---------------------------------------------------------------------------
# Boundary interface node (reified cut edge / API contract)
# ---------------------------------------------------------------------------
class BoundaryPortNode(Node):
    """A reified boundary interface port produced by module mitosis.

    When an edge is severed by the spectral cut, each severed end is capped
    with a :class:`BoundaryPortNode` inside its own partition.  The two caps
    form the two halves of the new API contract ``∂Ω`` -- the only channel
    through which the isolated module interior communicates with the outside.
    The principal port carries a wildcard type so it can terminate any wire
    without weakening the linear typing of the data it stands in for.
    """

    symbol = "∂"

    def __init__(self, node_id: str, contract_id: str, wire_type: MELLType):
        super().__init__(node_id)
        self.contract_id = contract_id
        self.wire_type = wire_type
        self._set_principal(MELLType.any_())


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------
class UASTToHINTranslator:
    """Translate normalized UAST syntax-trees into homomorphic HIN networks."""

    def __init__(self, auto_split_threshold: int = 120):
        self.auto_split_threshold = auto_split_threshold
        # Name -> Port scope resolver.  Each frame maps a live variable name to
        # the *source* port currently providing its value.
        self.scope_stack: List[Dict[str, Port]] = []
        # Remaining reference counts per scope frame, used to decide whether a
        # variable use is a plain edge (last use) or needs a duplicator fork.
        self._ref_remaining: List[Dict[str, int]] = []

    # -- public entry points -----------------------------------------------
    def translate(self, uast_root: dict) -> HINNetwork:
        """Translate a UAST root into a HIN network (public API)."""
        return self.translate_uast(uast_root)

    def translate_uast(self, uast: dict) -> HINNetwork:
        net = HINNetwork()
        self.scope_stack = []
        self._ref_remaining = []
        self._push_scope(uast)
        result = self._build_container(uast, net)
        # Seal the module: cap the trailing output so Int(Ω) is a closed net.
        if result is not None and result.target is None:
            self._terminate(net, result)
        self._pop_scope(net)
        return net

    # -- recursive builder -------------------------------------------------
    def _traverse_and_build(self, node: dict, net: HINNetwork) -> Optional[Port]:
        """Translate a single UAST node, returning its output port (if any).

        AST node geometries:

        * Function declarations  -> :class:`ConstructorNode`
        * Variable bindings/refs -> port wiring + :class:`DuplicatorNode` forks
        * Constant literals      -> :class:`ValueNode`
        * If/Else branches       -> :class:`SwitchNode`
        * Calls/applications     -> :class:`DestructorNode`
        """
        if not isinstance(node, dict):
            return None
        kind = _kind(node)

        if kind in _CONTAINER_KINDS:
            return self._build_container(node, net)
        if kind in _FUNCTION_KINDS:
            return self._build_function(node, net)
        if kind in _BINDING_KINDS:
            return self._build_binding(node, net)
        if kind in _REFERENCE_KINDS:
            return self._build_reference(node, net)
        if kind in _LITERAL_KINDS:
            return self._build_literal(node, net)
        if kind in _IF_KINDS:
            return self._build_if(node, net)
        if kind in _CALL_KINDS:
            return self._build_call(node, net)

        # Unknown node: translate its children as a sequence.
        return self._build_container(node, net)

    # -- containers --------------------------------------------------------
    def _build_container(self, node: dict, net: HINNetwork) -> Optional[Port]:
        """Translate a sequence of statements; return the last value port."""
        last: Optional[Port] = None
        for child in self._children(node):
            out = self._traverse_and_build(child, net)
            # A standalone expression statement leaves a free output: cap it so
            # the module stays a closed, fully-terminated net (Int(Ω) sealed).
            if last is not None and last.target is None:
                self._terminate(net, last)
            last = out
        return last

    # -- function declarations --------------------------------------------
    def _build_function(self, node: dict, net: HINNetwork) -> Port:
        ctor = ConstructorNode(net.fresh_id("γ"))
        net.register_node(ctor)

        body = node.get("body", self._children(node))
        body_node = {"type": "body", "children": self._as_list(body)}

        self._push_scope(body_node)
        param = node.get("param") or node.get("name_param")
        params = node.get("params") or ([param] if param else [])
        # The closure binds its first parameter to a_1 (input argument).
        if params:
            self.scope_stack[-1][params[0]] = ctor.a_1
        else:
            # No parameter: the argument wire is unused, cap it.
            self._terminate(net, ctor.a_1)

        result = self._build_container(body_node, net)
        if result is None:
            result = ValueNode(net.fresh_id("V"), None).p
            net.register_node(result.owner)
        # Wire the body result to the return path a_2.
        net._link(result, ctor.a_2)
        self._pop_scope(net)

        # Register the closure under its declared name in the enclosing scope.
        name = node.get("name")
        if name and self.scope_stack:
            self.scope_stack[-1][name] = ctor.p
            self._ref_remaining[-1].setdefault(
                name, self._count_name(node, name)
            )
        return ctor.p

    # -- bindings ----------------------------------------------------------
    def _build_binding(self, node: dict, net: HINNetwork) -> None:
        value = node.get("value") or node.get("init") or node.get("expr")
        name = node.get("name") or node.get("target")
        port = self._traverse_and_build(value, net) if value else None
        if port is None:
            port = ValueNode(net.fresh_id("V"), None).p
            net.register_node(port.owner)
        if name:
            self.scope_stack[-1][name] = port
        else:
            self._terminate(net, port)
        return None

    # -- references --------------------------------------------------------
    def _build_reference(self, node: dict, net: HINNetwork) -> Port:
        name = node.get("name") or node.get("text") or node.get("value")
        return self._resolve(str(name), net)

    def _resolve(self, name: str, net: HINNetwork) -> Port:
        """Resolve a variable name to a port, forking on multi-reference.

        On the *final* use of a name the bound source port is returned
        directly -- a plain linear edge.  On every earlier use a
        :class:`DuplicatorNode` is spliced in: one output copy is handed to the
        consumer, the other becomes the new source for subsequent uses.  This
        is exactly how the scope resolver eliminates program variables.
        """
        for idx in range(len(self.scope_stack) - 1, -1, -1):
            frame = self.scope_stack[idx]
            if name not in frame:
                continue
            src = frame[name]
            remaining = self._ref_remaining[idx].get(name, 1) - 1
            self._ref_remaining[idx][name] = remaining
            if remaining <= 0:
                # Last reference: consume the source directly.
                del frame[name]
                return src
            # More references to come: fork a duplicator.
            dup = DuplicatorNode(net.fresh_id("δ"))
            net.register_node(dup)
            net._link(src, dup.p)
            frame[name] = dup.a_2
            return dup.a_1

        # Free / external variable: model as an unbound value node.
        node = ValueNode(net.fresh_id("V"), None)
        net.register_node(node)
        return node.p

    # -- literals ----------------------------------------------------------
    def _build_literal(self, node: dict, net: HINNetwork) -> Port:
        value = node.get("value", node.get("text"))
        vnode = ValueNode(net.fresh_id("V"), value)
        net.register_node(vnode)
        return vnode.p

    # -- conditionals ------------------------------------------------------
    def _build_if(self, node: dict, net: HINNetwork) -> Port:
        switch = SwitchNode(net.fresh_id("σ"))
        net.register_node(switch)

        cond = node.get("condition") or node.get("test") or node.get("cond")
        then_branch = node.get("then") or node.get("consequent")
        else_branch = node.get("else") or node.get("alternate")

        cond_port = self._traverse_and_build(cond, net) if cond else None
        if cond_port is None:
            cond_port = ValueNode(net.fresh_id("V"), False).p
            net.register_node(cond_port.owner)
        net._link(cond_port, switch.p)

        then_port = self._branch_port(then_branch, net)
        else_port = self._branch_port(else_branch, net)
        net._link(then_port, switch.a_1)
        net._link(else_port, switch.a_2)

        return switch.a_3

    def _branch_port(self, branch, net: HINNetwork) -> Port:
        if branch is None:
            node = ValueNode(net.fresh_id("V"), None)
            net.register_node(node)
            return node.p
        port = self._traverse_and_build(branch, net)
        if port is None:
            node = ValueNode(net.fresh_id("V"), None)
            net.register_node(node)
            return node.p
        return port

    # -- calls / applications ----------------------------------------------
    def _build_call(self, node: dict, net: HINNetwork) -> Port:
        dtor = DestructorNode(net.fresh_id("γ⁻¹"))
        net.register_node(dtor)

        func = node.get("function") or node.get("callee") or node.get("func")
        arg = node.get("argument") or node.get("arg")
        args = node.get("arguments") or ([arg] if arg is not None else [])

        func_port = self._traverse_and_build(func, net) if func else None
        if func_port is None:
            func_port = ValueNode(net.fresh_id("V"), None).p
            net.register_node(func_port.owner)
        net._link(func_port, dtor.p)

        if args:
            arg_port = self._traverse_and_build(args[0], net)
            if arg_port is None:
                arg_port = ValueNode(net.fresh_id("V"), None).p
                net.register_node(arg_port.owner)
            net._link(arg_port, dtor.a_1)
        else:
            self._terminate(net, dtor.a_1)

        return dtor.a_2

    # -- scope helpers -----------------------------------------------------
    def _push_scope(self, subtree: dict) -> None:
        self.scope_stack.append({})
        self._ref_remaining.append(self._count_refs(subtree))

    def _pop_scope(self, net: HINNetwork) -> None:
        frame = self.scope_stack.pop()
        self._ref_remaining.pop()
        # Cap any source port left unreferenced so no wire dangles.
        for port in frame.values():
            if port.target is None:
                self._terminate(net, port)

    def _terminate(self, net: HINNetwork, port: Port) -> None:
        """Cap a free wire with an eraser (no un-terminated ports)."""
        if port is None or port.target is not None:
            return
        eraser = EraserNode(net.fresh_id("ε"))
        net.register_node(eraser)
        net._link(eraser.p, port)

    # -- reference counting ------------------------------------------------
    def _count_refs(self, subtree: dict) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        self._count_into(subtree, counts)
        return counts

    def _count_into(self, node, counts: Dict[str, int]) -> None:
        if isinstance(node, list):
            for item in node:
                self._count_into(item, counts)
            return
        if not isinstance(node, dict):
            return
        if _kind(node) in _REFERENCE_KINDS:
            name = node.get("name") or node.get("text") or node.get("value")
            if name is not None:
                counts[str(name)] = counts.get(str(name), 0) + 1
        for value in node.values():
            if isinstance(value, (dict, list)):
                self._count_into(value, counts)

    def _count_name(self, node, name: str) -> int:
        counts: Dict[str, int] = {}
        self._count_into(node, counts)
        return counts.get(name, 0)

    # -- misc helpers ------------------------------------------------------
    @staticmethod
    def _children(node: dict) -> List[dict]:
        children = node.get("children") or node.get("body") or []
        return [c for c in UASTToHINTranslator._as_list(children) if isinstance(c, dict)]

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    # ======================================================================
    # Complexity + module mitosis
    # ======================================================================
    def evaluate_complexity(self, network: HINNetwork) -> Dict[str, float]:
        """Compute node/edge connectivity density of a network."""
        nodes = list(network.nodes.values())
        n = len(nodes)
        adj = self._adjacency(network, nodes)
        if np is None:
            edge_count = float(sum((sum(row) for row in adj)) / 2.0)
        else:
            edge_count = float(np.sum(adj) / 2.0)
        max_edges = n * (n - 1) / 2.0
        density = edge_count / max_edges if max_edges > 0 else 0.0
        avg_degree = (2.0 * edge_count / n) if n > 0 else 0.0
        return {
            "node_count": float(n),
            "edge_count": edge_count,
            "density": density,
            "avg_degree": avg_degree,
            "exceeds_threshold": float(n > self.auto_split_threshold),
        }

    def execute_mitosis(
        self, module: HINNetwork, threshold: Optional[int] = None
    ) -> Tuple[HINNetwork, HINNetwork]:
        """Split ``module`` when its node count exceeds ``threshold``.

        Returns a pair of sub-networks.  If the module is under threshold the
        original network is returned unchanged alongside an empty network, so
        callers can treat the result uniformly.
        """
        limit = self.auto_split_threshold if threshold is None else threshold
        if len(module.nodes) <= limit:
            return module, HINNetwork()
        if np is None:
            return module, HINNetwork()
        return self.split_module(module)

    def compute_fiedler_vector(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Compute the Fiedler vector for spectral bisection.

        The Fiedler vector is the eigenvector of the *second-smallest*
        eigenvalue of the graph Laplacian ``L = D - A``; its component signs
        give a near-minimum balanced cut.
        """
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        sorted_indices = np.argsort(eigenvalues)
        fiedler_index = sorted_indices[1]
        return eigenvectors[:, fiedler_index]

    def split_module(
        self, net: HINNetwork
    ) -> Tuple[HINNetwork, HINNetwork]:
        """Bisect ``net`` along the spectral cut, reifying crossing edges.

        Every wire crossing the partition boundary is severed and capped on
        both sides by a :class:`BoundaryPortNode`, forming the reified API
        contract ``∂Ω``.  Linear typing is preserved and no port is left
        un-terminated: each severed end is bound to its boundary cap.
        """
        node_list = list(net.nodes.values())
        n = len(node_list)
        part_1 = HINNetwork()
        part_2 = HINNetwork()

        # Degenerate cases: nothing to bisect.
        if n < 2:
            for node in node_list:
                part_1.register_node(node)
            part_1.boundary_contracts = []  # type: ignore[attr-defined]
            part_2.boundary_contracts = []  # type: ignore[attr-defined]
            self._rescan_active(part_1)
            return part_1, part_2

        node_to_idx = {node.node_id: idx for idx, node in enumerate(node_list)}
        adj = self._adjacency(net, node_list)

        fiedler = self.compute_fiedler_vector(adj)
        # Partition by the sign of each Fiedler component (the bisection).
        side = {
            node_list[i].node_id: (0 if fiedler[i] >= 0 else 1)
            for i in range(n)
        }
        # Guard against a degenerate all-one-side split: fall back to a
        # median threshold so both partitions are non-empty.
        if len({s for s in side.values()}) == 1:
            median = float(np.median(fiedler))
            side = {
                node_list[i].node_id: (0 if fiedler[i] >= median else 1)
                for i in range(n)
            }

        nets = (part_1, part_2)

        # 1. Identify and reify every boundary-crossing edge BEFORE moving
        #    nodes, so the cut is recorded against the original topology.
        contracts: List[dict] = []
        seen: set = set()
        for node in node_list:
            for port in node.ports():
                target = port.target
                if target is None:
                    continue
                other = target.owner
                if other.node_id not in node_to_idx:
                    continue  # external/already-capped
                key = frozenset((id(port), id(target)))
                if key in seen:
                    continue
                seen.add(key)
                if side[node.node_id] == side[other.node_id]:
                    continue  # internal edge, untouched

                contract_id = f"contract#{len(contracts)}"
                wire_type = port.type
                # Cap this end inside its own partition.
                cap_a = BoundaryPortNode(
                    nets[side[node.node_id]].fresh_id("∂"),
                    contract_id,
                    wire_type,
                )
                cap_b = BoundaryPortNode(
                    nets[side[other.node_id]].fresh_id("∂"),
                    contract_id,
                    target.type,
                )
                nets[side[node.node_id]].register_node(cap_a)
                nets[side[other.node_id]].register_node(cap_b)
                # Sever the cross edge and bind each end to its boundary cap.
                port.target = cap_a.p
                cap_a.p.target = port
                target.target = cap_b.p
                cap_b.p.target = target
                contracts.append(
                    {
                        "contract_id": contract_id,
                        "side_a": side[node.node_id],
                        "side_b": side[other.node_id],
                        "endpoint_a": (node.node_id, port.name),
                        "endpoint_b": (other.node_id, target.name),
                        "type": repr(wire_type),
                    }
                )

        # 2. Move the original nodes into their partitions.
        for node in node_list:
            nets[side[node.node_id]].register_node(node)

        # 3. Recompute active pairs locally and expose the cut for inspection.
        self._rescan_active(part_1)
        self._rescan_active(part_2)
        part_1.boundary_contracts = contracts  # type: ignore[attr-defined]
        part_2.boundary_contracts = contracts  # type: ignore[attr-defined]
        return part_1, part_2

    # -- mitosis helpers ---------------------------------------------------
    @staticmethod
    def _adjacency(net: HINNetwork, node_list: List[Node]) -> np.ndarray:
        n = len(node_list)
        node_to_idx = {node.node_id: idx for idx, node in enumerate(node_list)}
        if np is None:
            adj = [[0.0 for _ in range(n)] for _ in range(n)]
        else:
            adj = np.zeros((n, n))
        for node in node_list:
            i = node_to_idx[node.node_id]
            for port in node.ports():
                target = port.target
                if target is None:
                    continue
                j = node_to_idx.get(target.owner.node_id)
                if j is None or j == i:
                    continue
                adj[i][j] += 1.0
        # Symmetrise (each undirected wire contributes to both endpoints).
        if np is None:
            for i in range(n):
                for j in range(i + 1, n):
                    weight = max(adj[i][j], adj[j][i])
                    adj[i][j] = weight
                    adj[j][i] = weight
        else:
            adj = np.maximum(adj, adj.T)
        return adj

    @staticmethod
    def _rescan_active(net: HINNetwork) -> None:
        net.active_pairs = []
        seen: set = set()
        for node in net.nodes.values():
            p = node.p
            if p is None or p.target is None or not p.target.is_principal:
                continue
            other = p.target.owner
            if other.node_id not in net.nodes:
                continue
            key = frozenset((node.node_id, other.node_id))
            if key in seen:
                continue
            seen.add(key)
            net.active_pairs.append((node, other))


__all__ = [
    "UASTToHINTranslator",
    "BoundaryPortNode",
]
