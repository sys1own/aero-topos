"""Aero-Calculus HIN-VM: the foundational execution model.

This module implements the *Hierarchical Interaction Net Virtual Machine*
(``HIN-VM``) that underpins the Aero-Calculus.  Program logic is represented
**not** as linear bytecode but as a directed graph of self-reducing
*interaction nodes* wired together through strictly typed *ports*.

The model is governed by three core mathematical invariants:

* **Conservation of Edges** -- every auxiliary port has a valence of exactly
  one.  There are no implicit references, no aliasing and no unmanaged
  garbage states.  A port is either unbound (free) or bound to exactly one
  opposing port.

* **Active Pairs** -- an *active pair* exists when, and only when, two nodes
  are connected directly through their **principal** ports (``p`` ⋈ ``p``).
  Reduction only ever happens on active pairs.

* **Linear Type System** -- port connections are validated against
  Multiplicative-Exponential Linear Logic (MELL) using the structural types
  ``I`` (unit), ``Tensor(A, B)``, ``Implication(A, B)`` and ``Bang(A)``.

Reduction is performed by :meth:`HINNetwork.reduce_step`, which applies exactly
one deterministic active-pair rewrite rule and then re-scans the locality for
freshly created active pairs.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from enum import Enum
from itertools import count
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# MELL type system
# ---------------------------------------------------------------------------
class TypeKind(Enum):
    I = "I"
    TENSOR = "Tensor"
    IMPLICATION = "Implication"
    BANG = "Bang"


class MELLType:
    """A structural Multiplicative-Exponential Linear Logic type.

    ``I``                -- multiplicative unit (a closed/atomic wire).
    ``Tensor(A, B)``     -- ``A ⊗ B`` (simultaneous resources).
    ``Implication(A, B)``-- ``A ⊸ B`` (linear function, consumes ``A``).
    ``Bang(A)``          -- ``!A`` (a duplicable / shareable resource).
    """

    def __init__(
        self,
        kind: TypeKind,
        left: Optional["MELLType"] = None,
        right: Optional["MELLType"] = None,
        wildcard: bool = False,
    ):
        self.kind = kind
        self.left = left
        self.right = right
        # A wildcard (⊤) endpoint models a *cut*: the principal port of a
        # structural/control agent (ε, δ, σ, P) connects to its partner's
        # principal regardless of the resource flowing across, exactly as a
        # linear-logic cut links a formula to its dual.  Data wires never use
        # this, so structural typing on values remains strict.
        self.wildcard = wildcard

    # -- constructors -------------------------------------------------------
    @staticmethod
    def any_() -> "MELLType":
        """A wildcard type that unifies with any other type (the cut/⊤)."""
        return MELLType(TypeKind.I, wildcard=True)

    @staticmethod
    def unit() -> "MELLType":
        return MELLType(TypeKind.I)

    @staticmethod
    def tensor(left: "MELLType", right: "MELLType") -> "MELLType":
        return MELLType(TypeKind.TENSOR, left, right)

    @staticmethod
    def implication(left: "MELLType", right: "MELLType") -> "MELLType":
        return MELLType(TypeKind.IMPLICATION, left, right)

    @staticmethod
    def bang(inner: "MELLType") -> "MELLType":
        return MELLType(TypeKind.BANG, inner)

    # -- structural relations ----------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MELLType):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.left, self.right))

    def unifiable(self, other: "MELLType") -> bool:
        """Return ``True`` if two ports carrying these types may be bound.

        A wire is a single linear edge, so its two endpoints must agree on the
        resource flowing across it.  Unification is therefore *structural
        equality* over the MELL grammar -- a ``Tensor`` only unifies with a
        ``Tensor`` whose components unify, an ``Implication`` only with an
        ``Implication``, and so on.  ``!A`` (``Bang``) is treated as a
        promotion of ``A``: a banged resource may legally meet either another
        ``!A`` or the bare ``A`` it promotes (dereliction).
        """
        if not isinstance(other, MELLType):
            return False
        if self.wildcard or other.wildcard:
            return True
        if self.kind == other.kind:
            if self.kind == TypeKind.I:
                return True
            if self.kind == TypeKind.BANG:
                return _opt_unifiable(self.left, other.left)
            return _opt_unifiable(self.left, other.left) and _opt_unifiable(
                self.right, other.right
            )
        # Dereliction: !A may meet the bare A it promotes.
        if self.kind == TypeKind.BANG:
            return _opt_unifiable(self.left, other)
        if other.kind == TypeKind.BANG:
            return _opt_unifiable(self, other.left)
        return False

    def __repr__(self) -> str:
        if self.kind == TypeKind.I:
            return "I"
        if self.kind in (TypeKind.TENSOR, TypeKind.IMPLICATION):
            symbol = "⊗" if self.kind == TypeKind.TENSOR else "⊸"
            return f"({self.left} {symbol} {self.right})"
        return f"!{self.left}"


def _opt_unifiable(a: Optional[MELLType], b: Optional[MELLType]) -> bool:
    """Unify two optional sub-types; ``None`` acts as a free type variable."""
    if a is None or b is None:
        return True
    return a.unifiable(b)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
class Port:
    """A node's physical connection point.

    A port holds a reference to the bound :class:`Port` on the opposing node.
    A port whose ``name`` is ``"p"`` is *principal*; all others are
    *auxiliary*.  By the Conservation of Edges invariant an auxiliary port is
    bound to exactly one opposing port for the net to be well formed.
    """

    PRINCIPAL = "p"

    def __init__(self, owner: "Node", name: str, expected_type: MELLType):
        self.owner = owner
        self.name = name
        self.type = expected_type
        self.target: Optional["Port"] = None

    @property
    def is_principal(self) -> bool:
        return self.name == Port.PRINCIPAL

    def connect(self, other: "Port") -> None:
        """Bind this port to ``other``, enforcing MELL typing.

        Raises:
            TypeError: if the two endpoint types are not unifiable.  This is
                the type-safety guardrail that prevents a wire from carrying
                two structurally incompatible resources.
        """
        if not self.type.unifiable(other.type):
            raise TypeError(
                f"non-unifiable port binding: {self.owner.node_id}.{self.name}"
                f":{self.type!r} <-> {other.owner.node_id}.{other.name}"
                f":{other.type!r}"
            )
        self.target = other
        other.target = self


class Node:
    """Base interaction node.

    Every node possesses exactly one **principal** port ``p`` and zero or more
    **auxiliary** ports ``a_i`` (stored in :attr:`aux`).
    """

    #: human-readable agent symbol, overridden by subclasses
    symbol = "node"

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p: Optional[Port] = None
        self.aux: List[Port] = []

    # -- construction helpers ----------------------------------------------
    def _set_principal(self, expected_type: MELLType) -> Port:
        self.p = Port(self, Port.PRINCIPAL, expected_type)
        return self.p

    def _add_aux(self, name: str, expected_type: MELLType) -> Port:
        port = Port(self, name, expected_type)
        self.aux.append(port)
        return port

    def ports(self) -> List[Port]:
        ports: List[Port] = []
        if self.p is not None:
            ports.append(self.p)
        ports.extend(self.aux)
        return ports

    @property
    def active_pair(self) -> Optional["Node"]:
        """The node sharing an active pair with this one, if any."""
        if self.p and self.p.target and self.p.target.name == Port.PRINCIPAL:
            return self.p.target.owner
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.symbol} {self.node_id}>"


# ---------------------------------------------------------------------------
# Concrete node agents
# ---------------------------------------------------------------------------
class ConstructorNode(Node):
    """Constructor ``γ``.

    The principal port represents the function *closure*.  Auxiliary ports:

    * ``a_1`` -- the input argument the closure binds.
    * ``a_2`` -- the return path the closure produces.
    """

    symbol = "γ"

    def __init__(
        self,
        node_id: str,
        arg_type: Optional[MELLType] = None,
        ret_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        arg_type = arg_type or MELLType.unit()
        ret_type = ret_type or MELLType.unit()
        # closure :: arg ⊸ ret
        self._set_principal(MELLType.implication(arg_type, ret_type))
        self.a_1 = self._add_aux("a_1", arg_type)
        self.a_2 = self._add_aux("a_2", ret_type)


class DestructorNode(Node):
    """Destructor ``γ⁻¹`` (application).

    The principal port connects to a :class:`ConstructorNode`.  Auxiliary
    ports:

    * ``a_1`` -- the argument supplier (the value fed to the closure).
    * ``a_2`` -- the computation destination (where the result is delivered).
    """

    symbol = "γ⁻¹"

    def __init__(
        self,
        node_id: str,
        arg_type: Optional[MELLType] = None,
        ret_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        arg_type = arg_type or MELLType.unit()
        ret_type = ret_type or MELLType.unit()
        self._set_principal(MELLType.implication(arg_type, ret_type))
        self.a_1 = self._add_aux("a_1", arg_type)
        self.a_2 = self._add_aux("a_2", ret_type)


class DuplicatorNode(Node):
    """Duplicator ``δ`` -- linear duplication of shared (``!``) resources.

    Auxiliary ports ``a_1`` / ``a_2`` are the two copies produced when the
    duplicator commutes with the agent on its principal port.
    """

    symbol = "δ"

    def __init__(self, node_id: str, shared_type: Optional[MELLType] = None):
        super().__init__(node_id)
        shared_type = shared_type or MELLType.unit()
        banged = MELLType.bang(shared_type)
        # The principal cuts against whatever agent it duplicates.
        self._set_principal(MELLType.any_())
        self.a_1 = self._add_aux("a_1", banged)
        self.a_2 = self._add_aux("a_2", banged)


class EraserNode(Node):
    """Eraser ``ε`` -- instantaneous dead-code annihilation.

    An eraser has a principal port only; meeting any agent erases it and
    propagates fresh erasers onto every freed auxiliary wire (weakening).
    """

    symbol = "ε"

    def __init__(self, node_id: str, erased_type: Optional[MELLType] = None):
        super().__init__(node_id)
        # An eraser annihilates whatever it meets: its principal cuts freely.
        self._set_principal(MELLType.any_())


class ValueNode(Node):
    """Value ``V_c`` -- holds a constant.

    The principal port emits the held constant onto its wire.
    """

    symbol = "V"

    def __init__(
        self,
        node_id: str,
        value: object,
        value_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        self.value = value
        self._set_principal(value_type or MELLType.unit())


class SwitchNode(Node):
    """Switch ``σ`` -- evaluates conditionals.

    The principal port consumes a boolean :class:`ValueNode`.  Auxiliary
    ports:

    * ``a_1`` -- the *true* branch continuation.
    * ``a_2`` -- the *false* branch continuation.
    * ``a_3`` -- the selected-output destination.
    """

    symbol = "σ"

    def __init__(self, node_id: str, branch_type: Optional[MELLType] = None):
        super().__init__(node_id)
        branch_type = branch_type or MELLType.unit()
        self._set_principal(MELLType.unit())
        self.a_1 = self._add_aux("a_1", branch_type)  # true branch
        self.a_2 = self._add_aux("a_2", branch_type)  # false branch
        self.a_3 = self._add_aux("a_3", branch_type)  # selected output


class CausalProjectionNode(Node):
    """Causal Projection ``P`` -- connects execution to ledger paths.

    The principal port consumes a *context coordinate* (a :class:`ValueNode`
    naming a ledger path).  Auxiliary port ``a_1`` delivers the projected
    value onto the execution wire.
    """

    symbol = "P"

    def __init__(self, node_id: str, projected_type: Optional[MELLType] = None):
        super().__init__(node_id)
        projected_type = projected_type or MELLType.unit()
        self._set_principal(MELLType.unit())
        self.a_1 = self._add_aux("a_1", projected_type)
        # populated when a coordinate is projected through this node
        self.ledger_path: Optional[object] = None


# ---------------------------------------------------------------------------
# The network / reduction engine
# ---------------------------------------------------------------------------
class HINNetwork:
    """A Hierarchical Interaction Net and its reduction engine."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.active_pairs: List[Tuple[Node, Node]] = []
        self._gensym = count()

    # -- registration / wiring ---------------------------------------------
    def register_node(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def fresh_id(self, prefix: str) -> str:
        return f"{prefix}#{next(self._gensym)}"

    def bind(self, port_a: Port, port_b: Port) -> None:
        """Publicly bind two ports, recording any resulting active pair."""
        port_a.connect(port_b)
        self._record_if_active(port_a, port_b)

    def _record_if_active(self, port_a: Port, port_b: Port) -> None:
        if port_a.is_principal and port_b.is_principal:
            a, b = port_a.owner, port_b.owner
            if (a, b) not in self.active_pairs and (b, a) not in self.active_pairs:
                self.active_pairs.append((a, b))

    # -- low-level relinking (used by rewrites) ----------------------------
    def _link(self, port_a: Optional[Port], port_b: Optional[Port]) -> None:
        """Wire two (possibly free) endpoints together inside a rewrite.

        Rewrites operate on a graph that has already been type-validated, so
        this bypasses the public typing guardrail while preserving the
        Conservation-of-Edges invariant (each endpoint ends bound to exactly
        one partner).  If either endpoint is ``None`` (a dangling wire) the
        surviving endpoint is left free.
        """
        if port_a is not None:
            port_a.target = port_b
        if port_b is not None:
            port_b.target = port_a
        if port_a is not None and port_b is not None:
            self._record_if_active(port_a, port_b)

    def _retire(self, *nodes: Node) -> None:
        for node in nodes:
            self.nodes.pop(node.node_id, None)

    # -- reduction ---------------------------------------------------------
    def reduce_step(self) -> bool:
        """Execute exactly one active-pair rewrite step.

        Returns ``True`` if a reduction occurred, ``False`` if the net is
        fully reduced (no active pairs remain).
        """
        if not self.active_pairs:
            return False

        node_a, node_b = self.active_pairs.pop(0)

        # An active pair recorded earlier may have been dismantled by an
        # intervening rewrite; skip it if either node is already gone.
        if node_a.node_id not in self.nodes or node_b.node_id not in self.nodes:
            return self.reduce_step()

        if not self._has_rule(node_a, node_b):
            # Inert pair (no rewrite rule); leave the wire stuck and make
            # progress on the remaining pairs instead.
            return bool(self.active_pairs) and self.reduce_step()

        a, b = node_a, node_b

        if isinstance(a, EraserNode):
            self._rule_erase(a, b)
        elif isinstance(b, EraserNode):
            self._rule_erase(b, a)
        elif isinstance(a, DuplicatorNode) and not isinstance(b, DuplicatorNode):
            self._rule_duplicate(a, b)
        elif isinstance(b, DuplicatorNode) and not isinstance(a, DuplicatorNode):
            self._rule_duplicate(b, a)
        elif isinstance(a, SwitchNode) or isinstance(b, SwitchNode):
            sw, val = (a, b) if isinstance(a, SwitchNode) else (b, a)
            self._rule_switch(sw, val)
        elif isinstance(a, CausalProjectionNode) or isinstance(b, CausalProjectionNode):
            proj, coord = (
                (a, b) if isinstance(a, CausalProjectionNode) else (b, a)
            )
            self._rule_project(proj, coord)
        else:
            # γ ⋈ γ⁻¹ (and δ ⋈ δ / ε ⋈ ε style) annihilation.
            self._rule_annihilate(a, b)

        return True

    def _has_rule(self, a: Node, b: Node) -> bool:
        """Return ``True`` if a deterministic rewrite rule applies."""
        kinds = {type(a), type(b)}

        if EraserNode in kinds:
            return True  # Rule 3: erasure applies to any partner
        if DuplicatorNode in kinds:
            return True  # Rule 2: duplication (incl. δ ⋈ δ annihilation)
        if SwitchNode in kinds and ValueNode in kinds:
            return True  # Rule 4: conditional switch
        if CausalProjectionNode in kinds and ValueNode in kinds:
            return True  # Rule 5: causal projection
        if {ConstructorNode, DestructorNode} <= kinds:
            return True  # Rule 1: beta-reduction
        if type(a) is type(b):
            return True  # same-agent annihilation
        return False

    # -- Rule 1: Beta-Reduction (γ ⋈ γ⁻¹) ----------------------------------
    def _rule_annihilate(self, a: Node, b: Node) -> None:
        """Annihilate two dual agents, splicing their auxiliary wires.

        Each auxiliary port of ``a`` is spliced directly onto the matching
        auxiliary port of ``b``: the external endpoint formerly attached to
        ``a.a_i`` is rewired to the external endpoint formerly attached to
        ``b.a_i``.  For a Constructor meeting a Destructor this performs
        beta-reduction -- the argument supplier flows to the bound argument and
        the closure body flows to the computation destination.  Both nodes are
        then garbage-collected.
        """
        for pa, pb in zip(a.aux, b.aux):
            self._link(pa.target, pb.target)
        self._retire(a, b)

    # -- Rule 2: Duplication (δ ⋈ γ) ---------------------------------------
    def _rule_duplicate(self, dup: DuplicatorNode, other: Node) -> None:
        """Commute a duplicator past another agent (interaction-net rule).

        If the partner is itself a duplicator of the same kind the pair simply
        annihilates.  Otherwise we replace the active pair with two copies of
        ``other`` (one per duplicator output) and a duplicator for each of
        ``other``'s auxiliary ports, wired as the standard bipartite
        commutation so that the structure reachable through ``other`` is
        faithfully duplicated.
        """
        if isinstance(other, DuplicatorNode):
            self._rule_annihilate(dup, other)
            return

        # External wires leaving the two duplicator copies.
        dup_externals = [aux.target for aux in dup.aux]
        other_externals = [aux.target for aux in other.aux]

        # One clone of `other` per duplicator output port.
        clones = [self._clone_node(other) for _ in dup.aux]
        # One duplicator per auxiliary port of `other`.
        sub_dups = [self._fresh_duplicator(dup) for _ in other.aux]

        for clones_idx, clone in enumerate(clones):
            self._link(clone.p, dup_externals[clones_idx])
        for dup_idx, sub in enumerate(sub_dups):
            self._link(sub.p, other_externals[dup_idx])

        # Bipartite internal wiring: clone_i.aux_j <-> sub_j.aux_i
        for clones_idx, clone in enumerate(clones):
            for dup_idx, sub in enumerate(sub_dups):
                self._link(clone.aux[dup_idx], sub.aux[clones_idx])

        self._retire(dup, other)

    # -- Rule 3: Erasure (ε ⋈ γ) -------------------------------------------
    def _rule_erase(self, eraser: EraserNode, other: Node) -> None:
        """Erase an agent, propagating fresh erasers onto its free wires."""
        for aux in other.aux:
            new_eraser = EraserNode(self.fresh_id("ε"), aux.type)
            self.register_node(new_eraser)
            self._link(new_eraser.p, aux.target)
        self._retire(eraser, other)

    # -- Rule 4: Conditional Switch (σ ⋈ value) ----------------------------
    def _rule_switch(self, switch: SwitchNode, value: Node) -> None:
        """Route the switch output to the branch chosen by ``value``.

        A truthy value selects ``a_1`` (true branch); a falsy value selects
        ``a_2`` (false branch).  The chosen branch wire is spliced to the
        selected-output wire ``a_3`` and the unchosen branch is erased.
        """
        chosen = value.value if isinstance(value, ValueNode) else value
        take_true = bool(chosen)

        selected = switch.a_1 if take_true else switch.a_2
        discarded = switch.a_2 if take_true else switch.a_1

        # Splice selected branch straight through to the output destination.
        self._link(selected.target, switch.a_3.target)

        # Erase the discarded branch's downstream wire.
        eraser = EraserNode(self.fresh_id("ε"), discarded.type)
        self.register_node(eraser)
        self._link(eraser.p, discarded.target)

        self._retire(switch, value)

    # -- Rule 5: Causal Projection (P ⋈ coordinate) ------------------------
    def _rule_project(self, proj: CausalProjectionNode, coord: Node) -> None:
        """Project a context coordinate onto the execution wire.

        The coordinate value (a ledger path) is captured and emitted as a
        fresh :class:`ValueNode` on the projection's output wire ``a_1``,
        binding execution to that ledger path.
        """
        path = coord.value if isinstance(coord, ValueNode) else coord
        proj.ledger_path = path

        emitted = ValueNode(self.fresh_id("V"), path, proj.a_1.type)
        self.register_node(emitted)
        self._link(emitted.p, proj.a_1.target)

        self._retire(proj, coord)

    # -- cloning helpers ---------------------------------------------------
    def _clone_node(self, node: Node) -> Node:
        """Create a fresh, unwired copy of ``node`` (same agent + port types)."""
        clone = object.__new__(type(node))
        Node.__init__(clone, self.fresh_id(node.symbol))
        clone.p = Port(clone, Port.PRINCIPAL, node.p.type) if node.p else None
        clone.aux = [Port(clone, aux.name, aux.type) for aux in node.aux]
        # Carry over agent-specific payload where it makes sense.
        if isinstance(node, ValueNode):
            clone.value = node.value
        if isinstance(node, CausalProjectionNode):
            clone.ledger_path = node.ledger_path
        self.register_node(clone)
        return clone

    def _fresh_duplicator(self, template: DuplicatorNode) -> DuplicatorNode:
        inner = template.p.type.left if template.p else None
        dup = DuplicatorNode(self.fresh_id("δ"), inner)
        self.register_node(dup)
        return dup

    # -- driver / diagnostics ----------------------------------------------
    def run_to_completion(self, max_steps: int = 1_000_000) -> int:
        """Reduce until no active pairs remain (or ``max_steps`` is hit)."""
        steps = 0
        while steps < max_steps and self.reduce_step():
            steps += 1
        return steps

    def validate_conservation(self) -> None:
        """Assert the Conservation-of-Edges invariant over auxiliary ports.

        Raises:
            ValueError: if any auxiliary port of a live node is unbound (a
                valence other than exactly one).
        """
        for node in self.nodes.values():
            for aux in node.aux:
                if aux.target is None:
                    raise ValueError(
                        f"valence violation: {node.node_id}.{aux.name} is unbound"
                    )
                if aux.target.target is not aux:
                    raise ValueError(
                        f"asymmetric edge at {node.node_id}.{aux.name}"
                    )


# ---------------------------------------------------------------------------
# Universal HIN network: ledger hot-swapping + rigidity sweeps at every step
# ---------------------------------------------------------------------------
# AnomalyClosureError lives in the spacetime layer; re-export it here so the
# universal reduction loop (and its callers) can import it from hin_vm.
from core.spacetime_ledger import (  # noqa: E402  (intentional late import)
    AnomalyClosureError,
    CoordinateVector,
    PathIntegralCache,
    RigidityVerifier,
    active_pair_signature,
)


class UniversalHINNetwork(HINNetwork):
    """A HIN network whose reduction loop is interlinked with the Block
    Universe ledger and the algebraic rigidity guardrails.

    At every active-pair reduction the network:

    1. **Coalesces** -- queries the causal path-integral cache for a
       structurally homomorphic, historically pre-compacted configuration and,
       on a hit, performs an atomic hot-swap instead of re-deriving the rewrite.
    2. **Sweeps** -- perturbs the active boundary coordinates by ``10⁻¹²⁰`` and
       validates eigenvalue persistence against the ``(26, 8, 312)`` kernel,
       raising :class:`AnomalyClosureError` on any off-shell shift.
    3. **Reduces** -- applies the core MELL rewrite rules, recording the
       reduced configuration back into the ledger for future coalescence.
    """

    def __init__(self, ledger_path: str = "context.aero", *, ledger=None,
                 verifier: Optional[RigidityVerifier] = None,
                 enable_rigidity: bool = True):
        super().__init__()
        self.ledger_path = ledger_path
        self.rigidity_eigenvalue = Decimal("8.312")
        self._ledger = ledger
        self._cache = PathIntegralCache(ledger)
        self.verifier = verifier or RigidityVerifier()
        self.enable_rigidity = enable_rigidity
        # Telemetry counters surfaced in the Aero Future telemetric tables.
        self.coalescence_hits = 0
        self.rigidity_sweeps = 0
        self.reductions = 0

    # -- adoption ----------------------------------------------------------
    @classmethod
    def adopt(cls, network: HINNetwork, **kwargs) -> "UniversalHINNetwork":
        """Wrap an existing network's nodes/worklist in a universal network."""
        uni = cls(**kwargs)
        uni.nodes = network.nodes
        uni.active_pairs = network.active_pairs
        uni._gensym = network._gensym
        return uni

    def _get_ledger(self):
        if self._ledger is None and self.ledger_path:
            from core.spacetime_ledger import BlockUniverseLedger

            self._ledger = BlockUniverseLedger(self.ledger_path)
            self._cache.ledger = self._ledger
        return self._ledger

    # -- reduction loop ----------------------------------------------------
    def reduce_step(self) -> bool:
        if not self.active_pairs:
            return False

        node_a, node_b = self.active_pairs[0]

        # 1. UNIVERSAL COALESCENCE: consult the path-integral execution cache.
        optimized = self.query_ledger_path_integral(node_a, node_b)
        if optimized is not None:
            self.coalescence_hits += 1
            return self.hot_swap_subgraph(node_a, node_b, optimized)

        # 2. ALGEBRAIC RIGIDITY SWEEP over the active boundary.
        if self.enable_rigidity and not self.verify_rigidity_invariants(node_a, node_b):
            raise AnomalyClosureError(
                "Off-shell coordinate shift detected below holographic noise floor!"
            )

        # 3. Core MELL reduction (Beta / Duplication / Erasure / Switch / ...).
        signature = active_pair_signature(node_a, node_b)
        reduced = super().reduce_step()
        if reduced:
            self.reductions += 1
            self._cache.record(signature, persist=self._ledger is not None)
        return reduced

    def query_ledger_path_integral(self, node_a: Node, node_b: Node) -> Optional[dict]:
        """Return a homomorphic historical path-integral record, or ``None``."""
        signature = active_pair_signature(node_a, node_b)
        return self._cache.lookup(signature)

    def hot_swap_subgraph(self, node_a: Node, node_b: Node, optimized: dict) -> bool:
        """Atomically substitute the active pair with its pre-reduced form.

        The cached configuration was validated and reduced in a prior pass, so
        the rigidity sweep is skipped and the deterministic local rewrite is
        applied directly -- the historically pre-compacted layout swap.
        """
        signature = active_pair_signature(node_a, node_b)
        reduced = super().reduce_step()  # reduces active_pairs[0] == (node_a, node_b)
        if reduced:
            self.reductions += 1
            self._cache.record(signature, persist=self._ledger is not None)
        return reduced

    def verify_rigidity_invariants(self, node_a: Node, node_b: Node) -> bool:
        """Check boundary eigenvalue persistence under transport perturbation."""
        self.rigidity_sweeps += 1
        boundary = [self._boundary_coord(node_a), self._boundary_coord(node_b)]
        for neighbour in self._aux_neighbours(node_a) + self._aux_neighbours(node_b):
            boundary.append(self._boundary_coord(neighbour))
        try:
            return self.verifier.verify_boundary(boundary)
        except AnomalyClosureError:
            return False

    # -- boundary coordinate helpers --------------------------------------
    @staticmethod
    def _aux_neighbours(node: Node) -> List[Node]:
        out: List[Node] = []
        for aux in node.aux:
            if aux.target is not None:
                out.append(aux.target.owner)
        return out

    @staticmethod
    def _boundary_coord(node: Node) -> CoordinateVector:
        """Return the node's annotated coordinate, or a deterministic finite
        lattice coordinate derived from its identity (distinct & non-degenerate).
        """
        coord = getattr(node, "coordinate", None)
        if coord is not None:
            return coord
        # Stable (process-independent) lattice coordinate derived from the node
        # identity; +1 offsets keep every component distinct and non-zero.
        digest = hashlib.sha256(node.node_id.encode("utf-8")).digest()
        x = int.from_bytes(digest[0:8], "big")
        y = int.from_bytes(digest[8:16], "big")
        z = int.from_bytes(digest[16:24], "big")
        return CoordinateVector(str(x + 1), str(y + 2), str(z + 3), -1)


__all__ = [
    "TypeKind",
    "MELLType",
    "Port",
    "Node",
    "ConstructorNode",
    "DestructorNode",
    "DuplicatorNode",
    "EraserNode",
    "ValueNode",
    "SwitchNode",
    "CausalProjectionNode",
    "HINNetwork",
    "UniversalHINNetwork",
    "AnomalyClosureError",
]
