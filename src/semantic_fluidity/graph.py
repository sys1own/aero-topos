"""
The system graph: maps every domain and invariant into a single
``networkx.DiGraph`` (mirroring the UAST pattern already used by
:class:`src.analysis.semantic_mapper.SemanticMapper`), so unrelated domains
stay non-conflicting -- each domain is its own node, each invariant is
namespaced under its domain -- while still being connected as interdependent
nodes in one graph rather than living in separate, disconnected structures.

Two kinds of edges exist:

* ``defines``    -- a domain node to one of its invariant nodes.
* ``references`` -- an equation/boundary node to a state-variable node it uses,
  *within the same domain*.
* ``shared_symbol`` -- a weak, automatically-detected link between two
  state variables in *different* domains that happen to share a bare symbol
  name (e.g. ``rate`` in both ``genomics`` and ``game_engine``).  This never
  merges the two nodes -- it only records that the synthesis layer may want to
  treat them as related when reasoning across domains.
* ``maps_to``    -- an explicit, caller-declared cross-domain relationship
  (see :meth:`SystemGraph.link_domains`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from src.semantic_fluidity.schema import InvariantSchema


class SystemGraph:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_domain(self, domain: str) -> str:
        node_id = self._domain_node_id(domain)
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, node_type="domain", domain=domain)
        return node_id

    def add_schema(self, schema: InvariantSchema) -> None:
        """Add every invariant in ``schema`` as a node, wired to its domain node."""
        for domain in schema.domains:
            self.add_domain(domain)
        for kind, items in (
            ("state_variable", schema.state_variables),
            ("boundary", schema.boundaries),
            ("equation", schema.equations),
        ):
            for item in items:
                self._add_invariant_node(item, kind)
        self._link_references_within_domains(schema)

    def _add_invariant_node(self, item: Any, kind: str) -> None:
        domain_node = self.add_domain(item.domain)
        self.graph.add_node(
            item.id,
            node_type=kind,
            domain=item.domain,
            symbol=item.symbol,
            data=item.to_dict(),
        )
        self.graph.add_edge(domain_node, item.id, edge_type="defines")

    def _link_references_within_domains(self, schema: InvariantSchema) -> None:
        symbols_by_domain: Dict[str, Dict[str, str]] = {}
        for var in schema.state_variables:
            symbols_by_domain.setdefault(var.domain, {})[var.symbol] = var.id

        for bucket in (schema.boundaries, schema.equations):
            for item in bucket:
                domain_symbols = symbols_by_domain.get(item.domain, {})
                for variable_name in item.variables:
                    target = domain_symbols.get(variable_name)
                    if target and target != item.id and self.graph.has_node(target):
                        self.graph.add_edge(item.id, target, edge_type="references", variable=variable_name)

    # ------------------------------------------------------------------
    # Cross-domain interdependence
    # ------------------------------------------------------------------

    def link_domains(self, domain_a: str, symbol_a: str, domain_b: str, symbol_b: str, relation: str = "maps_to") -> None:
        """Explicitly mark two invariants in different domains as interdependent.

        Both nodes keep their own namespaced ids and definitions -- this only
        adds an edge, it never merges or renames either node.
        """
        from src.semantic_fluidity.schema import make_id

        node_a, node_b = make_id(domain_a, symbol_a), make_id(domain_b, symbol_b)
        for node in (node_a, node_b):
            if not self.graph.has_node(node):
                raise KeyError(f"no invariant node '{node}' in the system graph")
        self.graph.add_edge(node_a, node_b, edge_type=relation)

    def auto_link_shared_symbols(self) -> int:
        """Heuristically link state variables across *different* domains that
        share a bare symbol name.  Returns the number of edges added.
        """
        by_symbol: Dict[str, List[str]] = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "state_variable":
                by_symbol.setdefault(data["symbol"], []).append(node_id)

        added = 0
        for node_ids in by_symbol.values():
            if len(node_ids) < 2:
                continue
            for i, node_a in enumerate(node_ids):
                for node_b in node_ids[i + 1 :]:
                    if self.graph.nodes[node_a]["domain"] == self.graph.nodes[node_b]["domain"]:
                        continue
                    self.graph.add_edge(node_a, node_b, edge_type="shared_symbol")
                    self.graph.add_edge(node_b, node_a, edge_type="shared_symbol")
                    added += 2
        return added

    # ------------------------------------------------------------------
    # Introspection / export
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        domains = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "domain"]
        by_type: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            by_type[node_type] = by_type.get(node_type, 0) + 1
        return {
            "domain_count": len(domains),
            "domains": [self.graph.nodes[n]["domain"] for n in domains],
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "nodes_by_type": by_type,
            "cross_domain_edges": sum(
                1
                for u, v in self.graph.edges()
                if self.graph.nodes[u].get("domain") != self.graph.nodes[v].get("domain")
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        nodes = [{"id": node_id, **data} for node_id, data in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _domain_node_id(domain: str) -> str:
        return f"domain::{domain}"
