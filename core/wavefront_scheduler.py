"""Topological wavefront scheduling for the Phase 5 execution engine.

A *wavefront* is a set of nodes that share the same dependency level in a DAG.
Nodes inside a single wave are independent and may execute in parallel; waves
are ordered by topological dependency.  ``WavefrontScheduler`` computes these
levels and supports incremental schedule repair after local graph mutations.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set

from core.hin_graph import GraphMutation, HINGraph


class SchedulerError(Exception):
    """Base exception for scheduling failures."""


class CycleError(SchedulerError):
    """Raised when the graph contains a cycle and cannot be levelised."""


@dataclass
class WavefrontScheduler:
    """Compute and incrementally update topological wavefront schedules."""

    def compute_wavefronts(self, graph: HINGraph) -> List[List[str]]:
        """Return level-ordered waves $W_k = \\{ v \\mid L(v) = k \\}$.

        Uses Kahn's algorithm grouped into breadth-first layers.
        """
        in_degree = dict(graph.in_degrees)
        queue: deque = deque(n for n in graph.nodes if in_degree.get(n, 0) == 0)
        waves: List[List[str]] = []

        while queue:
            wave: List[str] = []
            next_queue: deque = deque()
            for node in queue:
                wave.append(node)
                for nbr in graph.adj_list.get(node, []):
                    in_degree[nbr] -= 1
                    if in_degree[nbr] == 0:
                        next_queue.append(nbr)
            waves.append(sorted(wave))
            queue = next_queue

        if any(d > 0 for d in in_degree.values()):
            raise CycleError("Graph contains a cycle; cannot compute wavefront schedule")

        return waves

    def update_schedule(
        self,
        graph: HINGraph,
        old_schedule: List[List[str]],
        mutations: List[GraphMutation],
    ) -> List[List[str]]:
        """Repair a schedule after a local mutation without a full rebuild.

        1. Identify the *influence zone*: mutated nodes plus all ancestors and
           descendants reachable through the directed dependency graph.
        2. Recompute dependency levels only for the influence zone, anchoring to
           the unchanged levels of nodes outside the zone.
        3. Merge the unchanged and recomputed levels into a new schedule.
        """
        if not mutations:
            return old_schedule

        affected: Set[str] = set()
        for m in mutations:
            if m.node_id:
                affected.add(m.node_id)
            if m.edge:
                affected.update(m.edge)

        # Expand the influence zone both up- and downstream.
        stack = list(affected)
        while stack:
            node = stack.pop()
            for nbr in graph.adj_list.get(node, []):
                if nbr not in affected:
                    affected.add(nbr)
                    stack.append(nbr)
            for pred in graph.reverse_adj.get(node, []):
                if pred not in affected:
                    affected.add(pred)
                    stack.append(pred)

        # Snapshot the old levels for the untouched nodes.
        old_levels: Dict[str, int] = {}
        for level, wave in enumerate(old_schedule):
            for node in wave:
                old_levels[node] = level

        # Base level for an affected node is constrained by its unaffected
        # predecessors.
        base_level: Dict[str, int] = {}
        for node in affected:
            base = 0
            for pred in graph.reverse_adj.get(node, []):
                if pred not in affected:
                    base = max(base, old_levels.get(pred, 0) + 1)
            base_level[node] = base

        # In-degree within the affected subgraph.
        sub_indeg: Dict[str, int] = {}
        for node in affected:
            count = 0
            for pred in graph.reverse_adj.get(node, []):
                if pred in affected:
                    count += 1
            sub_indeg[node] = count

        new_level: Dict[str, int] = {}
        queue: deque = deque()
        for node in affected:
            new_level[node] = base_level[node]
            if sub_indeg[node] == 0:
                queue.append(node)

        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for nbr in graph.adj_list.get(node, []):
                if nbr not in affected:
                    continue
                candidate = new_level[node] + 1
                if candidate > new_level[nbr]:
                    new_level[nbr] = candidate
                sub_indeg[nbr] -= 1
                if sub_indeg[nbr] == 0:
                    queue.append(nbr)

        if processed != len(affected):
            raise CycleError("Affected subgraph contains a cycle")

        # Reassemble the schedule.  Unchanged nodes keep their old levels;
        # affected nodes are placed at their recomputed levels.
        levels: Dict[int, List[str]] = defaultdict(list)
        max_level = 0
        for node, level in old_levels.items():
            if node not in affected:
                levels[level].append(node)
                max_level = max(max_level, level)
        for node, level in new_level.items():
            levels[level].append(node)
            max_level = max(max_level, level)

        return [sorted(levels[i]) for i in range(max_level + 1)]
