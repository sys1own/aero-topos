"""Pareto frontier analysis with NSGA-II style non-dominated sorting."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


class ParetoOptimizer:
    """Multi-objective Pareto frontier analysis and selection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        evo = config.get("project", {}).get("evolutionary_bootstrap", {})
        pareto_cfg = evo.get("pareto_frontier", {})
        self.use_crowding = pareto_cfg.get("crowding_distance", True)
        self.use_elitism = pareto_cfg.get("elitism", True)

    def is_dominant(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Return True if *a* Pareto-dominates *b* (all objectives minimised)."""
        if not a or not b:
            return False
        keys = set(a.keys()) & set(b.keys())
        if not keys:
            return False
        at_least_one_better = False
        for k in keys:
            av, bv = a.get(k, float("inf")), b.get(k, float("inf"))
            if av > bv:
                return False
            if av < bv:
                at_least_one_better = True
        return at_least_one_better

    def analyze_frontier(self, population: Sequence[Any]) -> Dict[str, bool]:
        """Return a dict mapping individual.id -> dominated (True/False)."""
        dominated: Dict[str, bool] = {}
        for ind in population:
            dominated[ind.id] = False
        for i, a in enumerate(population):
            for j, b in enumerate(population):
                if i == j:
                    continue
                if self.is_dominant(b.fitness, a.fitness):
                    dominated[a.id] = True
                    break
        return dominated

    def _non_dominated_sort(self, population: Sequence[Any]) -> List[List[Any]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        domination_count = [0] * n
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.is_dominant(population[i].fitness, population[j].fitness):
                    dominated_set[i].append(j)
                elif self.is_dominant(population[j].fitness, population[i].fitness):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return [[population[i] for i in front] for front in fronts if front]

    def _crowding_distance(self, front: List[Any]) -> List[float]:
        n = len(front)
        if n <= 2:
            return [float("inf")] * n
        distances = [0.0] * n
        if not front[0].fitness:
            return distances
        objectives = list(front[0].fitness.keys())
        for obj in objectives:
            indices = sorted(range(n), key=lambda i: front[i].fitness.get(obj, float("inf")))
            distances[indices[0]] = float("inf")
            distances[indices[-1]] = float("inf")
            obj_min = front[indices[0]].fitness.get(obj, 0.0)
            obj_max = front[indices[-1]].fitness.get(obj, 0.0)
            span = obj_max - obj_min
            if span == 0:
                continue
            for k in range(1, n - 1):
                val_next = front[indices[k + 1]].fitness.get(obj, 0.0)
                val_prev = front[indices[k - 1]].fitness.get(obj, 0.0)
                distances[indices[k]] += (val_next - val_prev) / span
        return distances

    def select_best(self, population: Sequence[Any], count: int) -> List[Any]:
        fronts = self._non_dominated_sort(list(population))
        selected: List[Any] = []
        for front in fronts:
            if len(selected) + len(front) <= count:
                selected.extend(front)
            else:
                remaining = count - len(selected)
                if self.use_crowding and remaining > 0:
                    cd = self._crowding_distance(front)
                    ranked = sorted(zip(cd, front), key=lambda x: -x[0])
                    selected.extend(ind for _, ind in ranked[:remaining])
                else:
                    selected.extend(front[:remaining])
                break
        return selected

    def get_best(self, population: Sequence[Any]) -> Any:
        fronts = self._non_dominated_sort(list(population))
        if not fronts or not fronts[0]:
            return population[0] if population else None
        front = fronts[0]
        cd = self._crowding_distance(front)
        best_idx = int(np.argmax(cd))
        return front[best_idx]

    def tournament_select(self, candidates: Sequence[Any]) -> Any:
        if len(candidates) == 1:
            return candidates[0]
        dominated = self.analyze_frontier(candidates)
        non_dominated = [c for c in candidates if not dominated.get(c.id, False)]
        if non_dominated:
            return non_dominated[0]
        return candidates[0]
