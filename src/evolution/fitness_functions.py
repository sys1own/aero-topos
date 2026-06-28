"""Fitness evaluation for evolutionary candidates."""

from __future__ import annotations

from typing import Any, Dict, List


class FitnessEvaluator:
    """Evaluates individuals against the configured fitness objectives."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        evo = config.get("project", {}).get("evolutionary_bootstrap", {})
        self.objectives = evo.get("fitness_objectives", {})

    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, float]:
        fitness: Dict[str, float] = {}
        for objective_name, spec in self.objectives.items():
            raw = metrics.get(objective_name, float("inf"))
            weight = float(spec.get("weight", 1.0))
            fitness[objective_name] = raw * weight
        return fitness

    def weighted_score(self, fitness: Dict[str, float]) -> float:
        if not fitness:
            return float("inf")
        total = 0.0
        for objective_name, spec in self.objectives.items():
            value = fitness.get(objective_name, float("inf"))
            if value == float("inf"):
                return float("inf")
            total += value
        return total
