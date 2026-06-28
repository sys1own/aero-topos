"""
Self-Evolution Bootstrap Engine.

Implements the evolutionary optimisation loop with Pareto frontier analysis,
sandboxed evaluation, and multi-objective fitness tracking.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("evolution.bootstrap")

import numpy as np

from src.evolution.genetic_operators import CrossoverEngine, MutationEngine
from src.evolution.fitness_functions import FitnessEvaluator
from src.evolution.pareto_frontier import ParetoOptimizer
from src.evolution.sandbox_manager import SandboxManager


@dataclass
class Individual:
    """Represents an evolutionary candidate."""

    id: str
    genome: Dict[str, Any]
    fitness: Dict[str, float]
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    compilation_time: float = 0.0
    memory_peak: float = 0.0
    binary_size: float = 0.0


class SelfEvolutionEngine:
    """
    Main engine for self-evolution of the build system.

    Implements the evolutionary bootstrap loop with sandboxing and
    Pareto optimisation (NSGA-II style).
    """

    def __init__(self, blueprint_path: Path, workspace: Path) -> None:
        self.blueprint_path = blueprint_path
        self.workspace = workspace
        self.population: List[Individual] = []
        self.generation = 0
        self.best_candidate: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []

        with open(blueprint_path) as f:
            self.config = json.load(f)

        evo_config = self.config.get("project", {}).get("evolutionary_bootstrap", {})
        self.max_generations = evo_config.get("max_generations", 50)
        self.population_size = evo_config.get("population_size", 16)

        self.sandbox = SandboxManager(workspace, self.config)
        self.mutation_engine = MutationEngine(self.config)
        self.crossover_engine = CrossoverEngine(self.config)
        self.fitness_evaluator = FitnessEvaluator(self.config)
        self.pareto_optimizer = ParetoOptimizer(self.config)

        # Optional runtime-feedback and validation hooks (features #3, #5).
        # Both default to off; injectable callables keep the engine testable and
        # let real builds plug in benchmark/validation execution.
        from src.runtime.feedback import RuntimeFeedback
        from src.validation.validator import Validator

        self.runtime_feedback = RuntimeFeedback(self.config)
        self.validator = Validator(self.config)
        # runtime_evaluator: genome -> RuntimeMetrics ; validation_hook: genome -> bool
        self.runtime_evaluator: Optional[Any] = None
        self.validation_hook: Optional[Any] = None

    def evolve(self, max_generations: Optional[int] = None) -> Individual:
        """Run the evolutionary bootstrap loop synchronously."""
        limit = max_generations or self.max_generations
        self.population = self._initialize_population()

        for gen in range(limit):
            self.generation = gen
            self._evaluate_population()

            dominated = self.pareto_optimizer.analyze_frontier(self.population)
            parents = self._select_parents(dominated)
            offspring = self._generate_offspring(parents)
            self._evaluate_population(offspring)
            self.population = self._update_population(offspring)

            current_best = self._get_best_individual()
            if current_best and (
                self.best_candidate is None
                or self._is_dominant(current_best, self.best_candidate)
            ):
                self.best_candidate = current_best

            self._save_checkpoint()
            self.history.append(
                {
                    "generation": gen,
                    "best_id": self.best_candidate.id if self.best_candidate else None,
                    "best_fitness": dict(self.best_candidate.fitness) if self.best_candidate else {},
                    "population_size": len(self.population),
                }
            )

        if self.best_candidate is None and self.population:
            self.best_candidate = self.population[0]
        assert self.best_candidate is not None
        return self.best_candidate

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _initialize_population(self) -> List[Individual]:
        population: List[Individual] = []
        for i in range(self.population_size):
            genome = self.mutation_engine.generate_random()
            population.append(
                Individual(id=f"gen0_{i:04d}", genome=genome, fitness={})
            )
        return population

    def _evaluate_population(
        self, individuals: Optional[List[Individual]] = None
    ) -> None:
        target = individuals if individuals is not None else self.population
        for individual in target:
            if not individual.fitness:
                individual.fitness = self._evaluate_individual(individual)

    def _evaluate_individual(self, individual: Individual) -> Dict[str, float]:
        """Evaluate a single candidate by simulating compilation metrics."""
        genome = individual.genome
        threads = genome.get("parallel_compilation_threads", 8)
        inlining = genome.get("ast_inlining_aggressiveness", 50)
        lto = genome.get("lto_threshold", 75)

        compilation_latency = max(
            50.0,
            1000.0 - threads * 40.0 - inlining * 2.0 + random.gauss(0, 30),
        )
        memory_peak = max(
            50_000_000.0,
            200_000_000.0 + inlining * 1_000_000 - threads * 5_000_000 + random.gauss(0, 10_000_000),
        )
        binary_size = max(
            1_000_000.0,
            20_000_000.0 - lto * 100_000 + random.gauss(0, 500_000),
        )

        # Numerical-library choices shift the trade-off surface so the engine can
        # evolve toward the best BLAS/LAPACK/MPI/CUDA combination (feature #2).
        lat_mult, mem_mult, size_mult = self._library_effect(genome)
        compilation_latency *= lat_mult
        memory_peak *= mem_mult
        binary_size *= size_mult

        individual.compilation_time = compilation_latency
        individual.memory_peak = memory_peak
        individual.binary_size = binary_size

        fitness = {
            "compilation_latency": compilation_latency,
            "memory_peak_rss": memory_peak,
            "binary_footprint": binary_size,
        }

        # Validation gatekeeper (feature #5): a candidate that fails validation
        # is given worst-possible fitness so it is dominated and never reaches
        # the Pareto front.
        if self.validation_hook is not None and not self.validation_hook(genome):
            individual.fitness = {key: float("inf") for key in fitness}
            return individual.fitness

        # Runtime feedback (feature #3): blend measured runtime objectives into
        # the fitness so candidates are tuned for the real workload.
        if self.runtime_evaluator is not None:
            try:
                metrics = self.runtime_evaluator(genome)
                fitness = self.runtime_feedback.blend_into_fitness(fitness, metrics)
            except Exception as exc:
                logger.warning("Runtime evaluator failed for genome, skipping feedback blending: %s", exc)

        return fitness

    @staticmethod
    def _library_effect(genome: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return (latency, memory, size) multipliers for a genome's libraries.

        Optimised numerical backends accelerate compute (lower latency) at the
        cost of larger binaries / higher memory, giving the Pareto search a real
        trade-off to explore.  Genomes without library genes are unaffected.
        """
        lat, mem, size = 1.0, 1.0, 1.0
        blas = str(genome.get("blas", "none")).lower()
        if blas == "mkl":
            lat, mem, size = lat * 0.85, mem * 1.15, size * 1.25
        elif blas == "openblas":
            lat, mem, size = lat * 0.90, mem * 1.05, size * 1.10
        lapack = str(genome.get("lapack", "none")).lower()
        if lapack in ("mkl", "openblas"):
            lat *= 0.97
        if str(genome.get("cuda", "none")).lower() == "auto":
            lat, size = lat * 0.80, size * 1.15
        if str(genome.get("mpi_flavor", "none")).lower() in ("openmpi", "mpich"):
            lat, mem = lat * 0.95, mem * 1.10
        return lat, mem, size

    def _select_parents(self, dominated: Dict[str, bool]) -> List[Individual]:
        candidates = [ind for ind in self.population if not dominated.get(ind.id, False)]
        if not candidates:
            candidates = list(self.population)
        parents: List[Individual] = []
        for _ in range(self.population_size // 2):
            sample = random.sample(candidates, min(3, len(candidates)))
            winner = self.pareto_optimizer.tournament_select(sample)
            parents.append(winner)
        return parents

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        for i in range(0, len(parents) - 1, 2):
            child_genome = self.crossover_engine.crossover(
                parents[i].genome, parents[i + 1].genome
            )
            child_genome = self.mutation_engine.mutate(child_genome)
            offspring.append(
                Individual(
                    id=f"gen{self.generation + 1}_{len(offspring):04d}",
                    genome=child_genome,
                    fitness={},
                    parent_ids=[parents[i].id, parents[i + 1].id],
                )
            )
        while len(offspring) < self.population_size:
            genome = self.mutation_engine.generate_random()
            offspring.append(
                Individual(
                    id=f"gen{self.generation + 1}_{len(offspring):04d}",
                    genome=genome,
                    fitness={},
                )
            )
        return offspring

    def _update_population(self, offspring: List[Individual]) -> List[Individual]:
        combined = list(self.population) + list(offspring)
        return self.pareto_optimizer.select_best(combined, self.population_size)

    def _is_dominant(self, a: Individual, b: Individual) -> bool:
        return self.pareto_optimizer.is_dominant(a.fitness, b.fitness)

    def _get_best_individual(self) -> Optional[Individual]:
        if not self.population:
            return None
        return self.pareto_optimizer.get_best(self.population)

    def _save_checkpoint(self) -> None:
        checkpoint_dir = self.workspace / ".aero" / "evolution_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_gen{self.generation}.json"
        data = {
            "generation": self.generation,
            "population": [
                {
                    "id": ind.id,
                    "genome": ind.genome,
                    "fitness": ind.fitness,
                    "parent_ids": ind.parent_ids,
                }
                for ind in self.population
            ],
            "best_candidate": (
                {
                    "id": self.best_candidate.id,
                    "genome": self.best_candidate.genome,
                    "fitness": self.best_candidate.fitness,
                }
                if self.best_candidate
                else None
            ),
        }
        path.write_text(json.dumps(data, indent=2))
