"""Tests for the Self-Evolution Bootstrap Engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evolution.bootstrap import Individual, SelfEvolutionEngine
from src.evolution.genetic_operators import CrossoverEngine, MutationEngine
from src.evolution.fitness_functions import FitnessEvaluator
from src.evolution.pareto_frontier import ParetoOptimizer
from src.evolution.sandbox_manager import SandboxManager

_MINIMAL_CONFIG = {
    "project": {
        "evolutionary_bootstrap": {
            "enabled": True,
            "max_generations": 3,
            "population_size": 4,
            "fitness_objectives": {
                "compilation_latency": {"weight": 0.35, "metric": "wall_time_ms", "target": "< 1000"},
                "memory_peak_rss": {"weight": 0.35, "metric": "bytes", "target": "< 500MB"},
                "binary_footprint": {"weight": 0.30, "metric": "bytes", "target": "< 50MB"},
            },
            "mutation_vectors": {
                "parallel_compilation_threads": {"range": [1, 8], "step": 1, "default": 4},
                "ast_inlining_aggressiveness": {"range": [0, 100], "step": 10, "default": 50},
                "lto_threshold": {"range": [0, 100], "step": 5, "default": 75},
            },
            "pareto_frontier": {
                "enable_nondominated_sorting": True,
                "crowding_distance": True,
                "elitism": True,
            },
        }
    }
}


class TestMutationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = MutationEngine(_MINIMAL_CONFIG)

    def test_generate_random_genome(self):
        genome = self.engine.generate_random()
        self.assertIn("parallel_compilation_threads", genome)
        self.assertIn("ast_inlining_aggressiveness", genome)
        self.assertIn("lto_threshold", genome)

    def test_mutate_modifies_genome(self):
        genome = self.engine.generate_random()
        mutated = self.engine.mutate(genome)
        self.assertIsInstance(mutated, dict)
        self.assertEqual(set(genome.keys()), set(mutated.keys()))


class TestCrossoverEngine(unittest.TestCase):
    def test_crossover_produces_valid_child(self):
        engine = CrossoverEngine(_MINIMAL_CONFIG)
        parent_a = {"a": 1, "b": 2, "c": 3}
        parent_b = {"a": 10, "b": 20, "c": 30}
        child = engine.crossover(parent_a, parent_b)
        self.assertEqual(set(child.keys()), {"a", "b", "c"})
        for key in child:
            self.assertIn(child[key], [parent_a[key], parent_b[key]])


class TestFitnessEvaluator(unittest.TestCase):
    def test_evaluate_returns_weighted_metrics(self):
        evaluator = FitnessEvaluator(_MINIMAL_CONFIG)
        metrics = {"compilation_latency": 500, "memory_peak_rss": 100_000_000, "binary_footprint": 10_000_000}
        fitness = evaluator.evaluate(metrics)
        self.assertAlmostEqual(fitness["compilation_latency"], 500 * 0.35)
        self.assertAlmostEqual(fitness["memory_peak_rss"], 100_000_000 * 0.35)

    def test_weighted_score(self):
        evaluator = FitnessEvaluator(_MINIMAL_CONFIG)
        fitness = {"compilation_latency": 175, "memory_peak_rss": 35_000_000, "binary_footprint": 3_000_000}
        score = evaluator.weighted_score(fitness)
        self.assertGreater(score, 0)


class TestParetoOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = ParetoOptimizer(_MINIMAL_CONFIG)

    def test_is_dominant(self):
        a = {"x": 1, "y": 1}
        b = {"x": 2, "y": 2}
        self.assertTrue(self.optimizer.is_dominant(a, b))
        self.assertFalse(self.optimizer.is_dominant(b, a))

    def test_is_not_dominant_when_equal(self):
        a = {"x": 1, "y": 1}
        self.assertFalse(self.optimizer.is_dominant(a, a))

    def test_analyze_frontier(self):
        pop = [
            Individual(id="a", genome={}, fitness={"x": 1, "y": 1}),
            Individual(id="b", genome={}, fitness={"x": 2, "y": 2}),
            Individual(id="c", genome={}, fitness={"x": 1, "y": 3}),
        ]
        dominated = self.optimizer.analyze_frontier(pop)
        self.assertFalse(dominated["a"])
        self.assertTrue(dominated["b"])

    def test_select_best(self):
        pop = [
            Individual(id="a", genome={}, fitness={"x": 1, "y": 5}),
            Individual(id="b", genome={}, fitness={"x": 5, "y": 1}),
            Individual(id="c", genome={}, fitness={"x": 3, "y": 3}),
            Individual(id="d", genome={}, fitness={"x": 10, "y": 10}),
        ]
        selected = self.optimizer.select_best(pop, 2)
        self.assertEqual(len(selected), 2)

    def test_tournament_select(self):
        pop = [
            Individual(id="a", genome={}, fitness={"x": 1, "y": 1}),
            Individual(id="b", genome={}, fitness={"x": 10, "y": 10}),
        ]
        winner = self.optimizer.tournament_select(pop)
        self.assertEqual(winner.id, "a")


class TestSandboxManager(unittest.TestCase):
    def test_create_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SandboxManager(Path(tmp))
            path = mgr.create_sandbox("test1")
            self.assertTrue(path.exists())
            mgr.cleanup_sandbox("test1")
            self.assertFalse(path.exists())

    def test_cleanup_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SandboxManager(Path(tmp))
            mgr.create_sandbox("a")
            mgr.create_sandbox("b")
            mgr.cleanup_all()
            self.assertIsNone(mgr.get_sandbox_path("a"))
            self.assertIsNone(mgr.get_sandbox_path("b"))


class TestSelfEvolutionEngine(unittest.TestCase):
    def test_evolve_returns_best_individual(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(json.dumps(_MINIMAL_CONFIG))
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            engine = SelfEvolutionEngine(cfg_path, workspace)
            best = engine.evolve(max_generations=2)
            self.assertIsInstance(best, Individual)
            self.assertIn("compilation_latency", best.fitness)
            self.assertGreater(len(engine.history), 0)


if __name__ == "__main__":
    unittest.main()
