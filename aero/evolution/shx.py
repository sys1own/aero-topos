
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class SearchHistoryDrivenCrossover:
    """SHX: Recombine successful historical configurations."""
    
    def __init__(self, historical_points: List[Dict[str, Any]], n_clusters: int = 5):
        """Cluster historical configurations and weight clusters by fitness.

        *historical_points* is a list of ``{"parameters": [...], "metrics":
        {...}}`` entries pulled from the block-universe ledger.  Successful
        regions of the search space (high ``speed_gain``) are identified so
        :meth:`select_offspring` can bias new candidates toward them.
        """
        self.dim = 0
        self.centroids: np.ndarray = np.empty((0, 0))
        self.cluster_fitness: np.ndarray = np.empty((0,))

        points: List[np.ndarray] = []
        fitnesses: List[float] = []
        for entry in historical_points:
            params = entry.get("parameters")
            if not params:
                continue
            points.append(np.asarray(params, dtype=float))
            metrics = entry.get("metrics") or {}
            fitnesses.append(float(metrics.get("speed_gain", 0.0)))

        if not points:
            return

        # Pad/truncate to a common dimensionality so ragged history is safe.
        self.dim = max(p.shape[0] for p in points)
        matrix = np.zeros((len(points), self.dim), dtype=float)
        for i, p in enumerate(points):
            matrix[i, : p.shape[0]] = p[: self.dim]
        fitness = np.asarray(fitnesses, dtype=float)

        k = max(1, min(int(n_clusters), len(points)))
        labels, centroids = self._kmeans(matrix, k)
        self.centroids = centroids

        # Mean fitness per cluster, normalised to non-negative weights.
        cluster_fitness = np.zeros(centroids.shape[0], dtype=float)
        for c in range(centroids.shape[0]):
            mask = labels == c
            cluster_fitness[c] = float(fitness[mask].mean()) if mask.any() else 0.0
        shifted = cluster_fitness - cluster_fitness.min()
        total = shifted.sum()
        self.cluster_fitness = (shifted / total) if total > 0 else np.ones_like(shifted) / len(shifted)

    # -- clustering --------------------------------------------------------

    @staticmethod
    def _kmeans(matrix: np.ndarray, k: int, iterations: int = 25, seed: int = 1234):
        """A small, deterministic Lloyd's k-means (no sklearn dependency)."""
        rng = np.random.default_rng(seed)
        n = matrix.shape[0]
        # k-means++ style spread for stable, well-separated seeds.
        first = rng.integers(0, n)
        centroids = [matrix[first].copy()]
        for _ in range(1, k):
            dists = np.min(
                [np.sum((matrix - c) ** 2, axis=1) for c in centroids], axis=0
            )
            total = dists.sum()
            if total <= 0:
                centroids.append(matrix[rng.integers(0, n)].copy())
                continue
            probs = dists / total
            centroids.append(matrix[rng.choice(n, p=probs)].copy())
        centroids = np.array(centroids, dtype=float)

        labels = np.zeros(n, dtype=int)
        for _ in range(iterations):
            new_labels = np.argmin(
                np.stack([np.sum((matrix - c) ** 2, axis=1) for c in centroids], axis=1),
                axis=1,
            )
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = matrix[mask].mean(axis=0)
        return labels, centroids

    # -- offspring selection ----------------------------------------------

    def _score(self, candidate: np.ndarray) -> float:
        """Score a candidate by fitness-weighted proximity to known-good clusters."""
        if self.centroids.shape[0] == 0:
            return 0.0
        vec = np.zeros(self.dim, dtype=float)
        flat = np.asarray(candidate, dtype=float).ravel()
        vec[: min(self.dim, flat.shape[0])] = flat[: self.dim]
        score = 0.0
        for c in range(self.centroids.shape[0]):
            distance = float(np.linalg.norm(vec - self.centroids[c]))
            affinity = 1.0 / (1.0 + distance)
            score += self.cluster_fitness[c] * affinity
        return score

    def select_offspring(
        self, candidate_pool: List[np.ndarray], population_size: int
    ) -> List[np.ndarray]:
        """Select the *population_size* most promising candidates from the pool.

        Candidates near high-fitness historical clusters rank first.  With no
        usable history the pool is returned head-truncated, so the caller's
        evolution loop degrades gracefully rather than stalling.
        """
        pool = list(candidate_pool)
        if not pool:
            return []
        if self.centroids.shape[0] == 0 or self.cluster_fitness.sum() == 0:
            return pool[:population_size]
        ranked = sorted(pool, key=self._score, reverse=True)
        return ranked[: max(0, population_size)]