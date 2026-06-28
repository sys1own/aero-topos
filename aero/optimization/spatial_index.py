"""Vantage-Point Tree for fast similarity search over the block universe.

The evolution loop indexes every historically-explored parameter vector so it
can ask *"have we already searched near here?"* in logarithmic time instead of
scanning the whole ledger.  The tree is metric-agnostic and ships with cosine
and euclidean distances.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    similarity = float(np.dot(a, b) / (na * nb))
    # Cosine distance in [0, 2]; clamp guards against fp drift.
    return 1.0 - max(-1.0, min(1.0, similarity))


_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "euclidean": _euclidean,
    "cosine": _cosine,
}


class _Node:
    __slots__ = ("index", "threshold", "inside", "outside")

    def __init__(self, index: int) -> None:
        self.index = index
        self.threshold: float = 0.0
        self.inside: Optional["_Node"] = None
        self.outside: Optional["_Node"] = None


class VPTree:
    """A Vantage-Point Tree over a list of ``{"parameters": [...]}`` points."""

    def __init__(
        self,
        points: Sequence[Dict[str, Any]],
        distance_metric: str = "euclidean",
        seed: int = 7,
    ) -> None:
        if distance_metric not in _METRICS:
            raise ValueError(f"Unknown distance_metric: {distance_metric!r}")
        self.metric_name = distance_metric
        self._distance = _METRICS[distance_metric]
        self._rng = random.Random(seed)

        self.points: List[Dict[str, Any]] = list(points)
        if self.points:
            dim = max(len(p.get("parameters", [])) for p in self.points)
            self._vectors = np.zeros((len(self.points), dim), dtype=float)
            for i, p in enumerate(self.points):
                params = np.asarray(p.get("parameters", []), dtype=float)
                self._vectors[i, : params.shape[0]] = params[:dim]
            self.dim = dim
        else:
            self._vectors = np.empty((0, 0))
            self.dim = 0

        self._root = self._build(list(range(len(self.points))))

    # -- construction ------------------------------------------------------

    def _build(self, indices: List[int]) -> Optional[_Node]:
        if not indices:
            return None
        # Choose a random vantage point for balanced expected depth.
        pivot_pos = self._rng.randrange(len(indices))
        indices[0], indices[pivot_pos] = indices[pivot_pos], indices[0]
        vantage = indices[0]
        rest = indices[1:]
        node = _Node(vantage)
        if not rest:
            return node

        distances = [(self._dist_idx(vantage, i), i) for i in rest]
        distances.sort(key=lambda t: t[0])
        median = len(distances) // 2
        node.threshold = distances[median][0]

        inside = [i for d, i in distances if d < node.threshold]
        outside = [i for d, i in distances if d >= node.threshold]
        node.inside = self._build(inside)
        node.outside = self._build(outside)
        return node

    # -- distance helpers --------------------------------------------------

    def _to_vector(self, point: Any) -> np.ndarray:
        if isinstance(point, dict):
            params = point.get("parameters", [])
        else:
            params = point
        vec = np.zeros(self.dim, dtype=float)
        arr = np.asarray(params, dtype=float).ravel()
        vec[: min(self.dim, arr.shape[0])] = arr[: self.dim]
        return vec

    def _dist_idx(self, a: int, b: int) -> float:
        return self._distance(self._vectors[a], self._vectors[b])

    # -- queries -----------------------------------------------------------

    def query(self, point: Any, k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """Return the *k* nearest stored points as ``(distance, point)`` tuples."""
        if self._root is None or k <= 0:
            return []
        target = self._to_vector(point)
        heap: List[Tuple[float, int]] = []  # (-distance, index) max-heap by neg dist
        self._search(self._root, target, k, heap)
        results = sorted(((-negd, idx) for negd, idx in heap), key=lambda t: t[0])
        return [(dist, self.points[idx]) for dist, idx in results]

    def _search(
        self,
        node: Optional[_Node],
        target: np.ndarray,
        k: int,
        heap: List[Tuple[float, int]],
    ) -> None:
        import heapq

        if node is None:
            return
        distance = self._distance(self._vectors[node.index], target)
        if len(heap) < k:
            heapq.heappush(heap, (-distance, node.index))
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, (-distance, node.index))

        worst = -heap[0][0] if len(heap) >= k else float("inf")
        if distance < node.threshold:
            self._search(node.inside, target, k, heap)
            if distance + worst >= node.threshold:
                self._search(node.outside, target, k, heap)
        else:
            self._search(node.outside, target, k, heap)
            if distance - worst <= node.threshold:
                self._search(node.inside, target, k, heap)

    def nearest_distance(self, point: Any) -> Optional[float]:
        """Distance to the single closest stored point, or ``None`` if empty."""
        hits = self.query(point, k=1)
        return hits[0][0] if hits else None

    def __len__(self) -> int:
        return len(self.points)
