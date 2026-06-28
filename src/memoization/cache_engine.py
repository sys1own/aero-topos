"""
Memoization Engine.

Implements a query-driven, Salsa-style incremental computation cache with
AST-node-level granularity, semantic fingerprinting, dependency tracking,
and automatic invalidation on file or config changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("memoization.cache_engine")


@dataclass
class CacheEntry:
    """A single cached computation result."""

    key: str
    value: Any
    fingerprint: str
    timestamp: float
    dependencies: List[str] = field(default_factory=list)
    version: int = 1


class QueryCache:
    """
    In-memory + disk-backed query cache with dependency tracking.

    Supports semantic-delta invalidation: when a file changes, only entries
    whose *semantic fingerprint* actually differs are evicted.
    """

    def __init__(self, storage_path: Path, max_size_gb: float = 20.0) -> None:
        self.storage_path = storage_path
        self.max_size_bytes = int(max_size_gb * 1024 ** 3)
        self._memory: Dict[str, CacheEntry] = {}
        self._dep_graph: nx.DiGraph = nx.DiGraph()
        self._hit_count = 0
        self._miss_count = 0
        storage_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        entry = self._memory.get(key)
        if entry is not None:
            self._hit_count += 1
            return entry.value

        disk_entry = self._load_from_disk(key)
        if disk_entry is not None:
            self._memory[key] = disk_entry
            self._hit_count += 1
            return disk_entry.value

        self._miss_count += 1
        return None

    def put(
        self,
        key: str,
        value: Any,
        fingerprint: str,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        entry = CacheEntry(
            key=key,
            value=value,
            fingerprint=fingerprint,
            timestamp=time.time(),
            dependencies=dependencies or [],
        )
        self._memory[key] = entry
        self._save_to_disk(entry)

        for dep in entry.dependencies:
            self._dep_graph.add_edge(dep, key)

    def invalidate(self, key: str) -> List[str]:
        """Invalidate a key and all transitive dependents. Returns evicted keys."""
        evicted: List[str] = []
        if key in self._memory:
            del self._memory[key]
            evicted.append(key)
            self._remove_from_disk(key)

        if self._dep_graph.has_node(key):
            try:
                dependents = list(nx.descendants(self._dep_graph, key))
            except nx.NetworkXError:
                dependents = []
            for dep in dependents:
                if dep in self._memory:
                    del self._memory[dep]
                    evicted.append(dep)
                    self._remove_from_disk(dep)
        return evicted

    def invalidate_by_fingerprint(self, key: str, new_fingerprint: str) -> bool:
        """Invalidate only if the fingerprint actually changed (semantic delta)."""
        entry = self._memory.get(key)
        if entry and entry.fingerprint == new_fingerprint:
            return False
        self.invalidate(key)
        return True

    def clear(self) -> None:
        self._memory.clear()
        self._dep_graph.clear()
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path, ignore_errors=True)
            self.storage_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        return {
            "entries": len(self._memory),
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / total if total else 0.0,
            "dependency_edges": self._dep_graph.number_of_edges(),
        }

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _cache_file(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.storage_path / f"{safe}.json"

    def _save_to_disk(self, entry: CacheEntry) -> None:
        path = self._cache_file(entry.key)
        try:
            serialisable_value = entry.value
            try:
                json.dumps(serialisable_value)
            except (TypeError, ValueError):
                serialisable_value = str(serialisable_value)

            data = {
                "key": entry.key,
                "value": serialisable_value,
                "fingerprint": entry.fingerprint,
                "timestamp": entry.timestamp,
                "dependencies": entry.dependencies,
                "version": entry.version,
            }
            path.write_text(json.dumps(data))
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist cache entry %r to disk: %s", entry.key, exc)

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        path = self._cache_file(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CacheEntry(
                key=data["key"],
                value=data["value"],
                fingerprint=data["fingerprint"],
                timestamp=data["timestamp"],
                dependencies=data.get("dependencies", []),
                version=data.get("version", 1),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Failed to load cache entry %r from disk: %s", key, exc)
            return None

    def _remove_from_disk(self, key: str) -> None:
        path = self._cache_file(key)
        path.unlink(missing_ok=True)


class MemoizationEngine:
    """
    High-level memoization engine that wraps ``QueryCache`` and provides
    semantic-fingerprint computation, dependency tracking with cycle
    detection, and file-change-driven cache invalidation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        memo_cfg = config.get("memoization", {})
        storage_cfg = memo_cfg.get("storage", {})
        self.strategies = memo_cfg.get("strategies", {})
        self.dep_cfg = memo_cfg.get("dependency_tracking", {})
        self.invalidation_cfg = memo_cfg.get("cache_invalidation", {})

        storage_path = Path(storage_cfg.get("path", ".aero/query_cache"))
        max_size = float(storage_cfg.get("max_cache_size_gb", 20))
        self.cache = QueryCache(storage_path, max_size)
        self._file_fingerprints: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def memoize(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        dependencies: Optional[List[str]] = None,
        source_file: Optional[str] = None,
    ) -> Any:
        """Return cached value or compute, cache, and return."""
        fingerprint = self._compute_fingerprint(key, source_file)
        cached = self.cache.get(key)
        if cached is not None:
            entry = self.cache._memory.get(key)
            if entry and entry.fingerprint == fingerprint:
                return cached

        value = compute_fn()
        self.cache.put(key, value, fingerprint, dependencies)
        return value

    def on_file_changed(self, file_path: str) -> List[str]:
        """Invalidate cache entries affected by a file change."""
        strategy = self.invalidation_cfg.get("on_file_change", "semantic_delta")
        new_fp = self._file_semantic_fingerprint(file_path)
        old_fp = self._file_fingerprints.get(file_path)
        self._file_fingerprints[file_path] = new_fp

        if strategy == "semantic_delta" and old_fp == new_fp:
            return []

        return self.cache.invalidate(file_path)

    def on_config_changed(self) -> None:
        strategy = self.invalidation_cfg.get("on_config_change", "full_rebuild")
        if strategy == "full_rebuild":
            self.cache.clear()

    def check_dependency_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        if not self.dep_cfg.get("enable_cycle_detection", True):
            return []
        try:
            return list(nx.simple_cycles(self.cache._dep_graph))
        except nx.NetworkXError:
            return []

    def stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    def _compute_fingerprint(self, key: str, source_file: Optional[str] = None) -> str:
        parts = [key]
        if source_file:
            parts.append(self._file_semantic_fingerprint(source_file))
        raw = ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _file_semantic_fingerprint(self, file_path: str) -> str:
        """Fingerprint a file's semantic content (ignoring whitespace and comments)."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Cannot read %s for fingerprinting: %s", file_path, exc)
            return "missing"

        if self.strategies.get("ignore_comments_and_whitespace", True):
            content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            content = re.sub(r"\s+", " ", content).strip()

        return hashlib.sha256(content.encode()).hexdigest()[:32]
