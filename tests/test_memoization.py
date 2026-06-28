"""Tests for the Memoization Engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.memoization.cache_engine import CacheEntry, MemoizationEngine, QueryCache

_CONFIG = {
    "memoization": {
        "engine": "query-driven-salsa",
        "granularity": "ast-node-level",
        "storage": {"path": "", "max_cache_size_gb": 1, "compression": "zstd", "backend": "lmdb"},
        "strategies": {
            "ignore_comments_and_whitespace": True,
            "propagate_semantic_signatures_only": True,
            "cross_file_type_propagation": True,
            "incremental_parsing": True,
            "fingerprint_versioning": True,
        },
        "dependency_tracking": {"enable_cycle_detection": True, "on_cycle": "warn_and_skip", "max_depth": 100},
        "cache_invalidation": {"on_file_change": "semantic_delta", "on_config_change": "full_rebuild"},
    }
}


class TestQueryCache(unittest.TestCase):
    def _cache(self, path: str) -> QueryCache:
        return QueryCache(Path(path), max_size_gb=0.01)

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("k1", 42, "fp1")
            self.assertEqual(cache.get("k1"), 42)

    def test_get_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            self.assertIsNone(cache.get("nonexistent"))

    def test_invalidate_removes_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("k1", "val", "fp1")
            evicted = cache.invalidate("k1")
            self.assertIn("k1", evicted)
            self.assertIsNone(cache.get("k1"))

    def test_invalidate_cascades_to_dependents(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("base", 1, "fp1")
            cache.put("child", 2, "fp2", dependencies=["base"])
            evicted = cache.invalidate("base")
            self.assertIn("base", evicted)
            self.assertIn("child", evicted)

    def test_invalidate_by_fingerprint_no_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("k1", "val", "fp1")
            changed = cache.invalidate_by_fingerprint("k1", "fp1")
            self.assertFalse(changed)
            self.assertEqual(cache.get("k1"), "val")

    def test_invalidate_by_fingerprint_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("k1", "val", "fp1")
            changed = cache.invalidate_by_fingerprint("k1", "fp2")
            self.assertTrue(changed)
            self.assertIsNone(cache.get("k1"))

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("a", 1, "fp")
            cache.put("b", 2, "fp")
            cache.clear()
            self.assertIsNone(cache.get("a"))
            self.assertIsNone(cache.get("b"))

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("k1", 1, "fp")
            cache.get("k1")
            cache.get("missing")
            s = cache.stats()
            self.assertEqual(s["hits"], 1)
            self.assertEqual(s["misses"], 1)

    def test_disk_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put("persist", {"hello": "world"}, "fp1")
            # Clear memory, read from disk
            cache._memory.clear()
            val = cache.get("persist")
            self.assertEqual(val, {"hello": "world"})


class TestMemoizationEngine(unittest.TestCase):
    def _engine(self, cache_path: str) -> MemoizationEngine:
        cfg = dict(_CONFIG)
        cfg["memoization"] = dict(cfg["memoization"])
        cfg["memoization"]["storage"] = dict(cfg["memoization"]["storage"])
        cfg["memoization"]["storage"]["path"] = cache_path
        return MemoizationEngine(cfg)

    def test_memoize_computes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            counter = {"n": 0}

            def compute():
                counter["n"] += 1
                return 99

            v1 = engine.memoize("test_key", compute)
            v2 = engine.memoize("test_key", compute)
            self.assertEqual(v1, 99)
            self.assertEqual(v2, 99)
            # Only computed once; second call is cached
            self.assertEqual(counter["n"], 1)

    def test_on_file_changed_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            src = Path(tmp) / "test.py"
            src.write_text("x = 1\n")
            engine.cache.put(str(src), "cached_value", "old_fp")
            evicted = engine.on_file_changed(str(src))
            self.assertIn(str(src), evicted)

    def test_on_config_changed_clears_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            engine.cache.put("k1", 1, "fp")
            engine.on_config_changed()
            self.assertIsNone(engine.cache.get("k1"))

    def test_check_dependency_cycles_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            cycles = engine.check_dependency_cycles()
            self.assertEqual(cycles, [])

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            s = engine.stats()
            self.assertIn("entries", s)
            self.assertIn("hit_rate", s)


if __name__ == "__main__":
    unittest.main()
