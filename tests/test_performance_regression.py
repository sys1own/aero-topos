"""Performance / scaling regression suite (feature #8).

Simulates a large codebase (hundreds of files) and measures:

* build time with vs. without distributed mode,
* numerical accuracy with vs. without strict floating-point, and
* cache hit-rate after a small change.

The thresholds are deliberately loose so the suite is robust on shared CI
hardware while still catching gross regressions.
"""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path

from src.build.distributed import BuildTask, DistributedCoordinator
from src.memoization.cache_engine import MemoizationEngine
from src.precision_shield.shield import PrecisionShield

_FILE_COUNT = 250


def _make_large_codebase(root: Path, count: int = _FILE_COUNT) -> list:
    files = []
    for i in range(count):
        path = root / f"module_{i:04d}.py"
        path.write_text(
            f"# module {i}\n"
            f"VALUE_{i} = {i}\n"
            f"def compute_{i}(x):\n"
            f"    return x * VALUE_{i} + {i}\n"
        )
        files.append(path)
    return files


class TestDistributedBuildTime(unittest.TestCase):
    def test_distributed_faster_than_single(self):
        # Each "compilation" sleeps briefly; parallel workers should win.
        work = lambda: time.sleep(0.01) or True
        n = 60

        single = DistributedCoordinator({"distributed": {"enabled": False}})
        tasks = [BuildTask(task_id=f"t{i}", func=work) for i in range(n)]
        t0 = time.monotonic()
        single_results = single.dispatch(tasks)
        single_time = time.monotonic() - t0

        distributed = DistributedCoordinator(
            {"distributed": {"enabled": True, "worker_nodes": ["local"] * 10}}
        )
        tasks2 = [BuildTask(task_id=f"t{i}", func=work) for i in range(n)]
        t0 = time.monotonic()
        dist_results = distributed.dispatch(tasks2)
        distributed_time = time.monotonic() - t0

        self.assertTrue(all(r.success for r in single_results))
        self.assertTrue(all(r.success for r in dist_results))
        self.assertLess(distributed_time, single_time)


class TestStrictFloatingPointAccuracy(unittest.TestCase):
    def test_strict_flags_differ_from_relaxed(self):
        strict = PrecisionShield(
            {"precision_shield": {"ieee_compliance": "strict", "fast_math_override": False, "shield_zones": []}}
        )
        relaxed = PrecisionShield(
            {"precision_shield": {"ieee_compliance": "relaxed", "fast_math_override": True,
                                  "floating_point_contract": "allow", "shield_zones": []}}
        )
        self.assertIn("-fno-fast-math", strict.compiler_flags("gcc"))
        self.assertIn("-ffast-math", relaxed.compiler_flags("gcc"))

    def test_reassociation_changes_result(self):
        # Fast-math is free to reassociate; in finite precision that changes the
        # answer. The shield disallows it precisely because the result is no
        # longer reproducible/guaranteed -- this asserts the divergence exists.
        terms = [1.0, 1e16, -1e16]
        reference = math.fsum(terms)  # exact high-precision value == 1.0

        strict_order = 0.0            # left-to-right, as written (IEEE strict)
        for t in terms:
            strict_order += t

        reassociated = terms[0] + (terms[1] + terms[2])  # fast-math regrouping

        self.assertEqual(reference, 1.0)
        self.assertNotEqual(strict_order, reassociated)   # reordering changed it
        # At least one ordering diverges from the true value -> not safe to allow.
        self.assertTrue(
            abs(strict_order - reference) > 0 or abs(reassociated - reference) > 0
        )


class TestCacheHitRate(unittest.TestCase):
    def test_hit_rate_after_small_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _make_large_codebase(root, 50)
            engine = MemoizationEngine(
                {"memoization": {"storage": {"path": str(root / ".cache")},
                                 "strategies": {"ignore_comments_and_whitespace": True}}}
            )

            compute_calls = {"n": 0}

            def memo_all():
                for f in files:
                    def compute(f=f):
                        compute_calls["n"] += 1
                        return f.stat().st_size  # non-None so it is cached
                    engine.memoize(key=str(f), compute_fn=compute, source_file=str(f))

            memo_all()                         # cold: 50 computes
            self.assertEqual(compute_calls["n"], 50)

            compute_calls["n"] = 0
            memo_all()                         # warm: 0 computes (all hits)
            self.assertEqual(compute_calls["n"], 0)

            # Small semantic change to a single file.
            files[3].write_text("VALUE_3 = 99999\n")
            compute_calls["n"] = 0
            memo_all()                         # only the changed file recomputes
            self.assertEqual(compute_calls["n"], 1)

            stats = engine.stats()
            self.assertGreater(stats["hit_rate"], 0.5)

    def test_comment_only_change_is_a_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "m.py"
            f.write_text("X = 1\n")
            engine = MemoizationEngine(
                {"memoization": {"storage": {"path": str(root / ".cache")},
                                 "strategies": {"ignore_comments_and_whitespace": True},
                                 "cache_invalidation": {"on_file_change": "semantic_delta"}}}
            )
            engine.on_file_changed(str(f))         # establish baseline fingerprint
            f.write_text("X = 1   # a new comment\n")
            evicted = engine.on_file_changed(str(f))  # comment-only -> no eviction
            self.assertEqual(evicted, [])


class TestLargeCodebaseScales(unittest.TestCase):
    def test_uast_over_hundreds_of_files(self):
        from src.analysis.semantic_mapper import SemanticMapper

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "builder_brains").mkdir()
            _make_large_codebase(root / "builder_brains", 300)
            mapper = SemanticMapper({"analysis": {"semantic_proximity_mapping": {}}})
            t0 = time.monotonic()
            mapper.build_uast(root)
            elapsed = time.monotonic() - t0
            stats = mapper.get_statistics()
            self.assertGreater(stats["python_nodes"], 0)
            self.assertGreater(stats["unified_node_counts"].get("uast_function", 0), 100)
            self.assertLess(elapsed, 30.0)  # gross-regression guard


if __name__ == "__main__":
    unittest.main()
