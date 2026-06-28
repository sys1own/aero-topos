"""Tests for distributed build support (feature #4)."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.build.distributed import (
    BuildResult,
    BuildTask,
    DistributedCoordinator,
    LocalBackend,
    SharedCache,
    WorkerBackend,
)
from src.evolution.sandbox_manager import SandboxManager


class _FlakyBackend(WorkerBackend):
    """A backend that always fails -- used to exercise retry/fallback."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def run_task(self, task: BuildTask) -> BuildResult:
        self.calls += 1
        return BuildResult(task_id=task.task_id, worker=self.name, success=False, error="boom")


class TestLocalDispatch(unittest.TestCase):
    def test_disabled_runs_locally(self):
        coord = DistributedCoordinator({"distributed": {"enabled": False}})
        tasks = [BuildTask(task_id=f"t{i}", func=(lambda i=i: i * i)) for i in range(5)]
        results = coord.dispatch(tasks)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual([r.value for r in results], [0, 1, 4, 9, 16])
        self.assertEqual(coord.remote_worker_count, 0)

    def test_results_preserve_input_order(self):
        coord = DistributedCoordinator({"distributed": {"enabled": False}})
        tasks = [BuildTask(task_id=f"task-{i}", func=(lambda i=i: i)) for i in range(20)]
        results = coord.dispatch(tasks)
        self.assertEqual([r.task_id for r in results], [t.task_id for t in tasks])

    def test_command_task_failure_is_reported(self):
        coord = DistributedCoordinator({"distributed": {"enabled": False}})
        task = BuildTask(task_id="bad", command=["definitely_not_a_real_binary_xyz"])
        result = coord.dispatch([task])[0]
        self.assertFalse(result.success)


class TestFailureHandling(unittest.TestCase):
    def test_retry_falls_back_to_healthy_worker(self):
        coord = DistributedCoordinator({"distributed": {"enabled": False}}, max_attempts=3)
        flaky = _FlakyBackend("remote-1")
        # Insert a failing remote worker ahead of the local fallback.
        coord.backends.insert(0, flaky)
        coord._healthy[flaky.name] = True

        result = coord.dispatch([BuildTask(task_id="t", func=lambda: "ok")])[0]
        self.assertTrue(result.success)          # local fallback succeeds
        self.assertEqual(result.value, "ok")
        self.assertGreaterEqual(result.attempts, 2)  # tried flaky, then local
        self.assertFalse(coord.worker_stats()["healthy"][flaky.name])  # marked down

    def test_all_workers_failing_reports_failure(self):
        coord = DistributedCoordinator({"distributed": {"enabled": False}}, max_attempts=2)
        coord.backends = [_FlakyBackend("a"), _FlakyBackend("b")]
        coord._healthy = {"a": True, "b": True}
        result = coord.dispatch([BuildTask(task_id="t", func=lambda: 1)])[0]
        self.assertFalse(result.success)


class TestParallelSpeedup(unittest.TestCase):
    def test_multiple_workers_are_faster(self):
        # 8 local workers should beat a single serial worker on sleepy tasks.
        sleepy = lambda: time.sleep(0.02) or 1
        tasks = [BuildTask(task_id=f"t{i}", func=sleepy) for i in range(24)]

        single = DistributedCoordinator({"distributed": {"enabled": False}})
        t0 = time.monotonic()
        single.dispatch(tasks)
        serial_time = time.monotonic() - t0

        multi = DistributedCoordinator(
            {"distributed": {"enabled": True, "worker_nodes": ["local"] * 8}}
        )
        tasks2 = [BuildTask(task_id=f"t{i}", func=sleepy) for i in range(24)]
        t0 = time.monotonic()
        multi.dispatch(tasks2)
        parallel_time = time.monotonic() - t0

        self.assertLess(parallel_time, serial_time)


class TestSharedCache(unittest.TestCase):
    def test_nfs_put_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedCache("nfs", location=str(Path(tmp) / "shared"))
            artifact = Path(tmp) / "out.o"
            artifact.write_bytes(b"OBJECT")
            self.assertTrue(cache.put("k1", str(artifact)))
            dest = Path(tmp) / "fetched.o"
            self.assertTrue(cache.get("k1", str(dest)))
            self.assertEqual(dest.read_bytes(), b"OBJECT")

    def test_redis_falls_back_gracefully(self):
        # No redis server -> client init fails -> spill directory still works.
        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedCache("redis")
            artifact = Path(tmp) / "a.o"
            artifact.write_bytes(b"X")
            self.assertTrue(cache.put("k", str(artifact)))


class TestSandboxIntegration(unittest.TestCase):
    def test_sandbox_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SandboxManager(Path(tmp), {"distributed": {"enabled": False}})
            results = mgr.dispatch_build_tasks([BuildTask(task_id="k", func=lambda: 42)])
            self.assertEqual(results[0].value, 42)
            self.assertIn("total_workers", mgr.worker_stats())

    def test_existing_sandbox_api_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SandboxManager(Path(tmp))
            path = mgr.create_sandbox("s1")
            self.assertTrue(path.exists())
            mgr.cleanup_all()
            self.assertIsNone(mgr.get_sandbox_path("s1"))


if __name__ == "__main__":
    unittest.main()
