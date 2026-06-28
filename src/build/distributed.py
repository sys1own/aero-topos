"""
Distributed Build Support.

Coordinates compilation tasks across a pool of workers.  Three backends are
supported:

* ``local``  -- run tasks in local subprocesses / callables (always available).
* ``ssh``    -- dispatch to remote hosts via ``fabric`` (lazy-imported).
* ``k8s``    -- dispatch to Kubernetes pods via the ``kubernetes`` client.

The coordinator distributes work, collects results, and tolerates worker
failures: a task that fails on one worker is retried on the next healthy worker,
falling back to local execution so a build never stalls because a remote node
went away.  When distributed mode is disabled (or no workers are configured) the
coordinator transparently runs everything locally -- the single-machine path is
always intact (graceful fallback).
"""

from __future__ import annotations

import itertools
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.utils.serialization import dataclass_to_dict

logger = logging.getLogger("build.distributed")

@dataclass
class BuildTask:
    """A single unit of compilation work.

    A task either carries a shell ``command`` (executed on the worker) or a
    Python ``func`` callable (executed in-process by the local backend).  The
    callable form keeps the engine testable without a real toolchain.
    """

    task_id: str
    command: Optional[List[str]] = None
    func: Optional[Callable[[], Any]] = None
    cwd: Optional[str] = None
    artifact: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    task_id: str
    worker: str
    success: bool
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    attempts: int = 1
    value: Any = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(
            self,
            exclude=["stdout", "stderr", "value"],
            round_keys={"duration": 6},
        )


# ----------------------------------------------------------------------
# Worker backends
# ----------------------------------------------------------------------


class WorkerBackend:
    """Base class for a compilation worker."""

    name: str = "worker"

    def is_available(self) -> bool:
        return True

    def run_task(self, task: BuildTask) -> BuildResult:  # pragma: no cover - abstract
        raise NotImplementedError


class LocalBackend(WorkerBackend):
    """Runs tasks on the local machine (subprocess or in-process callable)."""

    def __init__(self, name: str = "local") -> None:
        self.name = name

    def run_task(self, task: BuildTask) -> BuildResult:
        start = time.monotonic()
        if task.func is not None:
            try:
                value = task.func()
                return BuildResult(
                    task_id=task.task_id,
                    worker=self.name,
                    success=True,
                    value=value,
                    duration=time.monotonic() - start,
                )
            except Exception as exc:  # noqa: BLE001 - surface as a failed task
                return BuildResult(
                    task_id=task.task_id,
                    worker=self.name,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration=time.monotonic() - start,
                )
        if not task.command:
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error="task has neither command nor func",
                duration=time.monotonic() - start,
            )
        try:
            proc = subprocess.run(
                task.command,
                cwd=task.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                # Inherit the full parent environment so build limits
                # (CARGO_BUILD_JOBS, codegen-units) and compiler paths survive.
                env=os.environ.copy(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration=time.monotonic() - start,
            )
        return BuildResult(
            task_id=task.task_id,
            worker=self.name,
            success=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout.decode("utf-8", "replace"),
            stderr=proc.stderr.decode("utf-8", "replace"),
            duration=time.monotonic() - start,
        )


class SSHBackend(WorkerBackend):
    """Runs shell tasks on a remote host via ``fabric`` (lazy-imported)."""

    def __init__(self, address: str) -> None:
        self.name = address
        self.address = address
        self._connection = None

    def is_available(self) -> bool:
        try:
            import fabric  # noqa: F401
        except ImportError:
            return False
        return True

    def _connect(self):
        if self._connection is None:
            from fabric import Connection  # lazy import

            self._connection = Connection(self.address)
        return self._connection

    def run_task(self, task: BuildTask) -> BuildResult:
        start = time.monotonic()
        if not task.command:
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error="ssh backend requires a shell command",
                duration=time.monotonic() - start,
            )
        try:
            conn = self._connect()
            import shlex
            command = " ".join(shlex.quote(part) for part in task.command)
            if task.cwd:
                command = f"cd {shlex.quote(task.cwd)} && {command}"
            result = conn.run(command, hide=True, warn=True)
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=result.exited == 0,
                returncode=result.exited,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001 - remote failures must not crash
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration=time.monotonic() - start,
            )


class KubernetesBackend(WorkerBackend):
    """Runs tasks as Kubernetes pods via the ``kubernetes`` client (lazy)."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.name = spec.get("name") or spec.get("image") or "k8s-pod"
        self.namespace = spec.get("namespace", "default")
        self.image = spec.get("image", "aero/builder:latest")

    def is_available(self) -> bool:
        try:
            import kubernetes  # noqa: F401
        except ImportError:
            return False
        return True

    def run_task(self, task: BuildTask) -> BuildResult:
        start = time.monotonic()
        if not task.command:
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error="k8s backend requires a shell command",
                duration=time.monotonic() - start,
            )
        try:
            from kubernetes import client, config as kube_config

            try:
                kube_config.load_incluster_config()
            except Exception:
                logger.debug("In-cluster k8s config unavailable, falling back to kubeconfig")
                kube_config.load_kube_config()

            core = client.CoreV1Api()
            pod_name = f"aero-build-{task.task_id}".lower().replace("_", "-")[:63]
            pod_manifest = {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": pod_name, "namespace": self.namespace},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "builder",
                            "image": self.image,
                            "command": task.command,
                            "workingDir": task.cwd or "/workspace",
                        }
                    ],
                },
            }
            core.create_namespaced_pod(namespace=self.namespace, body=pod_manifest)
            # A production implementation would watch the pod to completion and
            # stream logs; we report dispatch success and let the watcher collect
            # artefacts via the shared cache.
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=True,
                stdout=f"dispatched pod {pod_name}",
                duration=time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001
            return BuildResult(
                task_id=task.task_id,
                worker=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration=time.monotonic() - start,
            )


# ----------------------------------------------------------------------
# Shared cache
# ----------------------------------------------------------------------


class SharedCache:
    """A build-artifact cache shared between workers.

    ``nfs`` uses a shared filesystem directory; ``redis``/``s3`` lazily import
    their clients and fall back to a local directory when the client or service
    is unavailable, so cache sharing never breaks a build.
    """

    def __init__(self, mode: str = "nfs", location: Optional[str] = None) -> None:
        self.mode = mode
        self.location = location or os.environ.get("AERO_SHARED_CACHE", ".aero/shared_cache")
        self._backend_ok = True
        self._client = None
        if mode in ("nfs",):
            Path(self.location).mkdir(parents=True, exist_ok=True)
        elif mode == "redis":
            self._client = self._init_redis()
        elif mode == "s3":
            self._client = self._init_s3()
        # Always keep a local spill directory for fallbacks.
        self._spill = Path(self.location if mode == "nfs" else ".aero/shared_cache_spill")
        self._spill.mkdir(parents=True, exist_ok=True)

    def _init_redis(self):
        try:
            import redis  # type: ignore

            url = os.environ.get("AERO_REDIS_URL", "redis://localhost:6379/0")
            client = redis.Redis.from_url(url)
            client.ping()
            return client
        except ImportError:
            logger.debug("redis package not available")
            self._backend_ok = False
            return None
        except Exception as exc:
            logger.warning("Redis connection failed: %s", exc)
            self._backend_ok = False
            return None

    def _init_s3(self):
        try:
            import boto3  # type: ignore

            return boto3.client("s3")
        except ImportError:
            logger.debug("boto3 package not available")
            self._backend_ok = False
            return None
        except Exception as exc:
            logger.warning("S3 client initialization failed: %s", exc)
            self._backend_ok = False
            return None

    def put(self, key: str, path: str) -> bool:
        src = Path(path)
        if not src.exists():
            return False
        if self.mode == "redis" and self._client is not None:
            try:
                self._client.set(f"aero:cache:{key}", src.read_bytes())
                return True
            except Exception as exc:
                logger.warning("Redis cache put failed for key %r: %s", key, exc)
        if self.mode == "s3" and self._client is not None:
            bucket = os.environ.get("AERO_S3_BUCKET", "aero-build-cache")
            try:
                self._client.upload_file(str(src), bucket, key)
                return True
            except Exception as exc:
                logger.warning("S3 cache put failed for key %r: %s", key, exc)
        # nfs and all fallbacks: copy into the shared/spill directory.
        dest = self._spill / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True

    def get(self, key: str, dest: str) -> bool:
        if self.mode == "redis" and self._client is not None:
            try:
                blob = self._client.get(f"aero:cache:{key}")
                if blob is not None:
                    Path(dest).write_bytes(blob)
                    return True
            except Exception as exc:
                logger.warning("Redis cache get failed for key %r: %s", key, exc)
        if self.mode == "s3" and self._client is not None:
            bucket = os.environ.get("AERO_S3_BUCKET", "aero-build-cache")
            try:
                self._client.download_file(bucket, key, dest)
                return True
            except Exception as exc:
                logger.warning("S3 cache get failed for key %r: %s", key, exc)
        src = self._spill / key
        if src.exists():
            shutil.copy2(src, dest)
            return True
        return False


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------


class DistributedCoordinator:
    """Distributes build tasks across workers and collects their results."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_attempts: int = 3) -> None:
        self.config = config or {}
        dist_cfg = self.config.get("distributed", {}) or {}
        self.enabled = bool(dist_cfg.get("enabled", False))
        self.worker_specs = list(dist_cfg.get("worker_nodes", []))
        self.cache_sharing = dist_cfg.get("cache_sharing", "nfs")
        self.max_attempts = max_attempts

        self.backends = self._build_backends()
        self._healthy = {b.name: True for b in self.backends}
        self._lock = threading.Lock()
        self._rr = itertools.count()
        self.shared_cache = SharedCache(self.cache_sharing)

    # ------------------------------------------------------------------

    def _build_backends(self) -> List[WorkerBackend]:
        backends: List[WorkerBackend] = []
        if self.enabled:
            for spec in self.worker_specs:
                backend = self._backend_for_spec(spec)
                if backend is not None and backend.is_available():
                    backends.append(backend)
        # Local backend is always present as the final fallback.
        backends.append(LocalBackend())
        return backends

    @staticmethod
    def _backend_for_spec(spec: Any) -> Optional[WorkerBackend]:
        if isinstance(spec, dict):
            return KubernetesBackend(spec)
        text = str(spec)
        if text.startswith("k8s://"):
            return KubernetesBackend({"name": text[len("k8s://"):]})
        if text.startswith("ssh://"):
            return SSHBackend(text[len("ssh://"):])
        if text in ("local", "localhost"):
            return LocalBackend(text)
        return SSHBackend(text)

    @property
    def remote_worker_count(self) -> int:
        return sum(1 for b in self.backends if not isinstance(b, LocalBackend))

    def worker_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_workers": len(self.backends),
                "remote_workers": self.remote_worker_count,
                "healthy": dict(self._healthy),
            }

    # ------------------------------------------------------------------

    def _next_worker(self, exclude: Optional[set] = None) -> WorkerBackend:
        exclude = exclude or set()
        with self._lock:
            healthy = [b for b in self.backends if self._healthy.get(b.name, True) and b.name not in exclude]
        if not healthy:
            # Everything excluded/unhealthy -> last resort is a fresh local node.
            return LocalBackend("local-fallback")
        idx = next(self._rr) % len(healthy)
        return healthy[idx]

    def _mark_unhealthy(self, worker: WorkerBackend) -> None:
        if isinstance(worker, LocalBackend):
            return  # never disable the local fallback
        with self._lock:
            self._healthy[worker.name] = False

    def _run_one(self, task: BuildTask) -> BuildResult:
        tried: set = set()
        last: Optional[BuildResult] = None
        for attempt in range(1, self.max_attempts + 1):
            worker = self._next_worker(exclude=tried)
            result = worker.run_task(task)
            result.attempts = attempt
            if result.success:
                # On success, cache the produced artefact for other workers.
                if task.artifact and os.path.exists(task.artifact):
                    self.shared_cache.put(task.task_id, task.artifact)
                return result
            self._mark_unhealthy(worker)
            tried.add(worker.name)
            last = result
        if last is not None:
            return last
        return BuildResult(task_id=task.task_id, worker="none", success=False, error="no workers")

    def dispatch(self, tasks: List[BuildTask]) -> List[BuildResult]:
        """Run all tasks, distributing across workers, and collect results."""
        if not tasks:
            return []
        # Concurrency is bounded by the number of workers we can keep busy.
        max_workers = max(1, len(self.backends))
        results: List[BuildResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._run_one, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        # Preserve input order for deterministic reporting.
        order = {task.task_id: i for i, task in enumerate(tasks)}
        results.sort(key=lambda r: order.get(r.task_id, 0))
        return results
