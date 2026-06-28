"""
HPC Scheduler Integration.

Generates and submits batch jobs to SLURM (``sbatch``) or PBS (``qsub``),
monitors their state, and captures logs.  Compilation itself can be parallelised
by submitting a dedicated *build* job to the cluster, and the self-evolution
engine can submit many candidate builds concurrently.

Robustness is a first-class concern: missing schedulers, submission failures,
and timeouts are surfaced as a :class:`HPCJob` with ``state == "FAILED"`` /
``"TIMEOUT"`` and a populated ``error`` field rather than raising.

The class accepts an injectable ``runner`` (``cmd -> (rc, stdout, stderr)``) so
tests can mock the scheduler CLIs without a real cluster.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.serialization import dataclass_to_dict
from src.utils.subprocess_utils import run_command

Runner = Callable[[List[str]], Tuple[int, str, str]]


@dataclass
class HPCJob:
    job_id: str
    scheduler: str
    name: str
    script_path: str = ""
    state: str = "UNKNOWN"
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def submitted(self) -> bool:
        return bool(self.job_id) and self.state not in ("FAILED",)

    @property
    def succeeded(self) -> bool:
        return self.state == "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(
            self, exclude=["script_path", "stdout", "stderr"]
        )


# Scheduler-specific command vocabulary.
_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}


class HPCScheduler:
    """Submit and monitor jobs on a SLURM or PBS cluster."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, runner: Optional[Runner] = None) -> None:
        self.config = config or {}
        hpc = self.config.get("hpc", {}) or {}
        self.scheduler = str(hpc.get("scheduler", "none")).lower()
        self.queue = hpc.get("queue", "cpu")
        self.nodes = int(hpc.get("nodes", 1))
        self.tasks_per_node = int(hpc.get("tasks_per_node", 1))
        self.walltime = hpc.get("walltime", "01:00:00")
        self.environment = hpc.get("environment", {}) or {}
        self.build_on_login_node = bool(hpc.get("build_on_login_node", True))
        self.post_build_run = bool(hpc.get("post_build_run", False))
        self._runner = runner or self._default_runner

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.scheduler in ("slurm", "pbs")

    @property
    def submit_binary(self) -> Optional[str]:
        return {"slurm": "sbatch", "pbs": "qsub"}.get(self.scheduler)

    def available(self) -> bool:
        """Whether the scheduler CLI is actually installed."""
        if not self.enabled:
            return False
        binary = self.submit_binary
        return bool(binary) and shutil.which(binary) is not None

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_script(
        self,
        commands: List[str],
        job_name: str = "aero_job",
        workdir: Optional[str] = None,
        stdout_path: Optional[str] = None,
        stderr_path: Optional[str] = None,
    ) -> str:
        """Render a batch script for the configured scheduler."""
        out = stdout_path or f"{job_name}.out"
        err = stderr_path or f"{job_name}.err"
        lines = ["#!/bin/bash"]

        if self.scheduler == "slurm":
            lines += [
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --partition={self.queue}",
                f"#SBATCH --nodes={self.nodes}",
                f"#SBATCH --ntasks-per-node={self.tasks_per_node}",
                f"#SBATCH --time={self.walltime}",
                f"#SBATCH --output={out}",
                f"#SBATCH --error={err}",
            ]
        elif self.scheduler == "pbs":
            lines += [
                f"#PBS -N {job_name}",
                f"#PBS -q {self.queue}",
                f"#PBS -l nodes={self.nodes}:ppn={self.tasks_per_node}",
                f"#PBS -l walltime={self.walltime}",
                f"#PBS -o {out}",
                f"#PBS -e {err}",
            ]

        lines.append("")
        if workdir:
            lines.append(f"cd {shlex.quote(workdir)}")
        for module in self.environment.get("module_load", []) or []:
            lines.append(f"module load {shlex.quote(module)}")
        for key, value in (self.environment.get("env_vars", {}) or {}).items():
            lines.append(f"export {shlex.quote(key)}={shlex.quote(str(value))}")
        lines.append("")
        lines.extend(commands)
        return "\n".join(lines) + "\n"

    def write_script(self, commands: List[str], path: str, **kwargs: Any) -> str:
        script = self.generate_script(commands, **kwargs)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(script, encoding="utf-8")
        os.chmod(path, 0o755)
        return path

    # ------------------------------------------------------------------
    # Submission / monitoring
    # ------------------------------------------------------------------

    def submit(
        self,
        commands: List[str],
        job_name: str = "aero_job",
        workdir: Optional[str] = None,
        script_dir: Optional[str] = None,
    ) -> HPCJob:
        """Submit a batch job and return its :class:`HPCJob` handle."""
        job = HPCJob(job_id="", scheduler=self.scheduler, name=job_name)
        if not self.enabled:
            job.state = "FAILED"
            job.error = "no HPC scheduler configured"
            return job
        if not self.available():
            job.state = "FAILED"
            job.error = f"{self.submit_binary} not found on PATH"
            return job

        script_dir = script_dir or workdir or os.getcwd()
        script_path = os.path.join(script_dir, f"{job_name}.sh")
        try:
            self.write_script(commands, script_path, job_name=job_name, workdir=workdir)
        except OSError as exc:
            job.state = "FAILED"
            job.error = f"failed to write job script: {exc}"
            return job
        job.script_path = script_path

        rc, stdout, stderr = self._runner([self.submit_binary, script_path])
        job.returncode = rc
        job.stdout, job.stderr = stdout, stderr
        if rc != 0:
            job.state = "FAILED"
            job.error = stderr.strip() or f"{self.submit_binary} exited {rc}"
            return job

        job.job_id = self._parse_job_id(stdout)
        if not job.job_id:
            job.state = "FAILED"
            job.error = f"could not parse job id from: {stdout!r}"
            return job
        job.state = "PENDING"
        return job

    def status(self, job: HPCJob) -> str:
        """Query the current state of a submitted job."""
        if not job.job_id:
            return job.state
        if self.scheduler == "slurm":
            rc, out, _ = self._runner(["squeue", "-h", "-j", job.job_id, "-o", "%T"])
            if rc == 0 and out.strip():
                return self._normalize_state(out.strip().splitlines()[0])
            # Not in the queue any more -> consult sacct for the final state.
            rc, out, _ = self._runner(["sacct", "-n", "-X", "-j", job.job_id, "-o", "State"])
            if rc == 0 and out.strip():
                return self._normalize_state(out.strip().splitlines()[0])
            return "COMPLETED"
        if self.scheduler == "pbs":
            rc, out, _ = self._runner(["qstat", "-f", job.job_id])
            if rc != 0 or not out.strip():
                return "COMPLETED"
            match = re.search(r"job_state\s*=\s*(\w)", out)
            return self._normalize_state(match.group(1) if match else "C")
        return "UNKNOWN"

    def wait(
        self,
        job: HPCJob,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> HPCJob:
        """Poll until the job reaches a terminal state (or times out)."""
        if not job.submitted:
            return job
        start = time.monotonic()
        while True:
            state = self.status(job)
            job.state = state
            if state in _TERMINAL_STATES:
                return job
            if time.monotonic() - start > timeout:
                job.state = "TIMEOUT"
                job.error = f"job {job.job_id} exceeded {timeout}s"
                self.cancel(job)
                return job
            time.sleep(max(0.01, poll_interval))

    def cancel(self, job: HPCJob) -> None:
        if not job.job_id:
            return
        cancel_bin = {"slurm": "scancel", "pbs": "qdel"}.get(self.scheduler)
        if cancel_bin and shutil.which(cancel_bin):
            self._runner([cancel_bin, job.job_id])

    def submit_many(
        self,
        jobs: List[Tuple[str, List[str]]],
        workdir: Optional[str] = None,
        wait: bool = False,
        max_parallel: Optional[int] = None,
    ) -> List[HPCJob]:
        """Submit several (name, commands) candidate jobs concurrently."""
        if not jobs:
            return []
        max_parallel = max_parallel or min(len(jobs), self.nodes if self.nodes > 0 else len(jobs))
        submitted: List[HPCJob] = []
        with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
            futures = [
                pool.submit(self.submit, commands, name, workdir) for name, commands in jobs
            ]
            for future in futures:
                submitted.append(future.result())
        if wait:
            return [self.wait(job) if job.submitted else job for job in submitted]
        return submitted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_job_id(stdout: str) -> str:
        # SLURM: "Submitted batch job 123456"
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if match:
            return match.group(1)
        # PBS: "12345.headnode" or bare numeric id.
        match = re.search(r"(\d+(?:\.\w[\w.\-]*)?)", stdout.strip())
        return match.group(1) if match else ""

    @staticmethod
    def _normalize_state(raw: str) -> str:
        raw = raw.strip().upper()
        slurm_map = {
            "R": "RUNNING", "RUNNING": "RUNNING",
            "PD": "PENDING", "PENDING": "PENDING", "CF": "PENDING",
            "CG": "RUNNING", "COMPLETING": "RUNNING",
            "CD": "COMPLETED", "COMPLETED": "COMPLETED",
            "F": "FAILED", "FAILED": "FAILED",
            "CA": "CANCELLED", "CANCELLED": "CANCELLED",
            "TO": "TIMEOUT", "TIMEOUT": "TIMEOUT",
            "NF": "NODE_FAIL",
        }
        pbs_map = {
            "Q": "PENDING", "H": "PENDING", "W": "PENDING", "T": "PENDING",
            "R": "RUNNING", "E": "RUNNING", "B": "RUNNING",
            "C": "COMPLETED", "F": "COMPLETED",
        }
        if raw in slurm_map:
            return slurm_map[raw]
        if raw in pbs_map:
            return pbs_map[raw]
        return raw or "UNKNOWN"

    @staticmethod
    def _default_runner(cmd: List[str]) -> Tuple[int, str, str]:
        return run_command(cmd)
