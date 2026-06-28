# -*- coding: utf-8 -*-
"""Unit tests for src.hpc.scheduler."""

import os
import tempfile
import unittest

from src.hpc.scheduler import HPCJob, HPCScheduler, _TERMINAL_STATES


class TestHPCJob(unittest.TestCase):
    def test_submitted_property(self):
        job = HPCJob(job_id="123", scheduler="slurm", name="test", state="PENDING")
        self.assertTrue(job.submitted)

    def test_not_submitted_when_failed(self):
        job = HPCJob(job_id="123", scheduler="slurm", name="test", state="FAILED")
        self.assertFalse(job.submitted)

    def test_not_submitted_when_no_id(self):
        job = HPCJob(job_id="", scheduler="slurm", name="test", state="PENDING")
        self.assertFalse(job.submitted)

    def test_succeeded_property(self):
        job = HPCJob(job_id="1", scheduler="slurm", name="t", state="COMPLETED")
        self.assertTrue(job.succeeded)
        job.state = "RUNNING"
        self.assertFalse(job.succeeded)

    def test_to_dict(self):
        job = HPCJob(job_id="42", scheduler="pbs", name="myjob", state="RUNNING")
        d = job.to_dict()
        self.assertEqual(d["job_id"], "42")
        self.assertEqual(d["scheduler"], "pbs")
        self.assertEqual(d["name"], "myjob")
        self.assertEqual(d["state"], "RUNNING")


class TestHPCSchedulerConfig(unittest.TestCase):
    def test_defaults(self):
        sched = HPCScheduler()
        self.assertEqual(sched.scheduler, "none")
        self.assertFalse(sched.enabled)
        self.assertEqual(sched.queue, "cpu")
        self.assertEqual(sched.nodes, 1)
        self.assertEqual(sched.tasks_per_node, 1)
        self.assertEqual(sched.walltime, "01:00:00")

    def test_slurm_config(self):
        config = {"hpc": {"scheduler": "slurm", "queue": "gpu", "nodes": 4}}
        sched = HPCScheduler(config=config)
        self.assertTrue(sched.enabled)
        self.assertEqual(sched.scheduler, "slurm")
        self.assertEqual(sched.queue, "gpu")
        self.assertEqual(sched.nodes, 4)
        self.assertEqual(sched.submit_binary, "sbatch")

    def test_pbs_config(self):
        config = {"hpc": {"scheduler": "pbs"}}
        sched = HPCScheduler(config=config)
        self.assertTrue(sched.enabled)
        self.assertEqual(sched.submit_binary, "qsub")

    def test_disabled_scheduler(self):
        config = {"hpc": {"scheduler": "none"}}
        sched = HPCScheduler(config=config)
        self.assertFalse(sched.enabled)
        self.assertIsNone(sched.submit_binary)


class TestGenerateScript(unittest.TestCase):
    def test_slurm_script(self):
        config = {"hpc": {"scheduler": "slurm", "queue": "cpu", "nodes": 2,
                          "tasks_per_node": 4, "walltime": "02:00:00"}}
        sched = HPCScheduler(config=config)
        script = sched.generate_script(["echo hello", "python train.py"],
                                       job_name="test_job")
        self.assertIn("#!/bin/bash", script)
        self.assertIn("#SBATCH --job-name=test_job", script)
        self.assertIn("#SBATCH --partition=cpu", script)
        self.assertIn("#SBATCH --nodes=2", script)
        self.assertIn("#SBATCH --ntasks-per-node=4", script)
        self.assertIn("#SBATCH --time=02:00:00", script)
        self.assertIn("echo hello", script)
        self.assertIn("python train.py", script)

    def test_pbs_script(self):
        config = {"hpc": {"scheduler": "pbs", "queue": "batch", "nodes": 1,
                          "tasks_per_node": 8}}
        sched = HPCScheduler(config=config)
        script = sched.generate_script(["make build"], job_name="build_job")
        self.assertIn("#PBS -N build_job", script)
        self.assertIn("#PBS -q batch", script)
        self.assertIn("#PBS -l nodes=1:ppn=8", script)
        self.assertIn("make build", script)

    def test_workdir_included(self):
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config)
        script = sched.generate_script(["ls"], workdir="/home/user/project")
        self.assertIn("cd /home/user/project", script)

    def test_environment_modules(self):
        config = {"hpc": {"scheduler": "slurm",
                          "environment": {"module_load": ["cuda/11.8", "python/3.11"],
                                          "env_vars": {"OMP_NUM_THREADS": "4"}}}}
        sched = HPCScheduler(config=config)
        script = sched.generate_script(["./run.sh"])
        self.assertIn("module load cuda/11.8", script)
        self.assertIn("module load python/3.11", script)
        self.assertIn("export OMP_NUM_THREADS=4", script)


class TestWriteScript(unittest.TestCase):
    def test_writes_executable_file(self):
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "job.sh")
            sched.write_script(["echo test"], path, job_name="wj")
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.access(path, os.X_OK))
            with open(path) as f:
                content = f.read()
            self.assertIn("#!/bin/bash", content)


class TestSubmit(unittest.TestCase):
    def test_submit_disabled_scheduler(self):
        sched = HPCScheduler(config={"hpc": {"scheduler": "none"}})
        job = sched.submit(["echo hi"])
        self.assertEqual(job.state, "FAILED")
        self.assertIn("no HPC scheduler configured", job.error)

    def test_submit_binary_not_found(self):
        def runner(cmd):
            return (1, "", "not found")
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config, runner=runner)
        # Mock available() to return False
        job = sched.submit(["echo hi"])
        self.assertEqual(job.state, "FAILED")

    def test_submit_success_with_mock(self):
        def runner(cmd):
            return (0, "Submitted batch job 12345\n", "")
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config, runner=runner)
        # Need to mock available() - override the method
        sched.available = lambda: True
        with tempfile.TemporaryDirectory() as td:
            job = sched.submit(["echo hi"], job_name="test", script_dir=td)
            self.assertEqual(job.job_id, "12345")
            self.assertEqual(job.state, "PENDING")

    def test_submit_parse_pbs_job_id(self):
        def runner(cmd):
            return (0, "12345.headnode\n", "")
        config = {"hpc": {"scheduler": "pbs"}}
        sched = HPCScheduler(config=config, runner=runner)
        sched.available = lambda: True
        with tempfile.TemporaryDirectory() as td:
            job = sched.submit(["echo hi"], script_dir=td)
            self.assertEqual(job.job_id, "12345.headnode")


class TestStatus(unittest.TestCase):
    def test_status_no_job_id(self):
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config)
        job = HPCJob(job_id="", scheduler="slurm", name="t", state="UNKNOWN")
        self.assertEqual(sched.status(job), "UNKNOWN")

    def test_status_slurm_running(self):
        def runner(cmd):
            if "squeue" in cmd:
                return (0, "RUNNING\n", "")
            return (0, "", "")
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config, runner=runner)
        job = HPCJob(job_id="123", scheduler="slurm", name="t")
        self.assertEqual(sched.status(job), "RUNNING")

    def test_status_slurm_completed_via_sacct(self):
        def runner(cmd):
            if "squeue" in cmd:
                return (0, "", "")
            if "sacct" in cmd:
                return (0, "COMPLETED\n", "")
            return (0, "", "")
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config, runner=runner)
        job = HPCJob(job_id="123", scheduler="slurm", name="t")
        self.assertEqual(sched.status(job), "COMPLETED")

    def test_status_pbs(self):
        def runner(cmd):
            return (0, "job_state = R\n", "")
        config = {"hpc": {"scheduler": "pbs"}}
        sched = HPCScheduler(config=config, runner=runner)
        job = HPCJob(job_id="123.head", scheduler="pbs", name="t")
        self.assertEqual(sched.status(job), "RUNNING")


class TestNormalizeState(unittest.TestCase):
    def test_slurm_states(self):
        self.assertEqual(HPCScheduler._normalize_state("R"), "RUNNING")
        self.assertEqual(HPCScheduler._normalize_state("PD"), "PENDING")
        self.assertEqual(HPCScheduler._normalize_state("CD"), "COMPLETED")
        self.assertEqual(HPCScheduler._normalize_state("F"), "FAILED")
        self.assertEqual(HPCScheduler._normalize_state("CA"), "CANCELLED")
        self.assertEqual(HPCScheduler._normalize_state("TO"), "TIMEOUT")

    def test_pbs_states(self):
        self.assertEqual(HPCScheduler._normalize_state("Q"), "PENDING")
        self.assertEqual(HPCScheduler._normalize_state("C"), "COMPLETED")

    def test_unknown_state(self):
        self.assertEqual(HPCScheduler._normalize_state("MYSTERY"), "MYSTERY")

    def test_empty_state(self):
        self.assertEqual(HPCScheduler._normalize_state(""), "UNKNOWN")


class TestParseJobId(unittest.TestCase):
    def test_slurm_format(self):
        self.assertEqual(HPCScheduler._parse_job_id("Submitted batch job 99999"), "99999")

    def test_pbs_format(self):
        self.assertEqual(HPCScheduler._parse_job_id("12345.headnode"), "12345.headnode")

    def test_no_id_found(self):
        self.assertEqual(HPCScheduler._parse_job_id("no numbers here!"), "")


class TestWait(unittest.TestCase):
    def test_wait_not_submitted(self):
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config)
        job = HPCJob(job_id="", scheduler="slurm", name="t", state="FAILED")
        result = sched.wait(job)
        self.assertEqual(result.state, "FAILED")

    def test_wait_immediate_terminal(self):
        call_count = [0]

        def runner(cmd):
            call_count[0] += 1
            return (0, "COMPLETED\n", "")
        config = {"hpc": {"scheduler": "slurm"}}
        sched = HPCScheduler(config=config, runner=runner)
        job = HPCJob(job_id="1", scheduler="slurm", name="t", state="PENDING")
        result = sched.wait(job, poll_interval=0.01, timeout=1.0)
        self.assertEqual(result.state, "COMPLETED")


class TestSubmitMany(unittest.TestCase):
    def test_empty_list(self):
        sched = HPCScheduler(config={"hpc": {"scheduler": "slurm"}})
        result = sched.submit_many([])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
