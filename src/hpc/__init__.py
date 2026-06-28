"""HPC scheduler integration (SLURM / PBS).

Generates batch scripts from the blueprint, submits them, and monitors job
state.  When no scheduler is configured (or its CLI is unavailable) the module
reports ``enabled = False`` and callers fall back to local execution.
"""

from src.hpc.scheduler import HPCJob, HPCScheduler

__all__ = ["HPCJob", "HPCScheduler"]
