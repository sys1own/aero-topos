"""Shared subprocess execution helpers."""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional, Tuple


def run_command(
    cmd: List[str], workdir: Optional[str] = None
) -> Tuple[int, str, str]:
    """Run *cmd*, returning ``(returncode, stdout, stderr)``.

    Never raises on execution failure — the error is captured in *stderr*.

    The child always inherits the parent shell's full environment so build
    limits (``CARGO_BUILD_JOBS``, ``codegen-units``) and custom compiler paths
    are never dropped during live verification loops.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )
