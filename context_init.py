"""Initialise the 4D block-universe ledger (``context.aero``).

The ledger is an append-only history of every mutation the evolution engine
has ever attempted.  This module lays down its *genesis block* -- a tamper
-evident anchor that ties the history to a specific source commit and target
file -- so :class:`evolve.CryptographicLedger` always has a well-formed chain
to read and extend.
"""

import os
import json
import time
import hashlib
import sys

LEDGER_VERSION = "1.0"


def _genesis_hash(commit_sha: str, target_file: str, timestamp: str) -> str:
    payload = f"{commit_sha}|{target_file}|{timestamp}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_workspace_context(workspace: str, commit_sha: str, target_file: str) -> None:
    """Create ``<workspace>/context.aero`` with a genesis block if absent.

    Idempotent: an existing, well-formed ledger is left untouched so the
    append-only invariant of the block universe is never violated.  A missing
    or corrupt file is (re)seeded with a fresh genesis block.
    """
    ledger_path = os.path.join(workspace, "context.aero")

    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "mutation_history" in data:
                return  # Already initialised -- preserve the chain.
        except (OSError, json.JSONDecodeError):
            # Corrupt ledger: fall through and re-seed.
            pass

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    genesis = {
        "version": LEDGER_VERSION,
        "genesis": {
            "commit_sha": commit_sha,
            "target_file": target_file,
            "timestamp": timestamp,
            "genesis_hash": _genesis_hash(commit_sha, target_file, timestamp),
        },
        "mutation_history": [],
    }

    os.makedirs(workspace, exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as handle:
        json.dump(genesis, handle, indent=2)


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    sha = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    target = sys.argv[3] if len(sys.argv) > 3 else "main.py"
    initialize_workspace_context(ws, sha, target)
    print(f"Initialised block-universe ledger at {os.path.join(ws, 'context.aero')}")
