"""Pre-flight test auditor & automated bug patcher for Aero Future.

Runs the local test suite during the bootstrap sequence, classifies every
failure, and routes genuine core-logic bugs into the self-healing layer so the
source is patched automatically before the execution lifecycle completes.

Two failure categories are distinguished:

* **Category A -- Environment / Missing Library.**  Tracebacks containing
  ``ModuleNotFoundError`` / ``ImportError``.  These are *not* code bugs: the
  missing distribution is handed back to
  :class:`~core.environment_bootstrap.RuntimeEnvironmentBootstrapper` for
  silent provisioning.
* **Category B -- True Core Logic Bug.**  Everything else (semantic edge
  issues, cross-platform pathing anomalies, regression oversights).  The fault
  is wrapped into a context package and dispatched to the topological
  self-healer / ``error_interceptor`` to patch the offending source in place.

Each audit round runs in a fresh subprocess so patched modules are reloaded
cleanly, and the validation block is re-run until it converges (bounded by
``max_rounds``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cross-platform path-literal normalization (a concrete Category-B patcher)
# ---------------------------------------------------------------------------
_STRING_LITERAL = re.compile(r"""(['"])(?P<body>(?:\\.|(?!\1).)*)\1""")
# An escaped backslash pair (``\\`` in source) is how a path separator is
# written inside a normal Python string literal; a *single* backslash there is
# a control escape (``\n``, ``\t``, ...) and must never be touched.
_WINDOWS_SEP = re.compile(r"\\\\")


def normalize_path_literals(source: str) -> Tuple[str, bool]:
    """Rewrite hardcoded Windows path separators in string literals to ``/``.

    Forward slashes are accepted by ``open``/``pathlib`` on every platform
    (including Windows), so converting ``"a\\\\b\\\\c.txt"`` to ``"a/b/c.txt"``
    is a genuine, portable fix for the most common cross-OS pathing anomaly.
    Only *escaped* backslash pairs inside string literals are rewritten; control
    escapes such as ``\\n`` / ``\\t`` (a single backslash) are preserved.

    Returns ``(new_source, changed)``.
    """
    changed = False

    def _fix_literal(match: re.Match) -> str:
        nonlocal changed
        quote = match.group(1)
        body = match.group("body")
        new_body = _WINDOWS_SEP.sub("/", body)
        if new_body != body:
            changed = True
        return f"{quote}{new_body}{quote}"

    new_source = _STRING_LITERAL.sub(_fix_literal, source)
    return new_source, changed


class PreFlightTestAuditor:
    """Runs test integrity sweeps and self-heals Category-B logic bugs."""

    _ENV_MARKERS = ("ModuleNotFoundError", "ImportError", "No module named")
    _FILE_RE = re.compile(r'File "([^"]+)", line (\d+)')
    _MISSING_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")

    def __init__(self, test_dir: str = "tests", *, top_level: Optional[str] = None,
                 max_rounds: int = 3):
        self.test_dir = test_dir
        self.top_level = top_level or os.path.dirname(os.path.abspath(test_dir)) or "."
        self.max_rounds = max_rounds
        self.patched_files: List[str] = []

    # -- public entry ------------------------------------------------------
    def run_suite_and_heal(self) -> bool:
        """Run the suite, heal logic bugs, and re-validate until it passes."""
        print("[*] Commencing pre-flight test integrity audits...")
        if not os.path.isdir(self.test_dir):
            print("[+] No localized testing folder found. Skipping diagnostic sweep safely.")
            return True

        for round_index in range(self.max_rounds):
            ok, output = self._run_once()
            if ok:
                if round_index == 0:
                    print("[+] All localized test integrity assertions passed flawlessly.")
                else:
                    print("[+] Audit converged: suite passes after self-healing patches.")
                return True

            failures = self._parse_failures(output)
            print(
                f"[-] Diagnostics detected {len(failures)} structural test "
                f"disruption(s) on round {round_index + 1}."
            )

            applied_any = False
            for category, fault_file, detail in failures:
                if category == "environment":
                    print("[->] Intercepted missing-environment anomaly. Routing to bootstrapper...")
                    if self._route_env_error(detail):
                        applied_any = True
                    continue
                print("[CRITICAL] Category B Core Logic Bug isolated in execution flow.")
                print("[*] Dispatching traceback context payload to the Self-Healing Engine...")
                if self.trigger_self_heal_patch(fault_file, detail):
                    applied_any = True

            if not applied_any:
                print("[-] Automated self-healing patch sequence exhausted without convergence.")
                return False
            print("[*] Re-running audit validations over the patched manifold...")

        # Final verification after the last patch round.
        ok, _ = self._run_once()
        return ok

    # -- suite execution ---------------------------------------------------
    def _run_once(self) -> Tuple[bool, str]:
        """Run the suite in a fresh subprocess; return (passed, combined_output).

        Discovery runs with ``cwd`` at the top-level directory (and that
        directory on ``PYTHONPATH``) so sibling modules under test import
        cleanly, without requiring the test folder to be a package.
        """
        rel_tests = os.path.relpath(self.test_dir, self.top_level)
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", rel_tests]
        env = dict(os.environ)
        # Prevent nested auditor/bootstrap recursion inside the child process.
        env["AERO_DISABLE_BOOTSTRAP"] = "1"
        env["AERO_AUDIT_ACTIVE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.abspath(self.top_level), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                cwd=self.top_level, timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            return False, f"audit run timed out: {exc}"
        return proc.returncode == 0, (proc.stdout or "") + "\n" + (proc.stderr or "")

    # -- classification ----------------------------------------------------
    def _parse_failures(self, output: str) -> List[Tuple[str, Optional[str], str]]:
        """Split unittest output into per-failure (category, fault_file, detail)."""
        blocks = re.split(r"={60,}\n", output)
        failures: List[Tuple[str, Optional[str], str]] = []
        for block in blocks:
            if not (block.startswith("ERROR:") or block.startswith("FAIL:")):
                continue
            category = self.classify(block)
            fault_file = None if category == "environment" else self._extract_fault_file(block)
            failures.append((category, fault_file, block))
        return failures

    def classify(self, traceback_text: str) -> str:
        """Return ``"environment"`` or ``"logic"`` for a failure block."""
        if any(marker in traceback_text for marker in self._ENV_MARKERS):
            return "environment"
        return "logic"

    def _extract_fault_file(self, traceback_text: str) -> Optional[str]:
        """Pick the deepest project source file named in the traceback."""
        candidates = self._FILE_RE.findall(traceback_text)
        top = os.path.abspath(self.top_level)
        chosen: Optional[str] = None
        for path, _line in candidates:
            abs_path = os.path.abspath(path)
            try:
                within = os.path.commonpath([abs_path, top]) == top
            except ValueError:
                within = False  # different drive/root
            if not within:
                continue
            if "site-packages" in abs_path or f"{os.sep}unittest{os.sep}" in abs_path:
                continue
            chosen = abs_path  # keep the last (deepest) qualifying frame
        return chosen

    # -- healing -----------------------------------------------------------
    def _route_env_error(self, traceback_text: str) -> bool:
        """Detect a missing-module failure and report it as a contract violation.

        Under the Environment Contract model the engine never installs packages
        automatically.  The failure is logged; the operator is expected to
        satisfy the dependency manually and re-run the audit.
        """
        match = self._MISSING_MODULE_RE.search(traceback_text)
        if not match:
            return False
        module = match.group(1).split(".")[0]
        print(
            f"[->] Contract Violation: missing dependency '{module}'. "
            "Install it manually and re-run the audit."
        )
        return False

    def trigger_self_heal_patch(self, fault_file: Optional[str], traceback: str) -> bool:
        """Patch the offending source for a Category-B bug; return success.

        Strategy, in order of preference:

        1. **Cross-platform path normalization** -- rewrite hardcoded Windows
           separators in string literals (the dominant cross-OS pathing bug).
        2. **Syntactic self-healing** -- for ``SyntaxError`` faults, route the
           file through the toolchain self-healer.
        3. **Topological reification** -- record the fault as a broken HIN edge
           and re-wire it geometrically for the ledger trail.

        A patch is only reported as successful when the source actually changed.
        """
        if not fault_file or not os.path.isfile(fault_file):
            return False

        with open(fault_file, "r", encoding="utf-8") as handle:
            original = handle.read()

        # (1) Path-discrepancy normalization.
        patched, changed = normalize_path_literals(original)
        if changed:
            self._atomic_write(fault_file, patched)
            self.patched_files.append(fault_file)
            print(f"[+] Patched cross-platform path literals in {fault_file}")
            self._record_topological_trace(fault_file, traceback)
            return True

        # (2) Syntactic recovery for parser-level faults.
        if "SyntaxError" in traceback:
            if self._heal_syntax(fault_file):
                self.patched_files.append(fault_file)
                print(f"[+] Syntactically self-healed {fault_file}")
                return True

        # (3) Geometric reification (telemetry trail; no source delta).
        self._record_topological_trace(fault_file, traceback)
        return False

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        tmp = f"{path}.aero-heal.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)

    @staticmethod
    def _heal_syntax(fault_file: str) -> bool:
        try:
            from pathlib import Path

            from core.toolchain.self_healing import heal_module

            def _build_fn(path: Path):
                with open(path, "r", encoding="utf-8") as handle:
                    compile(handle.read(), str(path), "exec")
                return []

            report = heal_module(Path(fault_file), _build_fn, language="python")
            return bool(getattr(report, "success", False) and getattr(report, "applied", None))
        except Exception:  # noqa: BLE001 - healing is best-effort
            return False

    @staticmethod
    def _record_topological_trace(fault_file: str, traceback: str) -> None:
        """Reify the fault as a broken HIN edge and re-wire it (ledger trail)."""
        try:
            from core.hin_vm import HINNetwork
            from error_interceptor import reify_parse_failure_as_port
            from orchestrator import TopologicalSelfHealer

            net = HINNetwork()
            faulty = reify_parse_failure_as_port(net, f"{fault_file}: {traceback[:120]}")
            TopologicalSelfHealer().heal_unterminated_interface(net, faulty)
        except Exception:  # noqa: BLE001 - telemetry trail only
            pass


__all__ = ["PreFlightTestAuditor", "normalize_path_literals"]
