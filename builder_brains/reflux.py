"""
reflux.py — Aero Dependency Reflux Engine

Applies the structured mutation commands produced by
:class:`src.lsp_proxy.LspDiagnosticRefluxBinder` to source files in memory,
healing higher-level semantic defects (undefined symbols, missing imports,
unresolved Rust modules) before the compiler executes.

All patches operate on an in-memory copy of the file content; the returned
``bytes`` carry the mutated source so callers can either persist it or hand it
straight to the compilation pipeline without touching disk.

Supported actions:
  - ``RESOLVE_UNDEFINED_SYMBOL``    (Python) -> ``from aero_nova.core import {symbol}``
  - ``AUTO_REFLUX_IMPORT``          (Python) -> ``import {target}``
  - ``INJECT_RUST_USE_DECLARATION`` (Rust)   -> ``use crate::modules::{item};``
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AeroDependencyRefluxEngine:
    """Apply LSP-derived reflux patches to a source file's bytes."""

    AERO_CORE_NAMESPACE = "aero_nova.core"
    RUST_MODULE_NAMESPACE = "crate::modules"

    def apply_reflux_patches(
        self, file_path: str, actions: List[Dict[str, Any]]
    ) -> bytes:
        """Apply every reflux action to ``file_path`` and return mutated bytes.

        The file is read once, mutated in memory line-by-line, and returned as
        UTF-8 encoded bytes. Unknown actions and actions lacking a usable symbol
        are skipped (logged), leaving the source untouched for that entry.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("reflux: cannot read %s: %s", file_path, exc)
            return b""

        lines = text.splitlines(keepends=True)

        for action in actions:
            kind = action.get("action")
            symbol = action.get("symbol") or action.get("target")

            if not symbol:
                logger.debug("reflux: skipping action without symbol: %r", action)
                continue

            if kind == "RESOLVE_UNDEFINED_SYMBOL":
                lines = self._inject_python_core_import(lines, symbol)
            elif kind == "AUTO_REFLUX_IMPORT":
                lines = self._inject_python_import(lines, symbol)
            elif kind == "INJECT_RUST_USE_DECLARATION":
                lines = self._inject_rust_use(lines, symbol)
            else:
                logger.debug("reflux: unknown action kind %r", kind)

        # Apply standard import grouping to the final output (Python only).
        if file_path.endswith(".py"):
            lines = self._group_python_imports(lines)
        return "".join(lines).encode("utf-8")

    # ------------------------------------------------------------------ #
    # Python patches
    # ------------------------------------------------------------------ #
    def _inject_python_core_import(
        self, lines: List[str], symbol: str
    ) -> List[str]:
        """Inject ``from aero_nova.core import {symbol}`` after the docstring/comments."""
        statement = f"from {self.AERO_CORE_NAMESPACE} import {symbol}\n"
        if self._statement_present(lines, statement):
            return lines
        insert_at = self._python_header_offset(lines)
        return lines[:insert_at] + [statement] + lines[insert_at:]

    def _inject_python_import(self, lines: List[str], target: str) -> List[str]:
        """Insert ``import {target}`` at the head of the file (after the header)."""
        statement = f"import {target}\n"
        if self._statement_present(lines, statement):
            return lines
        insert_at = self._python_header_offset(lines)
        return lines[:insert_at] + [statement] + lines[insert_at:]

    @staticmethod
    def _python_header_offset(lines: List[str]) -> int:
        """Return the line index immediately following module docstrings/comments.

        Skips a leading shebang, encoding/comment lines, blank lines, a
        module-level triple-quoted docstring, and any existing ``from
        __future__ import ...`` statements, so injected imports always land
        *after* the compiler-flag header instead of being sorted into place
        after the fact. ``from __future__`` lines must occupy the absolute
        beginning of the file (only comments, blank lines, the module
        docstring, and other future statements may precede them) — inserting
        a regular import above one is an unrecoverable ``SyntaxError``, so
        this offset must never land before a future statement.
        """
        idx = 0
        n = len(lines)

        # Leading comments, shebangs and blank lines.
        while idx < n:
            stripped = lines[idx].strip()
            if stripped == "" or stripped.startswith("#"):
                idx += 1
            else:
                break

        # Module-level docstring.
        if idx < n:
            stripped = lines[idx].lstrip()
            for quote in ('"""', "'''"):
                if stripped.startswith(quote):
                    # Single-line docstring.
                    rest = stripped[len(quote):]
                    if rest.rstrip().endswith(quote) and len(stripped.rstrip()) > len(quote):
                        idx += 1
                    else:
                        idx += 1
                        while idx < n and quote not in lines[idx]:
                            idx += 1
                        if idx < n:
                            idx += 1  # consume the closing-quote line
                    break

        # Existing `from __future__ import ...` statements and the blank
        # lines/comments interleaved with them. These compiler flags must
        # stay pinned ahead of any newly injected import.
        while idx < n:
            stripped = lines[idx].strip()
            if stripped.startswith("from __future__ import"):
                idx += 1
            elif stripped == "" or stripped.startswith("#"):
                idx += 1
            else:
                break

        return idx

    @staticmethod
    def _statement_present(lines: List[str], statement: str) -> bool:
        """True when ``statement`` already exists (idempotency guard)."""
        target = statement.strip()
        return any(line.strip() == target for line in lines)

    # ------------------------------------------------------------------ #
    # Import grouping
    # ------------------------------------------------------------------ #
    # Python standard library module names for import classification.
    _STDLIB_MODULES: frozenset = (
        frozenset(sys.stdlib_module_names)
        if hasattr(sys, "stdlib_module_names")
        else frozenset({
            "abc", "aifc", "argparse", "ast", "asyncio", "atexit", "base64",
            "binascii", "builtins", "calendar", "cgi", "cmd", "codecs",
            "collections", "colorsys", "compileall", "concurrent", "configparser",
            "contextlib", "copy", "copyreg", "csv", "ctypes", "curses",
            "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
            "distutils", "doctest", "email", "enum", "errno", "faulthandler",
            "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
            "functools", "gc", "getopt", "getpass", "gettext", "glob",
            "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib",
            "importlib", "inspect", "io", "ipaddress", "itertools", "json",
            "keyword", "linecache", "locale", "logging", "lzma", "mailbox",
            "math", "mimetypes", "mmap", "multiprocessing", "netrc",
            "numbers", "operator", "optparse", "os", "pathlib", "pdb",
            "pickle", "pkgutil", "platform", "plistlib", "pprint",
            "profile", "pstats", "queue", "quopri", "random", "re",
            "readline", "reprlib", "resource", "rlcompleter", "runpy",
            "sched", "secrets", "select", "selectors", "shelve", "shlex",
            "shutil", "signal", "site", "smtplib", "socket", "socketserver",
            "sqlite3", "ssl", "stat", "statistics", "string", "struct",
            "subprocess", "sys", "sysconfig", "syslog", "tarfile", "tempfile",
            "test", "textwrap", "threading", "time", "timeit", "tkinter",
            "token", "tokenize", "tomllib", "trace", "traceback",
            "tracemalloc", "turtle", "types", "typing", "unicodedata",
            "unittest", "urllib", "uuid", "venv", "warnings", "wave",
            "weakref", "webbrowser", "xml", "xmlrpc", "zipfile", "zipimport",
            "zlib", "_thread", "__future__",
        })
    )

    # Pattern to detect top-level import/from-import lines.
    _IMPORT_LINE_RE: re.Pattern = re.compile(
        r"^\s*(import\s+\S+|from\s+\S+\s+import\s+.+)\s*$"
    )

    # Known project-internal namespace prefixes.
    _PROJECT_PREFIXES = ("aero_nova",)

    @classmethod
    def _classify_import(cls, line: str) -> int:
        """Classify an import line into group 1 (stdlib), 2 (third-party), or 3 (project)."""
        stripped = line.strip()
        # Extract the top-level module name.
        if stripped.startswith("from "):
            module = stripped.split()[1].split(".")[0]
        elif stripped.startswith("import "):
            module = stripped.split()[1].split(".")[0].rstrip(",")
        else:
            return 2  # fallback

        # Project-internal check.
        for prefix in cls._PROJECT_PREFIXES:
            if module == prefix or stripped.split()[1].startswith(prefix + "."):
                return 3

        # Standard library check.
        if module in cls._STDLIB_MODULES:
            return 1

        return 2  # third-party

    @classmethod
    def _group_python_imports(cls, lines: List[str]) -> List[str]:
        """Re-sort and group all imports at the header of the file.

        Imports are collected from the header section (after docstrings/comments
        and before the first non-import statement), categorised, sorted
        alphabetically within each group, then re-inserted with exactly one
        blank line between each group.
        """
        header_end = cls._python_header_offset(lines)

        # Find the import section boundaries: starts at header_end, ends when
        # we encounter a non-import, non-blank, non-comment line.
        import_start = header_end
        idx = import_start
        n = len(lines)
        import_lines: List[str] = []

        while idx < n:
            stripped = lines[idx].strip()
            if cls._IMPORT_LINE_RE.match(lines[idx]):
                import_lines.append(lines[idx])
                idx += 1
            elif stripped == "" or stripped.startswith("#"):
                # Blank or comment between imports — skip for now.
                idx += 1
            else:
                break
        import_end = idx

        if not import_lines:
            return lines

        # Categorise.
        groups: Dict[int, List[str]] = {1: [], 2: [], 3: []}
        for imp_line in import_lines:
            grp = cls._classify_import(imp_line)
            normalised = imp_line.rstrip("\n\r") + "\n"
            if normalised not in groups[grp]:
                groups[grp].append(normalised)

        # Sort within each group.
        for grp in groups:
            groups[grp].sort(key=lambda s: s.strip().lower())

        # Assemble grouped block.
        grouped_block: List[str] = []
        for grp_id in (1, 2, 3):
            if groups[grp_id]:
                if grouped_block:
                    grouped_block.append("\n")
                grouped_block.extend(groups[grp_id])

        # Ensure a trailing blank line after the import block for spacing.
        if grouped_block and (import_end < n and lines[import_end].strip() != ""):
            grouped_block.append("\n")

        return lines[:import_start] + grouped_block + lines[import_end:]

    # ------------------------------------------------------------------ #
    # Rust patches
    # ------------------------------------------------------------------ #
    def _inject_rust_use(self, lines: List[str], item: str) -> List[str]:
        """Prepend ``use crate::modules::{item};`` to the file."""
        statement = f"use {self.RUST_MODULE_NAMESPACE}::{item};\n"
        if self._statement_present(lines, statement):
            return lines
        insert_at = self._rust_header_offset(lines)
        return lines[:insert_at] + [statement] + lines[insert_at:]

    @staticmethod
    def _rust_header_offset(lines: List[str]) -> int:
        """Return the index after leading Rust comments / attributes / blanks."""
        idx = 0
        n = len(lines)
        while idx < n:
            stripped = lines[idx].strip()
            if (
                stripped == ""
                or stripped.startswith("//")
                or stripped.startswith("#![")  # inner attributes / crate attrs
            ):
                idx += 1
            else:
                break
        return idx
