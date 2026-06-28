"""
lsp_proxy.py — Stateful LSP Diagnostic Reflux Binder

Captures diagnostic feedback emitted by real-time Language Servers (Pyright,
rust-analyzer, Clangd) over the JSON-RPC wire protocol and converts it into
structured *reflux* mutation commands. Those commands are consumed downstream by
:class:`builder_brains.reflux.AeroDependencyRefluxEngine`, which performs the
in-memory file manipulations needed to heal higher-level semantic bugs (missing
relative imports, unlinked external variables, unresolved Rust modules) *before*
the compiler runs.

Pipeline:
  raw JSON-RPC frame  ->  ``ingest_payload``
                      ->  filter ``textDocument/publishDiagnostics``
                      ->  keep error/warning severities
                      ->  map diagnostic ``code`` -> mutation command
                      ->  accumulate in ``pending_reflux_commands``
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

# LSP DiagnosticSeverity enum (per the Language Server Protocol spec).
SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFORMATION = 3
SEVERITY_HINT = 4

# Severities we act on. Stylistic info/hint notices are intentionally skipped.
ACTIONABLE_SEVERITIES = frozenset({SEVERITY_ERROR, SEVERITY_WARNING})

PUBLISH_DIAGNOSTICS_METHOD = "textDocument/publishDiagnostics"

# Maps a diagnostic ``code`` (as produced by the language server) to the
# structured mutation command the reflux engine understands.
DIAGNOSTIC_CODE_MAP: Dict[str, str] = {
    # Pyright / Python
    "reportUndefinedVariable": "RESOLVE_UNDEFINED_SYMBOL",
    "reportMissingImports": "AUTO_REFLUX_IMPORT",
    # rust-analyzer / rustc
    "E0433": "INJECT_RUST_USE_DECLARATION",
    "E0405": "INJECT_RUST_USE_DECLARATION",
}

# Per-delimiter extractors. Matching each delimiter family independently is the
# key to surviving *nested* delimiters (e.g. a backtick token that itself
# contains a quote): we never let a quote inside a backticked span terminate the
# backtick match, which the old single combined character class did.
#   rust-analyzer / rustc : cannot find value `x` in `Foo`     -> x   (backtick)
#   Pyright               : "bar.baz" is not defined           -> bar.baz (dquote)
#   clangd                : use of undeclared identifier 'foo'  -> foo (squote)
_DELIMITER_PATTERNS = (
    re.compile(r"`([^`]+)`"),      # rust-analyzer / rustc (highest priority)
    re.compile(r'"([^"]+)"'),      # Pyright
    re.compile(r"'([^']+)'"),      # clangd / gcc
)

# A token that *looks like* a code symbol or dotted/scoped path. Used to skip
# quoted prose ("could not be resolved") and pick the real identifier.
_SYMBOL_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[.:]{1,2}[A-Za-z_][A-Za-z0-9_]*)*$")

# Retained for backward compatibility with external callers/tests.
_SYMBOL_PATTERN = re.compile(r"[\"'`]([^\"'`]+)[\"'`]")


class LspDiagnosticRefluxBinder:
    """Decode JSON-RPC diagnostic traffic into pending reflux commands.

    The binder is stateful: every actionable diagnostic produces a command that
    is appended to :attr:`pending_reflux_commands`, keyed by the resolved file
    path. Callers drain that map and feed each entry to the reflux engine.
    """

    def __init__(self) -> None:
        # file_path -> list of mutation command dicts awaiting application.
        self.pending_reflux_commands: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # Wire decoding
    # ------------------------------------------------------------------ #
    def ingest_payload(self, raw_payload: Any) -> List[Dict[str, Any]]:
        """Decode an inbound JSON-RPC payload and route diagnostics.

        Accepts a ``bytes`` / ``str`` JSON frame, or an already-decoded ``dict``.
        Returns the list of mutation commands generated from this payload (also
        merged into :attr:`pending_reflux_commands`).
        """
        message = self._decode_jsonrpc(raw_payload)
        if message is None:
            return []

        if message.get("method") != PUBLISH_DIAGNOSTICS_METHOD:
            return []

        params = message.get("params") or {}
        return self._handle_publish_diagnostics(params)

    def _decode_jsonrpc(self, raw_payload: Any) -> Optional[Dict[str, Any]]:
        """Parse a JSON-RPC frame into a dict, tolerating LSP header framing."""
        if isinstance(raw_payload, dict):
            return raw_payload

        if isinstance(raw_payload, (bytes, bytearray)):
            try:
                raw_payload = raw_payload.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("lsp_proxy: undecodable payload bytes")
                return None

        if not isinstance(raw_payload, str):
            return None

        # Strip optional ``Content-Length`` header block: headers and body are
        # separated by a blank line (``\r\n\r\n`` or ``\n\n``).
        body = raw_payload
        for sep in ("\r\n\r\n", "\n\n"):
            if sep in raw_payload:
                body = raw_payload.split(sep, 1)[1]
                break

        body = body.strip()
        if not body:
            return None

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("lsp_proxy: malformed JSON-RPC frame: %s", exc)
            return None

        return decoded if isinstance(decoded, dict) else None

    # ------------------------------------------------------------------ #
    # Diagnostic handling
    # ------------------------------------------------------------------ #
    def _handle_publish_diagnostics(
        self, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter diagnostics by severity and convert to mutation commands."""
        file_path = self._uri_to_path(params.get("uri", ""))
        diagnostics = params.get("diagnostics") or []

        produced: List[Dict[str, Any]] = []
        for diag in diagnostics:
            if not isinstance(diag, dict):
                continue

            # Default to ERROR when severity is omitted (servers may elide it).
            severity = diag.get("severity", SEVERITY_ERROR)
            if severity not in ACTIONABLE_SEVERITIES:
                continue

            command = self._build_command(file_path, diag)
            if command is not None:
                produced.append(command)

        if produced:
            self.pending_reflux_commands.setdefault(file_path, []).extend(produced)

        return produced

    def _build_command(
        self, file_path: str, diag: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Map a single diagnostic to a structured reflux command, or ``None``."""
        code = self._normalize_code(diag.get("code"))
        action = DIAGNOSTIC_CODE_MAP.get(code)
        if action is None:
            return None

        message = diag.get("message", "") or ""
        symbol = self.extract_symbol(message)

        return {
            "action": action,
            "code": code,
            "symbol": symbol,
            "file_path": file_path,
            "message": message,
            "range": diag.get("range"),
            "severity": diag.get("severity", SEVERITY_ERROR),
            "source": diag.get("source"),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_code(code: Any) -> str:
        """Coerce a diagnostic ``code`` (str | int | dict) to a plain string."""
        if isinstance(code, dict):
            # rust-analyzer emits ``{"value": "E0433", ...}``.
            code = code.get("value", "")
        if code is None:
            return ""
        return str(code)

    @staticmethod
    def extract_symbol(message: str) -> Optional[str]:
        """Extract the offending symbol token from a diagnostic message string.

        Robust across Pyright (double quotes), rust-analyzer/rustc (backticks)
        and clangd/gcc (single quotes), including messages that mix or *nest*
        delimiters. Each delimiter family is scanned independently (so a quote
        inside a backticked span can never truncate the backtick token), and the
        first candidate that looks like a real identifier/path wins. If no
        candidate looks like a symbol, the first quoted token is returned as a
        best-effort fallback. Returns ``None`` when nothing can be isolated.
        """
        if not message:
            return None

        first_fallback: Optional[str] = None
        for pattern in _DELIMITER_PATTERNS:
            for raw in pattern.findall(message):
                token = raw.strip()
                if not token:
                    continue
                if first_fallback is None:
                    first_fallback = token
                # Prefer a clean identifier/dotted-path token over quoted prose.
                if _SYMBOL_TOKEN.match(token):
                    return token
        return first_fallback

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        """Convert a ``file://`` document URI to a local filesystem path."""
        if not uri:
            return ""
        if uri.startswith("file://"):
            parsed = urlparse(uri)
            return unquote(parsed.path)
        return uri

    def drain(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return and clear all accumulated pending reflux commands."""
        snapshot = self.pending_reflux_commands
        self.pending_reflux_commands = {}
        return snapshot
