"""Stateful JSON-RPC Language Server Protocol (v3.17) client proxy.

A dependency-free LSP client that manages the connection lifecycle for language
servers (``clangd``, ``rust-analyzer``, ``pyright``, ...) and intercepts
``textDocument/publishDiagnostics`` notifications.  No cloud services, no LLMs.

The proxy enforces the precise initialization workflow so servers never reject
out-of-order synchronization messages:

1. ``initialize`` (with the target ``rootUri``) -> await the result,
2. send the ``initialized`` notification (session confirmed),
3. ``textDocument/didOpen`` to register a file buffer in memory,
4. incremental ``textDocument/didChange`` for live edits.

Diagnostics published by the server are captured as structured records
(message, file target, code, severity and precise line/character + byte spans).

Transport is abstracted: :class:`StdioTransport` drives a real server
subprocess; tests inject an in-memory transport, so the JSON-RPC framing and the
lifecycle state machine are exercised deterministically without a live server.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

_PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# JSON-RPC framing transports
# ---------------------------------------------------------------------------
class Transport:
    """Byte-stream transport with LSP ``Content-Length`` framing."""

    def write_message(self, payload: dict) -> None:
        raise NotImplementedError

    def read_message(self) -> Optional[dict]:
        """Read one framed message, or ``None`` at end-of-stream."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _encode(payload: dict) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _read_framed(readline, read_exact) -> Optional[dict]:
    """Parse one framed message using a ``readline``/``read(n)`` pair."""
    content_length = None
    while True:
        line = readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        text = line.decode("ascii", "replace").strip()
        if text.lower().startswith("content-length:"):
            content_length = int(text.split(":", 1)[1].strip())
    if content_length is None:
        return None
    body = read_exact(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


class StdioTransport(Transport):
    """Drives a language-server subprocess over stdin/stdout."""

    def __init__(self, command: List[str]) -> None:
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._wlock = threading.Lock()

    def write_message(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        with self._wlock:
            self.proc.stdin.write(_encode(payload))
            self.proc.stdin.flush()

    def read_message(self) -> Optional[dict]:
        assert self.proc.stdout is not None
        return _read_framed(self.proc.stdout.readline, self.proc.stdout.read)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        finally:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# State machine + data records
# ---------------------------------------------------------------------------
class LspState(str, Enum):
    DISCONNECTED = "disconnected"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"


class LspStateError(Exception):
    """Raised when an operation is attempted in the wrong lifecycle state."""


@dataclass
class Position:
    line: int
    character: int
    byte: Optional[int] = None


@dataclass
class Diagnostic:
    uri: str
    message: str
    severity: Optional[int]
    code: Optional[Union[str, int]]
    start: Position
    end: Position
    source: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
            "source": self.source,
            "start": {"line": self.start.line, "character": self.start.character, "byte": self.start.byte},
            "end": {"line": self.end.line, "character": self.end.character, "byte": self.end.byte},
        }


@dataclass
class _Buffer:
    uri: str
    language_id: str
    version: int
    text: str


def path_to_uri(path: _PathLike) -> str:
    return Path(path).resolve().as_uri()


def _position_to_byte(text: str, line: int, character: int) -> Optional[int]:
    """Convert an LSP (line, character) position to a byte offset in *text*."""
    lines = text.splitlines(keepends=True)
    if line < 0 or line > len(lines):
        return None
    offset = sum(len(lines[i].encode("utf-8")) for i in range(min(line, len(lines))))
    if line < len(lines):
        prefix = lines[line][:character]
        offset += len(prefix.encode("utf-8"))
    return offset


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------
class LspProxy:
    def __init__(
        self,
        transport: Transport,
        root_path: Optional[_PathLike] = None,
        *,
        client_name: str = "aero-nova",
    ) -> None:
        self.transport = transport
        self.root_path = Path(root_path).resolve() if root_path else Path.cwd()
        self.client_name = client_name

        self.state = LspState.DISCONNECTED
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._pending: Dict[int, threading.Event] = {}
        self._responses: Dict[int, dict] = {}
        self._buffers: Dict[str, _Buffer] = {}
        self._diagnostics: Dict[str, List[Diagnostic]] = {}
        self._diag_event = threading.Event()
        self._notification_handlers: Dict[str, Callable[[dict], None]] = {
            "textDocument/publishDiagnostics": self._on_publish_diagnostics,
        }
        self._reader: Optional[threading.Thread] = None
        self._running = False

    # -- reader loop ----------------------------------------------------------
    def _start_reader(self) -> None:
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._running:
            try:
                message = self.transport.read_message()
            except Exception:
                break
            if message is None:
                break
            self._dispatch(message)

    def _dispatch(self, message: dict) -> None:
        if "id" in message and ("result" in message or "error" in message):
            mid = message["id"]
            self._responses[mid] = message
            event = self._pending.get(mid)
            if event:
                event.set()
            return
        method = message.get("method")
        if method and "id" not in message:  # notification
            handler = self._notification_handlers.get(method)
            if handler:
                handler(message.get("params") or {})

    # -- JSON-RPC primitives --------------------------------------------------
    def _alloc_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _send_request(self, method: str, params: dict, timeout: float = 10.0) -> dict:
        mid = self._alloc_id()
        event = threading.Event()
        self._pending[mid] = event
        self.transport.write_message({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        if not event.wait(timeout):
            self._pending.pop(mid, None)
            raise TimeoutError(f"LSP request '{method}' timed out")
        self._pending.pop(mid, None)
        response = self._responses.pop(mid)
        if "error" in response:
            raise LspStateError(f"LSP error for '{method}': {response['error']}")
        return response.get("result", {})

    def _send_notification(self, method: str, params: dict) -> None:
        self.transport.write_message({"jsonrpc": "2.0", "method": method, "params": params})

    # -- lifecycle ------------------------------------------------------------
    def initialize(self, capabilities: Optional[dict] = None, timeout: float = 10.0) -> dict:
        """Step 1+2: send ``initialize``, await the result, send ``initialized``."""
        if self.state != LspState.DISCONNECTED:
            raise LspStateError(f"initialize() invalid in state {self.state.value}")
        self._start_reader()
        self.state = LspState.INITIALIZING
        params = {
            "processId": None,
            "clientInfo": {"name": self.client_name, "version": "0.1.0"},
            "rootUri": self.root_path.as_uri(),
            "rootPath": str(self.root_path),
            "capabilities": capabilities or {
                "textDocument": {
                    "synchronization": {"didSave": True, "dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                }
            },
            "workspaceFolders": [{"uri": self.root_path.as_uri(), "name": self.root_path.name}],
        }
        result = self._send_request("initialize", params, timeout=timeout)
        # Step 2: confirm the session.
        self._send_notification("initialized", {})
        self.state = LspState.INITIALIZED
        return result

    def _require_initialized(self, op: str) -> None:
        if self.state != LspState.INITIALIZED:
            raise LspStateError(f"{op} requires INITIALIZED state (current: {self.state.value})")

    def did_open(self, path: _PathLike, text: str, language_id: str, version: int = 1) -> str:
        """Step 3: register a file buffer in memory and notify the server."""
        self._require_initialized("did_open")
        uri = path_to_uri(path)
        self._buffers[uri] = _Buffer(uri=uri, language_id=language_id, version=version, text=text)
        self._send_notification("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": language_id, "version": version, "text": text}
        })
        return uri

    def did_change_full(self, path: _PathLike, text: str) -> None:
        """Convenience: a single full-document change event."""
        uri = path_to_uri(path)
        self.did_change(path, [{"text": text}], new_text=text)

    def did_change(self, path: _PathLike, content_changes: List[dict], new_text: Optional[str] = None) -> None:
        """Step 4: broadcast incremental edits; bumps the buffer version."""
        self._require_initialized("did_change")
        uri = path_to_uri(path)
        buffer = self._buffers.get(uri)
        if buffer is None:
            raise LspStateError(f"did_change before did_open for {uri}")
        buffer.version += 1
        if new_text is not None:
            buffer.text = new_text
        elif len(content_changes) == 1 and "range" not in content_changes[0]:
            buffer.text = content_changes[0]["text"]
        self._diag_event.clear()
        self._send_notification("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": buffer.version},
            "contentChanges": content_changes,
        })

    def shutdown(self) -> None:
        if self.state == LspState.INITIALIZED:
            try:
                self._send_request("shutdown", {}, timeout=5.0)
                self._send_notification("exit", {})
            except Exception:
                pass
        self.state = LspState.SHUTDOWN
        self._running = False
        self.transport.close()

    # -- diagnostics ----------------------------------------------------------
    def _on_publish_diagnostics(self, params: dict) -> None:
        uri = params.get("uri", "")
        buffer = self._buffers.get(uri)
        diagnostics: List[Diagnostic] = []
        for item in params.get("diagnostics", []):
            rng = item.get("range", {})
            start = rng.get("start", {})
            end = rng.get("end", {})
            start_pos = Position(start.get("line", 0), start.get("character", 0))
            end_pos = Position(end.get("line", 0), end.get("character", 0))
            if buffer is not None:
                start_pos.byte = _position_to_byte(buffer.text, start_pos.line, start_pos.character)
                end_pos.byte = _position_to_byte(buffer.text, end_pos.line, end_pos.character)
            diagnostics.append(Diagnostic(
                uri=uri,
                message=item.get("message", ""),
                severity=item.get("severity"),
                code=item.get("code"),
                source=item.get("source"),
                start=start_pos,
                end=end_pos,
            ))
        self._diagnostics[uri] = diagnostics
        self._diag_event.set()

    def get_diagnostics(self, path: _PathLike) -> List[Diagnostic]:
        return list(self._diagnostics.get(path_to_uri(path), []))

    def wait_for_diagnostics(self, timeout: float = 10.0) -> bool:
        """Block until a ``publishDiagnostics`` arrives (or timeout)."""
        return self._diag_event.wait(timeout)

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "LspProxy":
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()
