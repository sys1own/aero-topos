# -*- coding: utf-8 -*-
"""Tests for the stateful LSP client proxy (in-memory fake server)."""

import io
import os
import queue
import tempfile
import threading
import unittest

from core.toolchain.lsp_proxy import (
    LspProxy,
    LspState,
    LspStateError,
    Transport,
    _encode,
    _read_framed,
    path_to_uri,
)
from core.toolchain.lsp_proxy import _position_to_byte


class LoopbackTransport(Transport):
    """In-memory transport wired to a scripted fake LSP server thread."""

    def __init__(self):
        self.to_server = queue.Queue()
        self.to_client = queue.Queue()
        self._closed = False

    def write_message(self, payload):
        self.to_server.put(payload)

    def read_message(self):
        return self.to_client.get()

    def close(self):
        if not self._closed:
            self._closed = True
            self.to_client.put(None)


def _fake_server(t: LoopbackTransport, *, diag_range=((1, 4), (1, 7))):
    (s_line, s_char), (e_line, e_char) = diag_range
    while True:
        msg = t.to_server.get()
        if msg is None:
            break
        method = msg.get("method")
        if method == "initialize":
            t.to_client.put({"jsonrpc": "2.0", "id": msg["id"],
                             "result": {"capabilities": {"textDocumentSync": 2}}})
        elif method == "textDocument/didOpen":
            uri = msg["params"]["textDocument"]["uri"]
            t.to_client.put({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                             "params": {"uri": uri, "diagnostics": [{
                                 "range": {"start": {"line": s_line, "character": s_char},
                                           "end": {"line": e_line, "character": e_char}},
                                 "message": "undefined name 'foo'", "severity": 1,
                                 "code": "E0001", "source": "fake"}]}})
        elif method == "textDocument/didChange":
            uri = msg["params"]["textDocument"]["uri"]
            t.to_client.put({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                             "params": {"uri": uri, "diagnostics": []}})
        elif method == "shutdown":
            t.to_client.put({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        elif method == "exit":
            break


class _ProxyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = self.tmp.name
        self.file = os.path.join(self.ws, "buf.py")
        self.transport = LoopbackTransport()
        self.server = threading.Thread(target=_fake_server, args=(self.transport,), daemon=True)
        self.server.start()
        self.proxy = LspProxy(self.transport, root_path=self.ws)

    def tearDown(self):
        try:
            self.proxy.shutdown()
        except Exception:
            pass
        self.tmp.cleanup()


class TestFraming(unittest.TestCase):
    def test_encode_read_roundtrip(self):
        payload = {"jsonrpc": "2.0", "method": "x", "params": {"a": 1}}
        raw = _encode(payload)
        self.assertIn(b"Content-Length:", raw)
        stream = io.BytesIO(raw)
        parsed = _read_framed(stream.readline, stream.read)
        self.assertEqual(parsed, payload)

    def test_read_framed_eof(self):
        stream = io.BytesIO(b"")
        self.assertIsNone(_read_framed(stream.readline, stream.read))


class TestPositionToByte(unittest.TestCase):
    def test_byte_offset(self):
        text = "def f():\n    foo\n"  # line0=9 bytes incl \n
        self.assertEqual(_position_to_byte(text, 1, 4), 13)  # start of 'foo'
        self.assertEqual(_position_to_byte(text, 1, 7), 16)  # end of 'foo'

    def test_utf8_multibyte(self):
        text = "x = 'é'\n"  # 'é' is 2 bytes in utf-8
        # character index 6 is after the é; byte offset accounts for 2 bytes.
        self.assertEqual(_position_to_byte(text, 0, 7), len("x = 'é'".encode("utf-8")))


class TestLifecycle(_ProxyCase):
    def test_didopen_before_initialize_rejected(self):
        with self.assertRaises(LspStateError):
            self.proxy.did_open(self.file, "x=1\n", "python")

    def test_initialize_transitions_state(self):
        self.assertEqual(self.proxy.state, LspState.DISCONNECTED)
        result = self.proxy.initialize()
        self.assertEqual(self.proxy.state, LspState.INITIALIZED)
        self.assertEqual(result["capabilities"]["textDocumentSync"], 2)

    def test_double_initialize_rejected(self):
        self.proxy.initialize()
        with self.assertRaises(LspStateError):
            self.proxy.initialize()

    def test_shutdown_sets_state(self):
        self.proxy.initialize()
        self.proxy.shutdown()
        self.assertEqual(self.proxy.state, LspState.SHUTDOWN)


class TestDiagnostics(_ProxyCase):
    def test_didopen_captures_diagnostics_with_spans(self):
        self.proxy.initialize()
        src = "def f():\n    foo\n"
        self.proxy.did_open(self.file, src, "python")
        self.assertTrue(self.proxy.wait_for_diagnostics(5))
        diags = self.proxy.get_diagnostics(self.file)
        self.assertEqual(len(diags), 1)
        d = diags[0]
        self.assertEqual(d.message, "undefined name 'foo'")
        self.assertEqual(d.code, "E0001")
        self.assertEqual(d.severity, 1)
        self.assertEqual((d.start.line, d.start.character), (1, 4))
        # Byte spans computed from the in-memory buffer.
        self.assertEqual(d.start.byte, 13)
        self.assertEqual(d.end.byte, 16)

    def test_didchange_requires_open(self):
        self.proxy.initialize()
        with self.assertRaises(LspStateError):
            self.proxy.did_change(self.file, [{"text": "x=1\n"}], new_text="x=1\n")

    def test_didchange_bumps_version_and_updates_diagnostics(self):
        self.proxy.initialize()
        src = "def f():\n    foo\n"
        uri = self.proxy.did_open(self.file, src, "python")
        self.assertTrue(self.proxy.wait_for_diagnostics(5))
        self.assertEqual(len(self.proxy.get_diagnostics(self.file)), 1)
        v_before = self.proxy._buffers[uri].version
        self.proxy.did_change(self.file, [{"text": "def f():\n    return 1\n"}],
                              new_text="def f():\n    return 1\n")
        self.assertEqual(self.proxy._buffers[uri].version, v_before + 1)
        self.assertTrue(self.proxy.wait_for_diagnostics(5))
        self.assertEqual(self.proxy.get_diagnostics(self.file), [])

    def test_incremental_change_event_passthrough(self):
        self.proxy.initialize()
        self.proxy.did_open(self.file, "abc\n", "python")
        self.proxy.wait_for_diagnostics(5)
        # An incremental (ranged) change updates version even without new_text.
        uri = path_to_uri(self.file)
        self.proxy.did_change(self.file, [{
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
            "text": "A",
        }])
        self.assertGreaterEqual(self.proxy._buffers[uri].version, 2)


if __name__ == "__main__":
    unittest.main()
