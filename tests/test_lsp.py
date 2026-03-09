"""Tests for jaxtyc.lsp.server — LSP integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _lsp_message(method: str, params: dict, msg_id: int | None = None) -> bytes:
    """Encode an LSP JSON-RPC message with Content-Length header."""
    body: dict = {"jsonrpc": "2.0", "method": method, "params": params}
    if msg_id is not None:
        body["id"] = msg_id
    content = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
    return header + content


def _lsp_response(msg_id: int, result: dict) -> bytes:
    body = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    content = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
    return header + content


def _parse_lsp_messages(data: bytes) -> list[dict]:
    """Parse LSP messages from raw bytes."""
    messages = []
    text = data.decode("utf-8", errors="replace")
    while "Content-Length:" in text:
        header_end = text.index("\r\n\r\n")
        header = text[:header_end]
        cl_part = header.split("Content-Length:")[1]
        # May be followed by other headers separated by \r\n
        length = int(cl_part.split("\r\n")[0].strip())
        body_start = header_end + 4
        body = text[body_start : body_start + length]
        try:
            messages.append(json.loads(body))
        except json.JSONDecodeError:
            pass
        text = text[body_start + length :]
    return messages


class _Session:
    """Mutable state holder for an LSP test session."""

    def __init__(self, proc: subprocess.Popen, uri: str | None):
        self._proc = proc
        self.uri: str | None = uri
        self._next_id = 10
        self.messages: list[dict] = []

    def request(self, method: str, params: dict) -> int:
        """Send a JSON-RPC request and return the message ID."""
        msg_id = self._next_id
        self._next_id += 1
        self._proc.stdin.write(_lsp_message(method, params, msg_id=msg_id))
        self._proc.stdin.flush()
        return msg_id


def _find_response(messages: list[dict], msg_id: int) -> dict | None:
    """Find a response message by ID."""
    return next((m for m in messages if m.get("id") == msg_id), None)


@contextmanager
def _lsp_session(fixture: str | None = None, wait: float = 2.0):
    """Start an LSP server, initialize, optionally open a fixture, yield session, shutdown."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    proc.stdin.write(
        _lsp_message(
            "initialize",
            {"processId": None, "capabilities": {}, "rootUri": None},
            msg_id=1,
        )
    )
    proc.stdin.flush()
    proc.stdin.write(_lsp_message("initialized", {}))
    proc.stdin.flush()

    fixture_uri = None
    if fixture is not None:
        fixture_path = str(FIXTURES / fixture)
        fixture_uri = Path(fixture_path).as_uri()
        source = Path(fixture_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": fixture_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": source,
                    }
                },
            )
        )
        proc.stdin.flush()
        time.sleep(wait)

    session = _Session(proc, fixture_uri)
    try:
        yield session
    finally:
        time.sleep(0.5)
        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=999))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()
        stdout, _ = proc.communicate(timeout=10)
        session.messages = _parse_lsp_messages(stdout)


class TestLSPServer:
    def test_initialize_and_shutdown(self):
        """Test that the LSP server responds to initialize and shutdown."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Send initialize
        init_params = {
            "processId": None,
            "capabilities": {},
            "rootUri": None,
        }
        proc.stdin.write(_lsp_message("initialize", init_params, msg_id=1))
        proc.stdin.flush()

        # Send initialized notification
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Send shutdown
        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=2))
        proc.stdin.flush()

        # Send exit
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        stdout, stderr = proc.communicate(timeout=10)

        # Parse response
        messages = _parse_lsp_messages(stdout)
        # Should have at least the initialize response
        assert len(messages) >= 1
        init_response = messages[0]
        assert "result" in init_response
        assert "capabilities" in init_response["result"]

    def test_diagnostics_on_save(self):
        """Test that saving a file with shape errors produces diagnostics."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "wrong_transpose.py")
        fixture_uri = Path(fixture_path).as_uri()

        # Initialize
        proc.stdin.write(
            _lsp_message(
                "initialize", {"processId": None, "capabilities": {}, "rootUri": None}, msg_id=1
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Open the document
        source = Path(fixture_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": fixture_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": source,
                    }
                },
            )
        )
        proc.stdin.flush()

        # Save the document
        proc.stdin.write(
            _lsp_message(
                "textDocument/didSave",
                {"textDocument": {"uri": fixture_uri}},
            )
        )
        proc.stdin.flush()

        # Give server time to process, then shutdown
        import time

        time.sleep(2)

        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=2))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        stdout, stderr = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)

        # Should have diagnostics notification
        diag_messages = [
            m for m in messages if m.get("method") == "textDocument/publishDiagnostics"
        ]
        assert len(diag_messages) >= 1
        diag = diag_messages[0]
        assert len(diag["params"]["diagnostics"]) >= 1
        assert diag["params"]["diagnostics"][0]["severity"] == 1  # Error

    def test_server_responsive_during_analysis(self):
        """Server should respond to shutdown promptly even after triggering analysis.

        This verifies that analysis runs in a thread and doesn't block the event loop.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "correct_attention.py")
        fixture_uri = Path(fixture_path).as_uri()

        # Initialize
        proc.stdin.write(
            _lsp_message(
                "initialize", {"processId": None, "capabilities": {}, "rootUri": None}, msg_id=1
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Open a file (triggers analysis)
        source = Path(fixture_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": fixture_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": source,
                    }
                },
            )
        )
        proc.stdin.flush()

        # Immediately send shutdown — if threaded, this should respond quickly
        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=2))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        start = time.perf_counter()
        stdout, stderr = proc.communicate(timeout=10)
        elapsed = time.perf_counter() - start

        # Server should shut down within 5 seconds even if analysis is pending
        assert elapsed < 5.0, f"Server took {elapsed:.1f}s to shut down, expected < 5s"

        messages = _parse_lsp_messages(stdout)
        # Should have at least the initialize response
        assert any("result" in m for m in messages)

    def test_diagnostics_on_change(self):
        """Editing a correct file to buggy content via didChange should produce error diagnostics."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Use a correct file for didOpen (no errors)
        correct_path = str(FIXTURES / "correct_attention.py")
        correct_uri = Path(correct_path).as_uri()

        # Initialize
        proc.stdin.write(
            _lsp_message(
                "initialize", {"processId": None, "capabilities": {}, "rootUri": None}, msg_id=1
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Open correct file — should produce 0 errors
        correct_source = Path(correct_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": correct_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": correct_source,
                    }
                },
            )
        )
        proc.stdin.flush()

        # Wait for didOpen analysis to finish
        time.sleep(2)

        # Now send didChange with buggy content (wrong_transpose)
        buggy_source = Path(str(FIXTURES / "wrong_transpose.py")).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": correct_uri, "version": 2},
                    "contentChanges": [{"text": buggy_source}],
                },
            )
        )
        proc.stdin.flush()

        # Wait for debounce + analysis
        time.sleep(3)

        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=2))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        stdout, stderr = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)

        # Get all diagnostic notifications
        diag_messages = [
            m for m in messages if m.get("method") == "textDocument/publishDiagnostics"
        ]

        # The LAST diagnostics should contain errors (from the buggy didChange content)
        assert len(diag_messages) >= 2, (
            f"Expected at least 2 diagnostic notifications (didOpen + didChange), got {len(diag_messages)}"
        )
        last_diag = diag_messages[-1]
        assert len(last_diag["params"]["diagnostics"]) >= 1, (
            "didChange should have triggered analysis producing error diagnostics"
        )

    def test_progress_notification_on_analysis(self):
        """Server should send progress notifications during analysis."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "correct_attention.py")
        fixture_uri = Path(fixture_path).as_uri()

        # Initialize with window/workDoneProgress/create support
        proc.stdin.write(
            _lsp_message(
                "initialize",
                {
                    "processId": None,
                    "capabilities": {
                        "window": {"workDoneProgress": True},
                    },
                    "rootUri": None,
                },
                msg_id=1,
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Open file to trigger analysis with progress
        source = Path(fixture_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": fixture_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": source,
                    }
                },
            )
        )
        proc.stdin.flush()

        # Read stdout incrementally to respond to workDoneProgress/create requests
        # and collect all messages
        all_data = b""
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            import select

            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if ready:
                chunk = proc.stdout.read1(4096)
                if chunk:
                    all_data += chunk
                    # Check for workDoneProgress/create requests and respond
                    partial_msgs = _parse_lsp_messages(all_data)
                    for msg in partial_msgs:
                        if msg.get("method") == "window/workDoneProgress/create" and "id" in msg:
                            proc.stdin.write(_lsp_response(msg["id"], None))
                            proc.stdin.flush()

        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=2))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        stdout_rest, stderr = proc.communicate(timeout=10)
        all_data += stdout_rest
        messages = _parse_lsp_messages(all_data)

        # Should have progress-related messages (window/workDoneProgress/create request
        # or $/progress notifications)
        progress_messages = [
            m
            for m in messages
            if m.get("method") in ("window/workDoneProgress/create", "$/progress")
        ]
        assert len(progress_messages) >= 1, (
            f"Expected progress notifications, got messages: {[m.get('method') for m in messages]}"
        )

    def test_codelens_shows_shapes(self):
        """CodeLens should show shape annotations above jaxtyping-annotated functions."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "jaxtyc.cli.main", "lsp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        fixture_path = str(FIXTURES / "correct_attention.py")
        fixture_uri = Path(fixture_path).as_uri()

        # Initialize
        proc.stdin.write(
            _lsp_message(
                "initialize",
                {"processId": None, "capabilities": {}, "rootUri": None},
                msg_id=1,
            )
        )
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("initialized", {}))
        proc.stdin.flush()

        # Open and wait for analysis
        source = Path(fixture_path).read_text()
        proc.stdin.write(
            _lsp_message(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": fixture_uri,
                        "languageId": "python",
                        "version": 1,
                        "text": source,
                    }
                },
            )
        )
        proc.stdin.flush()
        time.sleep(2)

        # Request code lenses
        proc.stdin.write(
            _lsp_message(
                "textDocument/codeLens",
                {"textDocument": {"uri": fixture_uri}},
                msg_id=3,
            )
        )
        proc.stdin.flush()
        time.sleep(1)

        proc.stdin.write(_lsp_message("shutdown", {}, msg_id=4))
        proc.stdin.flush()
        proc.stdin.write(_lsp_message("exit", {}))
        proc.stdin.flush()

        stdout, stderr = proc.communicate(timeout=10)
        messages = _parse_lsp_messages(stdout)

        # Find the codeLens response (id=3)
        codelens_response = next((m for m in messages if m.get("id") == 3), None)
        assert codelens_response is not None, (
            f"No codeLens response found, got messages: {[m.get('id') for m in messages]}"
        )
        assert "result" in codelens_response
        lenses = codelens_response["result"]
        assert len(lenses) >= 1, "Expected at least one code lens for the attention function"

        # The lens should be on the function definition line (line 7, 0-indexed)
        lens = lenses[0]
        assert "range" in lens
        # Function `attention` is defined at line 8 (1-indexed), so 0-indexed = 7
        assert lens["range"]["start"]["line"] == 7

        # The lens command title should contain shape info
        assert "command" in lens
        title = lens["command"]["title"]
        assert "batch" in title
        assert "head_dim" in title


class TestLSPNavigation:
    """LSP navigation handler integration tests."""

    def test_document_symbol(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": s.uri}},
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        symbols = resp["result"]
        assert len(symbols) >= 1
        assert symbols[0]["name"] == "attention"
        assert "detail" in symbols[0]

    def test_definition_dim_jumps_to_first(self):
        """Click batch in k param, should jump to batch in q param."""
        with _lsp_session("correct_attention.py") as s:
            # batch in k param: line 10 (1-based) = line 9 (0-based), col 22
            rid = s.request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 9, "character": 22},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert result is not None
        # First occurrence: q param, line 9 (1-based) = line 8 (0-based)
        assert result["range"]["start"]["line"] == 8
        assert result["range"]["start"]["character"] == 21

    def test_definition_at_first_occurrence_returns_null(self):
        with _lsp_session("correct_attention.py") as s:
            # batch in q param: line 9 (1-based) = line 8 (0-based), col 22
            rid = s.request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 8, "character": 22},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        assert resp["result"] is None

    def test_references_dim(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 8, "character": 22},
                    "context": {"includeDeclaration": True},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        refs = resp["result"]
        # batch appears in q, k, v, return = 4 occurrences
        assert len(refs) == 4

    def test_document_highlight(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request(
                "textDocument/documentHighlight",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 8, "character": 22},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        highlights = resp["result"]
        assert len(highlights) == 4
        for h in highlights:
            assert h["kind"] == 2  # DocumentHighlightKind.Read

    def test_prepare_rename(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request(
                "textDocument/prepareRename",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 8, "character": 22},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert result is not None
        assert result["placeholder"] == "batch"
        assert result["range"]["start"]["line"] == 8
        assert result["range"]["start"]["character"] == 21
        assert result["range"]["end"]["character"] == 26

    def test_rename(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request(
                "textDocument/rename",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 8, "character": 22},
                    "newName": "batch_size",
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert result is not None
        edits = result["changes"][s.uri]
        assert len(edits) == 4
        for edit in edits:
            assert edit["newText"] == "batch_size"

    def test_workspace_symbol(self):
        with _lsp_session("correct_attention.py") as s:
            rid = s.request("workspace/symbol", {"query": "atten"})
        resp = _find_response(s.messages, rid)
        assert resp is not None
        symbols = resp["result"]
        assert len(symbols) >= 1
        assert symbols[0]["name"] == "attention"

    def test_implementation(self):
        with _lsp_session("correct_attention.py") as s:
            # def attention( is on line 8 (1-based) = line 7 (0-based)
            rid = s.request(
                "textDocument/implementation",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 7, "character": 5},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert result is not None
        assert result["range"]["start"]["line"] == 7

    def test_prepare_call_hierarchy(self):
        with _lsp_session("multi_function.py") as s:
            # autoencoder at line 22 (1-based) = line 21 (0-based)
            rid = s.request(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": s.uri},
                    "position": {"line": 21, "character": 5},
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        items = resp["result"]
        assert len(items) == 1
        assert items[0]["name"] == "autoencoder"
        assert "data" in items[0]

    def test_incoming_calls(self):
        with _lsp_session("multi_function.py") as s:
            rid = s.request(
                "callHierarchy/incomingCalls",
                {
                    "item": {
                        "name": "encode",
                        "kind": 12,
                        "uri": s.uri,
                        "range": {
                            "start": {"line": 7, "character": 0},
                            "end": {"line": 7, "character": 10},
                        },
                        "selectionRange": {
                            "start": {"line": 7, "character": 4},
                            "end": {"line": 7, "character": 10},
                        },
                        "data": {
                            "function_name": "encode",
                            "class_name": None,
                            "uri": s.uri,
                        },
                    }
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert len(result) == 1
        assert result[0]["from"]["name"] == "autoencoder"

    def test_outgoing_calls(self):
        with _lsp_session("multi_function.py") as s:
            rid = s.request(
                "callHierarchy/outgoingCalls",
                {
                    "item": {
                        "name": "autoencoder",
                        "kind": 12,
                        "uri": s.uri,
                        "range": {
                            "start": {"line": 21, "character": 0},
                            "end": {"line": 21, "character": 15},
                        },
                        "selectionRange": {
                            "start": {"line": 21, "character": 4},
                            "end": {"line": 21, "character": 15},
                        },
                        "data": {
                            "function_name": "autoencoder",
                            "class_name": None,
                            "uri": s.uri,
                        },
                    }
                },
            )
        resp = _find_response(s.messages, rid)
        assert resp is not None
        result = resp["result"]
        assert len(result) == 2
        names = {r["to"]["name"] for r in result}
        assert names == {"encode", "decode"}
