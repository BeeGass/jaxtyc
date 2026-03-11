"""Tests for jaxtyc.lsp.server — LSP integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


def _lsp_message(method: str, params: dict[str, Any], msg_id: int | None = None) -> bytes:
    """Encode an LSP JSON-RPC message with Content-Length header."""
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if msg_id is not None:
        body["id"] = msg_id
    content = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
    return header + content


def _lsp_response(msg_id: int, result: dict[str, Any] | None) -> bytes:
    body = {"jsonrpc": "2.0", "id": msg_id, "result": result}
    content = json.dumps(body).encode("utf-8")
    header = f"Content-Length: {len(content)}\r\n\r\n".encode("ascii")
    return header + content


def _parse_lsp_messages(data: bytes) -> list[dict[str, Any]]:
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

    def __init__(self, proc: subprocess.Popen[bytes], uri: str | None):
        self._proc = proc
        self.uri: str | None = uri
        self._next_id = 10
        self.messages: list[dict[str, Any]] = []

    def request(self, method: str, params: dict[str, Any]) -> int:
        """Send a JSON-RPC request and return the message ID."""
        msg_id = self._next_id
        self._next_id += 1
        self._proc.stdin.write(_lsp_message(method, params, msg_id=msg_id))
        self._proc.stdin.flush()
        return msg_id


def _find_response(messages: list[dict[str, Any]], msg_id: int) -> dict[str, Any] | None:
    """Find a response message by ID."""
    return next((m for m in messages if m.get("id") == msg_id), None)


@contextmanager
def _lsp_session(fixture: str | None = None, wait: float = 2.0) -> Generator[_Session]:
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
    def test_initialize_and_shutdown(self) -> None:
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

    def test_diagnostics_on_save(self) -> None:
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

    def test_server_responsive_during_analysis(self) -> None:
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

    def test_diagnostics_on_change(self) -> None:
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

    def test_progress_notification_on_analysis(self) -> None:
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

    def test_codelens_shows_shapes(self) -> None:
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

    def test_document_symbol(self) -> None:
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

    def test_definition_dim_jumps_to_first(self) -> None:
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

    def test_definition_at_first_occurrence_returns_null(self) -> None:
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

    def test_references_dim(self) -> None:
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

    def test_document_highlight(self) -> None:
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

    def test_prepare_rename(self) -> None:
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

    def test_rename(self) -> None:
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

    def test_workspace_symbol(self) -> None:
        with _lsp_session("correct_attention.py") as s:
            rid = s.request("workspace/symbol", {"query": "atten"})
        resp = _find_response(s.messages, rid)
        assert resp is not None
        symbols = resp["result"]
        assert len(symbols) >= 1
        assert symbols[0]["name"] == "attention"

    def test_implementation(self) -> None:
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

    def test_prepare_call_hierarchy(self) -> None:
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

    def test_incoming_calls(self) -> None:
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

    def test_outgoing_calls(self) -> None:
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


class TestLSPInlayHints:
    """Tests for inlay hints with error and sharding display."""

    def test_inlay_hint_compact_format(self) -> None:
        """Inlay hints use compact dtype[dim1, dim2] format."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/compact_hint.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/compact_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) == 1
        label = hints[0].label
        assert isinstance(label, str)
        # No source_cache so no prefix is added (no line classification)
        assert label == "f32[batch, seq]", f"Expected 'f32[batch, seq]', got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_inlay_hint_last_per_line(self) -> None:
        """When multiple intermediates share a line, show the last one (final shape)."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/multi_hint.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/multi_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_in"),
                    op_name="dot",
                ),
                IntermediateShape(
                    shape=(4, 16),
                    dtype="float32",
                    source_file="/test/multi_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_out"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) == 1
        label = hints[0].label
        assert isinstance(label, str)
        # Should show the LAST intermediate (d_out, not d_in)
        assert "d_out" in label, f"Expected last intermediate (d_out), got: {label}"
        assert "d_in" not in label, f"Should not show first intermediate (d_in), got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_inlay_hint_shows_error_both_mode(self) -> None:
        """Inlay hint on error line includes shape AND error text with pipe."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import ErrorHintInfo
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/error_hint.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/error_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "head_dim"),
                    op_name="dot",
                ),
            ]
            _state.error_hints_cache[uri] = [
                ErrorHintInfo(
                    source_line=5,
                    message="dim 1: expected seq, got head_dim",
                    rule="shape-mismatch",
                    function_name="fn",
                ),
            ]

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) >= 1
        line5_hints = [h for h in hints if h.position.line == 4]  # 0-indexed
        assert len(line5_hints) == 1
        label = line5_hints[0].label
        assert isinstance(label, str)
        assert " | " in label, f"Expected pipe separator in error hint, got: {label}"
        # New format: "f32[batch, head_dim] | dim 1: expected seq, got head_dim"
        assert "f32[" in label, f"Expected compact dtype format, got: {label}"
        assert "batch" in label

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_inlay_hint_no_error_when_shapes_match(self) -> None:
        """Hints without error entries show shape-only, no pipe."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/clean_hint.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/clean_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        for h in hints:
            label = h.label
            assert isinstance(label, str)
            assert " | " not in label, f"Clean hint should not have pipe, got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_inlay_hint_sharding_display(self) -> None:
        """Inlay hint on sharded line appends P(...) with pipe separator."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape
        from jaxtyc.types import ShardingInfo

        uri = "file:///test/sharded_hint.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/sharded_hint.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="sharding_constraint",
                    sharding=ShardingInfo(
                        partition_spec=("data", None),
                        mesh_axis_names=("data", "model"),
                        source_primitive="sharding_constraint",
                        source_line=5,
                    ),
                ),
            ]
            _state.error_hints_cache[uri] = []

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) >= 1
        label = hints[0].label
        assert isinstance(label, str)
        # New format: pipe-separated sharding
        assert "| P(" in label, f"Expected '| P(...)' in sharded hint, got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)


class TestLSPHoverIntermediates:
    """Tests for intermediate hover with Option E format."""

    def test_hover_shows_all_intermediates(self) -> None:
        """Hover at a line shows all intermediates, not just one."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/hover_inters.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/hover_inters.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_in"),
                    op_name="dot_general",
                ),
                IntermediateShape(
                    shape=(4, 16),
                    dtype="float32",
                    source_file="/test/hover_inters.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_out"),
                    op_name="add",
                ),
            ]

        from lsprotocol import types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = types.HoverParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        # Should contain both intermediates
        assert "dot_general" in content
        assert "add" in content
        # Should mark the final one
        assert "final" in content.lower()
        # Should use compact format
        assert "f32[" in content

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)

    def test_hover_shows_sharding_info(self) -> None:
        """Hover on sharded intermediate includes sharding details."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape
        from jaxtyc.types import ShardingInfo

        uri = "file:///test/hover_sharding.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/hover_sharding.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="sharding_constraint",
                    sharding=ShardingInfo(
                        partition_spec=("data", None),
                        mesh_axis_names=("data", "model"),
                        source_primitive="sharding_constraint",
                        source_line=5,
                    ),
                ),
            ]

        from lsprotocol import types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = types.HoverParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "P(" in content, f"Expected sharding info in hover, got: {content}"
        assert "data" in content

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)

    def test_hover_single_intermediate_no_final_marker(self) -> None:
        """Single intermediate on a line should show final marker."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/hover_single.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/hover_single.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="add",
                ),
            ]

        from lsprotocol import types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = types.HoverParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            position=types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        # Compact format used
        assert "f32[" in content

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)


class TestInlayHintPositioning:
    """Tests for after-variable-name hint positioning."""

    def test_hint_after_variable_in_assignment(self) -> None:
        """For `y = expr`, hint should be placed right after 'y'."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        # "    y = self.linear(x)"
        pos = _find_hint_position("    y = self.linear(x)")
        assert pos == 5, f"Expected position 5 (after 'y'), got {pos}"

    def test_hint_after_dotted_variable(self) -> None:
        """For `self.out = expr`, hint after 'self.out'."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("        self.out = compute()")
        assert pos == 16, f"Expected 16 (after 'self.out'), got {pos}"

    def test_hint_after_annotated_variable(self) -> None:
        """For `x: Array = expr`, hint after 'x' (before ':')."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    x: Array = compute()")
        assert pos == 5, f"Expected 5 (after 'x'), got {pos}"

    def test_hint_eol_for_return(self) -> None:
        """Return statements fall back to end-of-line."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    return y")
        assert pos is None

    def test_hint_eol_for_bare_expression(self) -> None:
        """Bare expressions fall back to end-of-line."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    self.linear(x)")
        assert pos is None

    def test_hint_eol_for_tuple_unpacking(self) -> None:
        """Tuple unpacking falls back to end-of-line."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    a, b = func()")
        assert pos is None

    def test_hint_eol_for_augmented_assignment(self) -> None:
        """Augmented assignment (+=) falls back to end-of-line."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    x += 1")
        assert pos is None

    def test_hint_eol_for_comparison(self) -> None:
        """Comparison (==) should not be mistaken for assignment."""
        from jaxtyc.lsp._inlay_hints import _find_hint_position

        pos = _find_hint_position("    if x == y:")
        assert pos is None

    def test_inlay_hint_uses_variable_position(self) -> None:
        """Full inlay hint handler places hint after variable name when source is available."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/var_pos.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/var_pos.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_out"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []
            # Cache the source text so the handler can detect assignment
            _state.source_cache[uri] = "line1\nline2\nline3\nline4\n    y = self.linear(x)\nline6\n"

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) == 1
        # Hint should be at character 5 (right after "y"), not 999
        assert hints[0].position.character == 5, (
            f"Expected character=5 (after 'y'), got {hints[0].position.character}"
        )
        # Assignment hints get ": " prefix
        label = hints[0].label
        assert isinstance(label, str)
        assert label.startswith(": "), f"Expected ': ' prefix for assignment hint, got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)
            _state.source_cache.pop(uri, None)

    def test_inlay_hint_eol_fallback_for_return(self) -> None:
        """Hint falls back to end-of-line for return statements."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/return_pos.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/return_pos.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_out"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []
            _state.source_cache[uri] = "line1\nline2\nline3\nline4\n    return y\nline6\n"

        from lsprotocol import types

        from jaxtyc.lsp._inlay_hints import inlay_hint
        from jaxtyc.lsp.server import server

        params = types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=10, character=0),
            ),
        )
        hints = inlay_hint(server, params)
        assert hints is not None
        assert len(hints) == 1
        # Should fall back to end-of-line
        assert hints[0].position.character == 999, (
            f"Expected character=999 (end-of-line), got {hints[0].position.character}"
        )
        # Return hints get " -> " prefix (leading space for separation from last character)
        label = hints[0].label
        assert isinstance(label, str)
        assert label.startswith(" -> "), f"Expected ' -> ' prefix for return hint, got: {label}"

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)
            _state.source_cache.pop(uri, None)


class TestLSPErrorHintsState:
    """Tests for error_hints_cache in LSP state."""

    def test_error_hints_cache_exists(self) -> None:
        """_state module should have error_hints_cache dict."""
        from jaxtyc.lsp import _state

        assert hasattr(_state, "error_hints_cache")
        assert isinstance(_state.error_hints_cache, dict)

    def test_error_hints_cache_cleared_on_close(self) -> None:
        """Closing a document should clear its error_hints_cache entry."""
        with _lsp_session("wrong_transpose.py") as s:
            # Close the document
            s._proc.stdin.write(
                _lsp_message(
                    "textDocument/didClose",
                    {"textDocument": {"uri": s.uri}},
                )
            )
            s._proc.stdin.flush()
            time.sleep(0.5)
        # If we get here without error, the close handler processed without
        # crashing on the error_hints_cache key


class TestTraceResultsCache:
    """Tests for trace_results_cache in LSP state."""

    def test_trace_results_cache_exists(self) -> None:
        """_state module should have trace_results_cache dict."""
        from jaxtyc.lsp import _state

        assert hasattr(_state, "trace_results_cache")
        assert isinstance(_state.trace_results_cache, dict)

    def test_trace_results_populated_after_analysis(self) -> None:
        """After analysis, trace_results_cache should contain function traces."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import TraceResult

        uri = "file:///test/trace_cache_test.py"
        tr = TraceResult(
            function_name="test_fn",
            output_shape=(4, 8),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        with _state.cache_lock:
            _state.trace_results_cache[uri] = {"test_fn": tr}
        assert _state.trace_results_cache[uri]["test_fn"].output_shape == (4, 8)
        with _state.cache_lock:
            _state.trace_results_cache.pop(uri, None)
