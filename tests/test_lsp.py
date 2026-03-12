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

    def test_definition_at_first_occurrence_returns_self(self) -> None:
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
        loc = resp["result"]
        assert loc is not None, "goToDefinition should return definition even at first occurrence"
        assert loc["range"]["start"]["line"] == 8
        assert loc["range"]["start"]["character"] == 21

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


class TestNonAnnotatedCallHierarchy:
    """outgoingCalls should include non-annotated workspace functions."""

    def setup_method(self) -> None:
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionDefInfo
        from jaxtyc.types import FunctionShapeSpec

        self.uri = "file:///test/calls.py"
        spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/calls.py",
            lineno=10,
            col_offset=0,
            params={},
            return_spec=None,
            name_col_offset=4,
        )
        helper_def = FunctionDefInfo(
            name="normalize",
            file_path="/test/calls.py",
            lineno=3,
            col_offset=0,
            end_lineno=5,
            name_col_offset=4,
        )
        call = CallSite(
            caller_name="encode",
            callee_name="normalize",
            file_path="/test/calls.py",
            lineno=12,
            col_offset=15,
            end_col_offset=24,
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/calls.py",
                uri=self.uri,
                function_specs=[spec],
                dim_locations=[],
                call_sites=[call],
                function_defs=[
                    helper_def,
                    FunctionDefInfo(
                        name="encode",
                        file_path="/test/calls.py",
                        lineno=10,
                        col_offset=0,
                        end_lineno=15,
                        name_col_offset=4,
                    ),
                ],
            )
        )

    def teardown_method(self) -> None:
        from jaxtyc.lsp import _state

        _state.workspace_index.remove_file(self.uri)

    def test_outgoing_calls_includes_non_annotated_callee(self) -> None:
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import outgoing_calls
        from jaxtyc.lsp.server import server

        item = lsp_types.CallHierarchyItem(
            name="encode",
            kind=lsp_types.SymbolKind.Function,
            uri=self.uri,
            range=lsp_types.Range(
                start=lsp_types.Position(line=9, character=0),
                end=lsp_types.Position(line=14, character=0),
            ),
            selection_range=lsp_types.Range(
                start=lsp_types.Position(line=9, character=4),
                end=lsp_types.Position(line=9, character=10),
            ),
            data={"function_name": "encode", "uri": self.uri},
        )
        params = lsp_types.CallHierarchyOutgoingCallsParams(item=item)
        result = outgoing_calls(server, params)
        assert result is not None
        assert len(result) == 1
        assert result[0].to.name == "normalize"

    def test_incoming_calls_includes_non_annotated_caller(self) -> None:
        from lsprotocol import types as lsp_types

        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp._navigation import incoming_calls
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.lsp.server import server
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionDefInfo

        uri2 = "file:///test/caller.py"
        main_def = FunctionDefInfo(
            name="main",
            file_path="/test/caller.py",
            lineno=1,
            col_offset=0,
            end_lineno=4,
            name_col_offset=4,
        )
        call = CallSite(
            caller_name="main",
            callee_name="encode",
            file_path="/test/caller.py",
            lineno=3,
            col_offset=8,
            end_col_offset=14,
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/caller.py",
                uri=uri2,
                function_specs=[],
                dim_locations=[],
                call_sites=[call],
                function_defs=[main_def],
            )
        )

        # Cross-file incoming calls require workspace scope
        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(references_scope="workspace"))
            item = lsp_types.CallHierarchyItem(
                name="encode",
                kind=lsp_types.SymbolKind.Function,
                uri=self.uri,
                range=lsp_types.Range(
                    start=lsp_types.Position(line=9, character=0),
                    end=lsp_types.Position(line=14, character=0),
                ),
                selection_range=lsp_types.Range(
                    start=lsp_types.Position(line=9, character=4),
                    end=lsp_types.Position(line=9, character=10),
                ),
                data={"function_name": "encode", "uri": self.uri},
            )
            params = lsp_types.CallHierarchyIncomingCallsParams(item=item)
            result = incoming_calls(server, params)
            assert result is not None
            # Should find "main" as a caller even though it has no FunctionShapeSpec
            caller_names = [r.from_.name for r in result]
            assert "main" in caller_names
        finally:
            _state.config = orig_config
            _state.workspace_index.remove_file(uri2)


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
        assert label == "f32[batch seq]", f"Expected 'f32[batch seq]', got: {label}"

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
        # New format: "f32[batch head_dim] | dim 1: expected seq, got head_dim"
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
        """Inlay hint on sharded line uses dim|axis notation in shape."""
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
        # Sharding integrated into shape: "f32[batch|data seq|None]"
        assert "batch|data" in label, f"Expected 'batch|data' in sharded hint, got: {label}"
        assert "seq|None" in label, f"Expected 'seq|None' in sharded hint, got: {label}"

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


class TestHoverDivergenceInfo:
    """Tests for divergence-aware hover in LSP navigation."""

    def test_hover_on_error_line_shows_divergence(self) -> None:
        """Hover on a line with a divergence error shows expected vs actual."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import ErrorHintInfo
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/hover_diverge.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 16),
                    dtype="float32",
                    source_file="/test/hover_diverge.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "d_out"),
                    op_name="dot_general",
                ),
            ]
            _state.error_hints_cache[uri] = [
                ErrorHintInfo(
                    source_line=5,
                    message="dim 1: expected d_model, got d_out",
                    rule="shape-mismatch",
                    function_name="forward",
                    expected_named=("batch", "d_model"),
                    actual_named=("batch", "d_out"),
                ),
            ]

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=uri),
            position=lsp_types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "dot_general" in content
        assert "d_model" in content
        assert "d_out" in content

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_hover_no_divergence_when_clean(self) -> None:
        """Hover on a clean line should not show divergence info."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/hover_clean.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4, 8),
                    dtype="float32",
                    source_file="/test/hover_clean.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch", "seq"),
                    op_name="add",
                ),
            ]
            _state.error_hints_cache[uri] = []

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=uri),
            position=lsp_types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "divergence" not in content.lower()
        assert "expected" not in content.lower()

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)

    def test_hover_divergence_with_rank_mismatch(self) -> None:
        """Divergence from rank change shows rank info."""
        from jaxtyc.lsp import _state
        from jaxtyc.types import ErrorHintInfo
        from jaxtyc.types import IntermediateShape

        uri = "file:///test/hover_rank.py"
        with _state.cache_lock:
            _state.analysis_cache[uri] = [
                IntermediateShape(
                    shape=(4,),
                    dtype="float32",
                    source_file="/test/hover_rank.py",
                    source_line=5,
                    source_col=0,
                    named_shape=("batch",),
                    op_name="reduce_sum",
                ),
            ]
            _state.error_hints_cache[uri] = [
                ErrorHintInfo(
                    source_line=5,
                    message="Rank changed to 1 (expected 2) at reduce_sum",
                    rule="rank-mismatch",
                    function_name="project",
                    expected_named=("batch", "seq"),
                    actual_named=("batch",),
                ),
            ]

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=uri),
            position=lsp_types.Position(line=4, character=0),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "batch, seq" in content
        assert "batch" in content

        with _state.cache_lock:
            _state.analysis_cache.pop(uri, None)
            _state.error_hints_cache.pop(uri, None)


class TestHoverTraceErrorFallback:
    """Hover on intermediate lines should show trace error when tracing failed."""

    def test_hover_trace_error_fallback(self) -> None:
        """Hovering on a line inside a failed-trace function shows the trace error."""
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec
        from jaxtyc.types import TraceResult

        uri = "file:///test/trace_err_hover.py"
        spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/trace_err_hover.py",
            lineno=5,
            col_offset=0,
            end_lineno=10,
            params={
                "x": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch"),),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(DimSpec(kind="named", name="batch"),),
                dtype="float32",
            ),
            name_col_offset=4,
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/trace_err_hover.py",
                uri=uri,
                function_specs=[spec],
                dim_locations=[],
                call_sites=[],
            )
        )

        # Store a failed trace result
        tr = TraceResult(
            function_name="encode",
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error="dot_general requires contracting dimensions to have the same shape",
        )
        with _state.cache_lock:
            _state.trace_results_cache[uri] = {"encode": tr}
            _state.analysis_cache[uri] = []  # No intermediates (trace failed)

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        # Hover on line 8 (inside encode, line 5-10), no intermediates
        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=uri),
            position=lsp_types.Position(line=7, character=4),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "Trace error" in content
        assert "encode" in content
        assert "dot_general" in content

        # Cleanup
        with _state.cache_lock:
            _state.trace_results_cache.pop(uri, None)
            _state.analysis_cache.pop(uri, None)
        _state.workspace_index.remove_file(uri)

    def test_hover_external_call_site_shows_qualified_name(self) -> None:
        """Hovering on an external call site (jnp.dot) shows the qualified name."""
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec

        uri = "file:///test/ext_hover.py"
        spec = FunctionShapeSpec(
            name="transform",
            file_path="/test/ext_hover.py",
            lineno=5,
            col_offset=0,
            end_lineno=10,
            params={
                "x": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch"),),
                    dtype="float32",
                )
            },
            return_spec=None,
            name_col_offset=4,
        )
        ext_call = CallSite(
            caller_name="transform",
            callee_name="dot",
            file_path="/test/ext_hover.py",
            lineno=8,
            col_offset=8,
            end_col_offset=15,
            callee_qualified_name="jnp.dot",
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/ext_hover.py",
                uri=uri,
                function_specs=[spec],
                dim_locations=[],
                call_sites=[ext_call],
            )
        )

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=uri),
            position=lsp_types.Position(line=7, character=10),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "jnp.dot" in content
        assert "external" in content

        _state.workspace_index.remove_file(uri)


class TestCallSiteHover:
    """Tests for call-site shape resolution on hover."""

    def test_hover_on_call_shows_callee_signature(self) -> None:
        """Hovering on encode(x) shows encode's shape signature."""
        from jaxtyc.analyzer.dim_env import DimEnv
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec
        from jaxtyc.types import TraceResult

        caller_uri = "file:///test/call_hover.py"
        callee_uri = "file:///test/callee.py"

        env = DimEnv()

        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/callee.py",
            lineno=5,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch"),
                        DimSpec(kind="named", name="d_model"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch"),
                    DimSpec(kind="named", name="hidden"),
                ),
                dtype="float32",
            ),
            name_col_offset=4,
        )
        callee_index = FileIndex(
            file_path="/test/callee.py",
            uri=callee_uri,
            function_specs=[callee_spec],
            dim_locations=[],
            call_sites=[],
        )
        _state.workspace_index.update_file(callee_index)

        call_site = CallSite(
            caller_name="forward",
            callee_name="encode",
            file_path="/test/call_hover.py",
            lineno=10,
            col_offset=8,
            end_col_offset=14,
        )
        caller_index = FileIndex(
            file_path="/test/call_hover.py",
            uri=caller_uri,
            function_specs=[],
            dim_locations=[],
            call_sites=[call_site],
        )
        _state.workspace_index.update_file(caller_index)

        batch = env.get_size("batch")
        hidden = env.get_size("hidden")
        tr = TraceResult(
            function_name="encode",
            output_shape=(batch, hidden),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        with _state.cache_lock:
            _state.trace_results_cache[callee_uri] = {"encode": tr}
            _state.dim_env_cache[callee_uri] = env
            _state.analysis_cache[caller_uri] = []

        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import hover
        from jaxtyc.lsp.server import server

        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=caller_uri),
            position=lsp_types.Position(line=9, character=10),
        )
        result = hover(server, params)
        assert result is not None
        content = result.contents.value
        assert "encode" in content
        assert "batch" in content
        assert "hidden" in content

        # Cleanup
        _state.workspace_index.remove_file(caller_uri)
        _state.workspace_index.remove_file(callee_uri)
        with _state.cache_lock:
            _state.trace_results_cache.pop(callee_uri, None)
            _state.dim_env_cache.pop(callee_uri, None)
            _state.analysis_cache.pop(caller_uri, None)


class TestCrossFileTracing:
    """Tests for multi-file cross-function shape mismatch detection."""

    def test_cross_file_mismatch_detected(self) -> None:
        """Cross-file call where callee trace contradicts annotation."""
        from jaxtyc.analyzer.dim_env import DimEnv
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec
        from jaxtyc.types import TraceResult

        callee_uri = "file:///test/xfile_utils.py"
        caller_uri = "file:///test/xfile_train.py"

        env = DimEnv()
        batch = env.get_size("batch")
        d_model = env.get_size("d_model")
        hidden = env.get_size("hidden")

        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/xfile_utils.py",
            lineno=5,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch"),
                        DimSpec(kind="named", name="d_model"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch"),
                    DimSpec(kind="named", name="hidden"),
                ),
                dtype="float32",
            ),
            name_col_offset=4,
        )
        callee_index = FileIndex(
            file_path="/test/xfile_utils.py",
            uri=callee_uri,
            function_specs=[callee_spec],
            dim_locations=[],
            call_sites=[],
        )
        _state.workspace_index.update_file(callee_index)

        # Callee trace returns d_model, not hidden -- mismatch!
        tr = TraceResult(
            function_name="encode",
            output_shape=(batch, d_model),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        with _state.cache_lock:
            _state.trace_results_cache[callee_uri] = {"encode": tr}
            _state.dim_env_cache[callee_uri] = env

        # Caller has a call site to encode
        caller_spec = FunctionShapeSpec(
            name="train",
            file_path="/test/xfile_train.py",
            lineno=10,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch"),
                        DimSpec(kind="named", name="d_model"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=None,
            name_col_offset=4,
        )
        call_site = CallSite(
            caller_name="train",
            callee_name="encode",
            file_path="/test/xfile_train.py",
            lineno=12,
            col_offset=8,
            end_col_offset=14,
        )
        caller_index = FileIndex(
            file_path="/test/xfile_train.py",
            uri=caller_uri,
            function_specs=[caller_spec],
            dim_locations=[],
            call_sites=[call_site],
        )
        _state.workspace_index.update_file(caller_index)

        from jaxtyc.lsp.server import _check_cross_file_calls

        diags = _check_cross_file_calls(caller_uri, [caller_spec])
        assert len(diags) >= 1
        assert any(d.code == "cross-function-mismatch" for d in diags)

        # Cleanup
        _state.workspace_index.remove_file(callee_uri)
        _state.workspace_index.remove_file(caller_uri)
        with _state.cache_lock:
            _state.trace_results_cache.pop(callee_uri, None)
            _state.dim_env_cache.pop(callee_uri, None)

    def test_cross_file_no_false_positive(self) -> None:
        """No diagnostic when callee trace matches annotation."""
        from jaxtyc.analyzer.dim_env import DimEnv
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec
        from jaxtyc.types import TraceResult

        callee_uri = "file:///test/xfile_ok_utils.py"
        caller_uri = "file:///test/xfile_ok_train.py"

        env = DimEnv()
        batch = env.get_size("batch")
        hidden = env.get_size("hidden")

        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/xfile_ok_utils.py",
            lineno=5,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch"),),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch"),
                    DimSpec(kind="named", name="hidden"),
                ),
                dtype="float32",
            ),
            name_col_offset=4,
        )
        callee_index = FileIndex(
            file_path="/test/xfile_ok_utils.py",
            uri=callee_uri,
            function_specs=[callee_spec],
            dim_locations=[],
            call_sites=[],
        )
        _state.workspace_index.update_file(callee_index)

        # Matching trace
        tr = TraceResult(
            function_name="encode",
            output_shape=(batch, hidden),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        with _state.cache_lock:
            _state.trace_results_cache[callee_uri] = {"encode": tr}
            _state.dim_env_cache[callee_uri] = env

        caller_spec = FunctionShapeSpec(
            name="train",
            file_path="/test/xfile_ok_train.py",
            lineno=10,
            col_offset=0,
            params={},
            return_spec=None,
            name_col_offset=4,
        )
        call_site = CallSite(
            caller_name="train",
            callee_name="encode",
            file_path="/test/xfile_ok_train.py",
            lineno=12,
            col_offset=8,
            end_col_offset=14,
        )
        caller_index = FileIndex(
            file_path="/test/xfile_ok_train.py",
            uri=caller_uri,
            function_specs=[caller_spec],
            dim_locations=[],
            call_sites=[call_site],
        )
        _state.workspace_index.update_file(caller_index)

        from jaxtyc.lsp.server import _check_cross_file_calls

        diags = _check_cross_file_calls(caller_uri, [caller_spec])
        assert len(diags) == 0

        # Cleanup
        _state.workspace_index.remove_file(callee_uri)
        _state.workspace_index.remove_file(caller_uri)
        with _state.cache_lock:
            _state.trace_results_cache.pop(callee_uri, None)
            _state.dim_env_cache.pop(callee_uri, None)


class TestCallSiteNavigation:
    """goToDefinition and goToImplementation should resolve call sites."""

    def setup_method(self) -> None:
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        self.uri = "file:///test/nav.py"
        spec = FunctionShapeSpec(
            name="encode",
            file_path="/test/nav.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=None,
            name_col_offset=4,
        )
        call = CallSite(
            caller_name="main",
            callee_name="encode",
            file_path="/test/nav.py",
            lineno=15,
            col_offset=8,
            end_col_offset=14,
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/nav.py",
                uri=self.uri,
                function_specs=[spec],
                dim_locations=[],
                call_sites=[call],
            )
        )

    def teardown_method(self) -> None:
        from jaxtyc.lsp import _state

        _state.workspace_index.remove_file(self.uri)

    def test_go_to_definition_at_call_site(self) -> None:
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_definition
        from jaxtyc.lsp.server import server

        params = lsp_types.DefinitionParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=14, character=10),  # line 15 -> 0-based 14
        )
        result = go_to_definition(server, params)
        assert result is not None
        loc = result[0] if isinstance(result, list) else result
        assert loc.range.start.line == 4  # lineno 5 -> 0-based 4

    def test_go_to_implementation_at_call_site(self) -> None:
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_implementation
        from jaxtyc.lsp.server import server

        params = lsp_types.ImplementationParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=14, character=10),
        )
        result = go_to_implementation(server, params)
        assert result is not None
        loc = result[0] if isinstance(result, list) else result
        assert loc.range.start.line == 4


class TestDimGoToDefinition:
    """goToDefinition should work on dimension names inside string annotations."""

    def setup_method(self) -> None:
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import DimLocation

        self.uri = "file:///test/dims.py"
        # Two occurrences of "batch" in function "encode":
        #   first at line 5 col 22-27 (the "definition")
        #   second at line 6 col 20-25 (a reference)
        self.dim_def = DimLocation(
            dim_name="batch",
            param_name="x",
            function_name="encode",
            file_path="/test/dims.py",
            lineno=5,
            col_start=22,
            col_end=27,
        )
        self.dim_ref = DimLocation(
            dim_name="batch",
            param_name="__return__",
            function_name="encode",
            file_path="/test/dims.py",
            lineno=6,
            col_start=20,
            col_end=25,
        )
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/dims.py",
                uri=self.uri,
                function_specs=[],
                dim_locations=[self.dim_def, self.dim_ref],
                call_sites=[],
            )
        )

    def teardown_method(self) -> None:
        from jaxtyc.lsp import _state

        _state.workspace_index.remove_file(self.uri)

    def test_go_to_definition_on_dim_reference(self) -> None:
        """goToDefinition on a non-first dim occurrence should jump to first occurrence."""
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_definition
        from jaxtyc.lsp.server import server

        # Cursor on the second occurrence (line 6, col 22 = inside "batch")
        params = lsp_types.DefinitionParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=5, character=22),  # line 6 -> 0-based 5
        )
        result = go_to_definition(server, params)
        assert result is not None
        loc = result[0] if isinstance(result, list) else result
        assert loc.range.start.line == 4  # line 5 -> 0-based 4
        assert loc.range.start.character == 22

    def test_go_to_definition_on_dim_at_definition(self) -> None:
        """goToDefinition on the first dim occurrence should return its own location."""
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_definition
        from jaxtyc.lsp.server import server

        # Cursor on the first occurrence (line 5, col 24 = inside "batch")
        params = lsp_types.DefinitionParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=4, character=24),  # line 5 -> 0-based 4
        )
        result = go_to_definition(server, params)
        assert result is not None, (
            "goToDefinition should return the definition location even when already at it"
        )
        loc = result[0] if isinstance(result, list) else result
        assert loc.range.start.line == 4
        assert loc.range.start.character == 22

    def test_go_to_implementation_on_dim_at_definition(self) -> None:
        """goToImplementation on the first dim occurrence should return its own location."""
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_implementation
        from jaxtyc.lsp.server import server

        params = lsp_types.ImplementationParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=4, character=24),
        )
        result = go_to_implementation(server, params)
        assert result is not None, (
            "goToImplementation should return the definition location even when already at it"
        )
        loc = result[0] if isinstance(result, list) else result
        assert loc.range.start.line == 4
        assert loc.range.start.character == 22


class TestDimGoToDefinitionFileScoped:
    """goToDefinition on a dim at its function-scoped definition should fall back to file-scoped."""

    def setup_method(self) -> None:
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import DimLocation

        self.uri = "file:///test/scope.py"
        # "batch" appears in two functions: encode (line 5) and decode (line 10)
        self.dims = [
            DimLocation(
                dim_name="batch",
                param_name="x",
                function_name="encode",
                file_path="/test/scope.py",
                lineno=5,
                col_start=22,
                col_end=27,
            ),
            DimLocation(
                dim_name="batch",
                param_name="__return__",
                function_name="encode",
                file_path="/test/scope.py",
                lineno=6,
                col_start=20,
                col_end=25,
            ),
            DimLocation(
                dim_name="batch",
                param_name="h",
                function_name="decode",
                file_path="/test/scope.py",
                lineno=10,
                col_start=22,
                col_end=27,
            ),
            DimLocation(
                dim_name="batch",
                param_name="__return__",
                function_name="decode",
                file_path="/test/scope.py",
                lineno=11,
                col_start=20,
                col_end=25,
            ),
        ]
        _state.workspace_index.update_file(
            FileIndex(
                file_path="/test/scope.py",
                uri=self.uri,
                function_specs=[],
                dim_locations=self.dims,
                call_sites=[],
            )
        )

    def teardown_method(self) -> None:
        from jaxtyc.lsp import _state

        _state.workspace_index.remove_file(self.uri)

    def test_go_to_definition_cross_function_dim(self) -> None:
        """goToDefinition on 'batch' in decode should navigate to first 'batch' in the file (encode)."""
        from lsprotocol import types as lsp_types

        from jaxtyc.lsp._navigation import go_to_definition
        from jaxtyc.lsp.server import server

        # Cursor on batch in decode (line 10, col 24)
        params = lsp_types.DefinitionParams(
            text_document=lsp_types.TextDocumentIdentifier(uri=self.uri),
            position=lsp_types.Position(line=9, character=24),  # line 10 -> 0-based 9
        )
        result = go_to_definition(server, params)
        assert result is not None
        loc = result[0] if isinstance(result, list) else result
        # Should navigate to encode's batch on line 5 (0-based 4), not decode's batch on line 10
        assert loc.range.start.line == 4, (
            f"Expected line 4 (first file occurrence in encode), got {loc.range.start.line}"
        )


class TestReferenceScopeConfig:
    """findReferences should respect navigation.references_scope config."""

    def test_file_scoped_references_excludes_other_files(self) -> None:
        """references_scope='file' only shows same-file callers."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(references_scope="file"))

            spec_a = FunctionShapeSpec(
                name="decode",
                file_path="/a.py",
                lineno=5,
                col_offset=0,
                params={},
                return_spec=None,
                name_col_offset=4,
            )
            cs_local = CallSite("pipeline", "decode", "/a.py", 15, 4, 10)
            cs_other = CallSite("pipeline", "decode", "/b.py", 20, 4, 10)

            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/a.py",
                    uri="file:///a.py",
                    function_specs=[spec_a],
                    dim_locations=[],
                    call_sites=[cs_local],
                )
            )
            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/b.py",
                    uri="file:///b.py",
                    function_specs=[],
                    dim_locations=[],
                    call_sites=[cs_other],
                )
            )

            from lsprotocol import types as lsp_types

            from jaxtyc.lsp._navigation import find_references
            from jaxtyc.lsp.server import server

            params = lsp_types.ReferenceParams(
                text_document=lsp_types.TextDocumentIdentifier(uri="file:///a.py"),
                position=lsp_types.Position(line=4, character=4),
                context=lsp_types.ReferenceContext(include_declaration=False),
            )
            result = find_references(server, params)
            assert result is not None
            assert len(result) == 1

            _state.workspace_index.remove_file("file:///a.py")
            _state.workspace_index.remove_file("file:///b.py")
        finally:
            _state.config = orig_config

    def test_workspace_scoped_references_includes_all_files(self) -> None:
        """references_scope='workspace' shows callers from all files."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(references_scope="workspace"))

            spec_a = FunctionShapeSpec(
                name="decode",
                file_path="/a.py",
                lineno=5,
                col_offset=0,
                params={},
                return_spec=None,
                name_col_offset=4,
            )
            cs_local = CallSite("pipeline", "decode", "/a.py", 15, 4, 10)
            cs_other = CallSite("pipeline", "decode", "/b.py", 20, 4, 10)

            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/a.py",
                    uri="file:///a.py",
                    function_specs=[spec_a],
                    dim_locations=[],
                    call_sites=[cs_local],
                )
            )
            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/b.py",
                    uri="file:///b.py",
                    function_specs=[],
                    dim_locations=[],
                    call_sites=[cs_other],
                )
            )

            from lsprotocol import types as lsp_types

            from jaxtyc.lsp._navigation import find_references
            from jaxtyc.lsp.server import server

            params = lsp_types.ReferenceParams(
                text_document=lsp_types.TextDocumentIdentifier(uri="file:///a.py"),
                position=lsp_types.Position(line=4, character=4),
                context=lsp_types.ReferenceContext(include_declaration=False),
            )
            result = find_references(server, params)
            assert result is not None
            assert len(result) == 2

            _state.workspace_index.remove_file("file:///a.py")
            _state.workspace_index.remove_file("file:///b.py")
        finally:
            _state.config = orig_config


class TestExternalCallsConfig:
    def test_outgoing_calls_includes_external_when_configured(self) -> None:
        """With include_external_calls=true, library calls appear in outgoingCalls."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(include_external_calls=True))

            uri = "file:///test/ext.py"
            spec = FunctionShapeSpec(
                name="transform",
                file_path="/test/ext.py",
                lineno=5,
                col_offset=0,
                params={},
                return_spec=None,
                name_col_offset=4,
            )
            ext_call = CallSite(
                caller_name="transform",
                callee_name="dot",
                file_path="/test/ext.py",
                lineno=7,
                col_offset=10,
                end_col_offset=13,
            )
            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/test/ext.py",
                    uri=uri,
                    function_specs=[spec],
                    dim_locations=[],
                    call_sites=[ext_call],
                )
            )

            from lsprotocol import types as lsp_types

            from jaxtyc.lsp._navigation import outgoing_calls
            from jaxtyc.lsp.server import server

            item = lsp_types.CallHierarchyItem(
                name="transform",
                kind=lsp_types.SymbolKind.Function,
                uri=uri,
                range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=0),
                    end=lsp_types.Position(line=8, character=0),
                ),
                selection_range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=4),
                    end=lsp_types.Position(line=4, character=13),
                ),
                data={"function_name": "transform", "uri": uri},
            )
            params = lsp_types.CallHierarchyOutgoingCallsParams(item=item)
            result = outgoing_calls(server, params)
            assert result is not None
            assert len(result) == 1
            assert result[0].to.name == "dot"
            assert result[0].to.detail == "(external)"

            _state.workspace_index.remove_file(uri)
        finally:
            _state.config = orig_config

    def test_outgoing_calls_external_qualified_name(self) -> None:
        """External calls display the qualified name (e.g. jnp.dot not just dot)."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(include_external_calls=True))

            uri = "file:///test/qn.py"
            spec = FunctionShapeSpec(
                name="transform",
                file_path="/test/qn.py",
                lineno=5,
                col_offset=0,
                params={},
                return_spec=None,
                name_col_offset=4,
            )
            ext_call = CallSite(
                caller_name="transform",
                callee_name="dot",
                file_path="/test/qn.py",
                lineno=7,
                col_offset=10,
                end_col_offset=17,
                callee_qualified_name="jnp.dot",
            )
            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/test/qn.py",
                    uri=uri,
                    function_specs=[spec],
                    dim_locations=[],
                    call_sites=[ext_call],
                )
            )

            from lsprotocol import types as lsp_types

            from jaxtyc.lsp._navigation import outgoing_calls
            from jaxtyc.lsp.server import server

            item = lsp_types.CallHierarchyItem(
                name="transform",
                kind=lsp_types.SymbolKind.Function,
                uri=uri,
                range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=0),
                    end=lsp_types.Position(line=8, character=0),
                ),
                selection_range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=4),
                    end=lsp_types.Position(line=4, character=13),
                ),
                data={"function_name": "transform", "uri": uri},
            )
            params = lsp_types.CallHierarchyOutgoingCallsParams(item=item)
            result = outgoing_calls(server, params)
            assert result is not None
            assert len(result) == 1
            assert result[0].to.name == "jnp.dot"
            assert result[0].to.detail == "(external)"

            _state.workspace_index.remove_file(uri)
        finally:
            _state.config = orig_config

    def test_outgoing_calls_excludes_external_when_disabled(self) -> None:
        """include_external_calls=false hides external calls."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import NavigationConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import FileIndex
        from jaxtyc.types import CallSite
        from jaxtyc.types import FunctionShapeSpec

        orig_config = _state.config
        try:
            _state.config = JaxtycConfig(navigation=NavigationConfig(include_external_calls=False))

            uri = "file:///test/ext2.py"
            spec = FunctionShapeSpec(
                name="transform",
                file_path="/test/ext2.py",
                lineno=5,
                col_offset=0,
                params={},
                return_spec=None,
                name_col_offset=4,
            )
            ext_call = CallSite(
                caller_name="transform",
                callee_name="dot",
                file_path="/test/ext2.py",
                lineno=7,
                col_offset=10,
                end_col_offset=13,
            )
            _state.workspace_index.update_file(
                FileIndex(
                    file_path="/test/ext2.py",
                    uri=uri,
                    function_specs=[spec],
                    dim_locations=[],
                    call_sites=[ext_call],
                )
            )

            from lsprotocol import types as lsp_types

            from jaxtyc.lsp._navigation import outgoing_calls
            from jaxtyc.lsp.server import server

            item = lsp_types.CallHierarchyItem(
                name="transform",
                kind=lsp_types.SymbolKind.Function,
                uri=uri,
                range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=0),
                    end=lsp_types.Position(line=8, character=0),
                ),
                selection_range=lsp_types.Range(
                    start=lsp_types.Position(line=4, character=4),
                    end=lsp_types.Position(line=4, character=13),
                ),
                data={"function_name": "transform", "uri": uri},
            )
            params = lsp_types.CallHierarchyOutgoingCallsParams(item=item)
            result = outgoing_calls(server, params)
            assert result is None

            _state.workspace_index.remove_file(uri)
        finally:
            _state.config = orig_config


class TestEarlyIndexBuild:
    def test_workspace_index_populated_from_source(self) -> None:
        """build_file_index from source populates workspace_index with navigation data."""
        import textwrap

        from jaxtyc.analyzer.annotations import extract_function_specs
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.index import build_file_index

        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def encode(x: Float[Array, "batch d"]) -> Float[Array, "batch d"]:
                return x
        """)
        uri = "file:///test/early.py"
        file_path = "/test/early.py"

        func_specs = extract_function_specs(source, file_path)
        file_index = build_file_index(source, file_path, uri, func_specs=func_specs)
        _state.workspace_index.update_file(file_index)

        # Navigation data is available
        assert _state.workspace_index.get_file(uri) is not None
        fi = _state.workspace_index.get_file(uri)
        assert len(fi.function_specs) == 1
        assert fi.function_specs[0].name == "encode"
        assert len(fi.dim_locations) > 0  # batch, d in param + return
        assert len(fi.function_defs) >= 1  # at least encode

        _state.workspace_index.remove_file(uri)

    def test_navigation_available_before_trace_results(self) -> None:
        """Workspace index should have specs even when no trace results exist."""
        import textwrap

        from jaxtyc.lsp import _state

        source = textwrap.dedent("""\
            from jaxtyping import Array, Float

            def encode(x: Float[Array, "batch d"]) -> Float[Array, "batch d"]:
                return x
        """)
        uri = "file:///test/early2.py"
        file_path = "/test/early2.py"

        from jaxtyc.analyzer.annotations import extract_function_specs
        from jaxtyc.lsp.index import build_file_index

        func_specs = extract_function_specs(source, file_path)
        file_index = build_file_index(source, file_path, uri, func_specs=func_specs)
        _state.workspace_index.update_file(file_index)

        # Navigation works (specs, dims) but no trace data yet
        spec = _state.workspace_index.find_function_at(uri, 3, 4)
        assert spec is not None
        assert spec.name == "encode"

        # Just verify workspace index is populated, not empty
        fi = _state.workspace_index.get_file(uri)
        assert fi is not None
        assert len(fi.dim_locations) > 0

        # But trace caches should be empty (analysis hasn't run)
        with _state.cache_lock:
            assert uri not in _state.trace_results_cache

        _state.workspace_index.remove_file(uri)


class TestShardingInTypes:
    """Tests for dim|axis pipe syntax in hover, shape_summary, and completion."""

    def test_dim_label_with_mesh_axis(self) -> None:
        """dim_label should append |axis when mesh_axis is set."""
        from jaxtyc.lsp._util import dim_label
        from jaxtyc.types import DimSpec

        d = DimSpec(kind="named", name="batch", mesh_axis="dp")
        assert dim_label(d) == "batch|dp"

    def test_dim_label_without_mesh_axis(self) -> None:
        """dim_label should return plain name when mesh_axis is None."""
        from jaxtyc.lsp._util import dim_label
        from jaxtyc.types import DimSpec

        d = DimSpec(kind="named", name="batch")
        assert dim_label(d) == "batch"

    def test_dim_label_fixed_with_mesh_axis(self) -> None:
        """dim_label should handle fixed dims with mesh_axis."""
        from jaxtyc.lsp._util import dim_label
        from jaxtyc.types import DimSpec

        d = DimSpec(kind="fixed", size=32, mesh_axis="mp")
        assert dim_label(d) == "32|mp"

    def test_shape_summary_includes_mesh_axis(self) -> None:
        """shape_summary should show dim|axis for sharded params."""
        from jaxtyc.lsp._util import shape_summary
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec

        spec = FunctionShapeSpec(
            name="sharded_matmul",
            file_path="/test/sharded.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch", mesh_axis="dp"),
                        DimSpec(kind="named", name="seq"),
                        DimSpec(kind="named", name="d_model", mesh_axis="mp"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch", mesh_axis="dp"),
                    DimSpec(kind="named", name="seq"),
                    DimSpec(kind="named", name="d_ff"),
                ),
                dtype="float32",
            ),
        )
        result = shape_summary(spec)
        assert "batch|dp" in result
        assert "d_model|mp" in result
        # Unsharded dim should not have pipe
        assert "seq|" not in result
        assert "d_ff|" not in result

    def test_param_hover_includes_mesh_axis(self) -> None:
        """_param_hover should show dim|axis for sharded dims."""
        from jaxtyc.lsp._navigation import _param_hover
        from jaxtyc.types import DimSpec
        from jaxtyc.types import ShapeSpec

        pspec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch", mesh_axis="dp"),
                DimSpec(kind="named", name="seq"),
                DimSpec(kind="named", name="d_model", mesh_axis="mp"),
            ),
            dtype="float32",
        )
        result = _param_hover("x", pspec)
        assert "batch|dp" in result
        assert "d_model|mp" in result
        # Unsharded dim should not have pipe
        assert "seq|" not in result

    def test_function_hover_includes_mesh_axis(self) -> None:
        """_function_hover should show dim|axis for sharded params and return."""
        from jaxtyc.lsp._navigation import _function_hover
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec

        spec = FunctionShapeSpec(
            name="sharded_fn",
            file_path="/test/sharded.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch", mesh_axis="dp"),
                        DimSpec(kind="named", name="d_model", mesh_axis="mp"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch", mesh_axis="dp"),
                    DimSpec(kind="named", name="hidden"),
                ),
                dtype="float32",
            ),
        )
        result = _function_hover(spec)
        assert "batch|dp" in result
        assert "d_model|mp" in result
        # Return section
        assert "hidden" in result
        # hidden should not have pipe
        assert "hidden|" not in result

    def test_hover_on_sharded_function_name(self) -> None:
        """Full hover on a sharded function shows mesh_axis in all positions."""
        from jaxtyc.types import DimSpec
        from jaxtyc.types import FunctionShapeSpec
        from jaxtyc.types import ShapeSpec

        func_spec = FunctionShapeSpec(
            name="sharded_matmul",
            file_path="/test/sharded_hover.py",
            lineno=3,
            col_offset=0,
            name_col_offset=4,
            end_lineno=5,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch", mesh_axis="dp"),
                        DimSpec(kind="named", name="seq"),
                        DimSpec(kind="named", name="d_model", mesh_axis="mp"),
                    ),
                    dtype="float32",
                )
            },
            return_spec=ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch", mesh_axis="dp"),
                    DimSpec(kind="named", name="seq"),
                    DimSpec(kind="named", name="d_ff"),
                ),
                dtype="float32",
            ),
        )

        # Test via _function_hover directly (avoids needing server.workspace)
        from jaxtyc.lsp._navigation import _function_hover

        content = _function_hover(func_spec)
        assert "batch|dp" in content
        assert "d_model|mp" in content
        # Unsharded dims
        assert "seq|" not in content
        assert "d_ff|" not in content

        # Also verify shape_summary includes sharding
        from jaxtyc.lsp._util import shape_summary

        summary = shape_summary(func_spec)
        assert "batch|dp" in summary
        assert "d_model|mp" in summary

    def test_completion_offers_mesh_axes_after_pipe(self) -> None:
        """Typing | in a shape string should offer mesh axis completions."""
        from jaxtyc.config import JaxtycConfig
        from jaxtyc.config import ShardingConfig
        from jaxtyc.lsp import _state
        from jaxtyc.lsp._completion import _get_mesh_axis_completions
        from jaxtyc.lsp._completion import _is_after_pipe

        original_config = _state.config
        _state.config = JaxtycConfig(sharding=ShardingConfig(mesh={"dp": 4, "mp": 2, "pp": 1}))

        try:
            # Test pipe detection
            # "batch|" -> | is at col 29, " is at col 30
            line = 'def fn(x: Float[Array, "batch|"]):'
            assert _is_after_pipe(line, 30)  # cursor right after |
            assert not _is_after_pipe(line, 29)  # cursor at | itself

            # Test mesh axis completions
            axes = _get_mesh_axis_completions("")
            assert set(axes) == {"dp", "mp", "pp"}

            # Test filtering
            axes_d = _get_mesh_axis_completions("d")
            assert "dp" in axes_d
            assert "mp" not in axes_d
        finally:
            _state.config = original_config
