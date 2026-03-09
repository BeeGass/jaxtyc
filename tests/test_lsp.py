"""Tests for jaxtyc.lsp.server — LSP integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
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
