"""Tests for jaxtyc.lsp.server — LSP integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
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
