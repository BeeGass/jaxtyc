"""LSP multiplexer: merges a primary type checker + jaxtyc shape checking behind one stdio pipe.

For dual-eligible methods, requests are sent to BOTH servers and results are merged:
  - Array methods (codeLens, codeAction, references, etc.): concatenate arrays
  - Hover: combine markdown from both servers
  - Completion: merge CompletionList items
  - Single-value (definition, rename, etc.): first non-null, prefer primary
Notifications broadcast to both. publishDiagnostics merged per-URI.

The primary server is discovered in order: pyright-langserver (ty shim), ty, pyright.
jaxtyc is started via ``sys.executable -m jaxtyc.cli.main lsp``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any  # noqa: F401 (used in annotations, invisible with __future__)
from urllib.parse import unquote
from urllib.parse import urlparse

DUAL_METHODS = frozenset(
    {
        "textDocument/hover",
        "textDocument/definition",
        "textDocument/references",
        "textDocument/documentHighlight",
        "textDocument/codeLens",
        "textDocument/codeAction",
        "textDocument/completion",
        "textDocument/signatureHelp",
        "textDocument/semanticTokens/full",
        "textDocument/inlayHint",
        "textDocument/linkedEditingRange",
        "textDocument/foldingRange",
        "textDocument/prepareRename",
        "textDocument/rename",
        "textDocument/implementation",
        "textDocument/prepareCallHierarchy",
        "callHierarchy/incomingCalls",
        "callHierarchy/outgoingCalls",
        "workspace/symbol",
    }
)

ARRAY_MERGE = frozenset(
    {
        "textDocument/codeLens",
        "textDocument/codeAction",
        "textDocument/references",
        "textDocument/documentHighlight",
        "textDocument/foldingRange",
        "textDocument/inlayHint",
        "workspace/symbol",
        "callHierarchy/incomingCalls",
        "callHierarchy/outgoingCalls",
    }
)

MERGE_TIMEOUT = 3.0


# Max characters for hover text before truncation
_HOVER_MAX_CHARS = 1500


def _hover_compact_enabled() -> bool:
    """Check if hover compaction is enabled.

    Controlled by JAXTYC_HOVER_COMPACT env var (default: true).
    Set to "0" or "false" to disable.
    """
    val = os.environ.get("JAXTYC_HOVER_COMPACT", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _clean_hover_text(text: str) -> str:
    """Compact hover markdown to reduce context consumption.

    When hover_compact is disabled (JAXTYC_HOVER_COMPACT=0), only performs
    minimal cleanup. When enabled (default), also unescapes markdown,
    strips trailing whitespace, collapses blank lines, and truncates.
    """
    # Always replace &nbsp; — these are never useful as raw text
    text = text.replace("&nbsp;", " ")

    if not _hover_compact_enabled():
        return text

    text = text.replace("\\_", "_")
    text = re.sub(r"  +$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > _HOVER_MAX_CHARS:
        text = text[:_HOVER_MAX_CHARS] + "\n\n*...(truncated)*"
    return text


def _merge_hover(primary_result: dict | None, jaxtyc_result: dict | None) -> dict | None:
    if primary_result is None and jaxtyc_result is None:
        return None

    def _extract_value(r: dict | None) -> str:
        if r is None:
            return ""
        contents = r.get("contents")
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, str):
            return contents
        return ""

    primary_val = _clean_hover_text(_extract_value(primary_result))
    jaxtyc_val = _clean_hover_text(_extract_value(jaxtyc_result))

    if primary_val and jaxtyc_val:
        combined = primary_val + "\n\n---\n\n" + jaxtyc_val
    else:
        combined = primary_val or jaxtyc_val

    if not combined:
        return primary_result or jaxtyc_result
    return {"contents": {"kind": "markdown", "value": combined}}


def _merge_completion(primary_result: Any, jaxtyc_result: Any) -> Any:
    def _items(r: Any) -> list:
        if isinstance(r, list):
            return r
        if isinstance(r, dict) and "items" in r:
            items = r["items"]
            return items if isinstance(items, list) else []
        return []

    merged = _items(primary_result) + _items(jaxtyc_result)
    if not merged:
        return None
    return {"isIncomplete": False, "items": merged}


def _merge_arrays(primary_result: object, jaxtyc_result: object) -> list | None:
    primary_arr = primary_result if isinstance(primary_result, list) else []
    jaxtyc_arr = jaxtyc_result if isinstance(jaxtyc_result, list) else []
    merged = primary_arr + jaxtyc_arr
    return merged if merged else None


def merge_results(method: str, primary_msg: dict | None, jaxtyc_msg: dict | None) -> object:
    """Merge results from both servers based on method type."""
    primary_result = primary_msg.get("result") if primary_msg else None
    jaxtyc_result = jaxtyc_msg.get("result") if jaxtyc_msg else None

    if method == "textDocument/hover":
        return _merge_hover(primary_result, jaxtyc_result)
    if method == "textDocument/completion":
        return _merge_completion(primary_result, jaxtyc_result)
    if method in ARRAY_MERGE:
        return _merge_arrays(primary_result, jaxtyc_result)
    return primary_result if primary_result is not None else jaxtyc_result


def encode_message(body: dict) -> bytes:
    raw = json.dumps(body).encode("utf-8")
    return f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw


async def read_message(reader: asyncio.StreamReader) -> dict | None:
    header = b""
    while True:
        line = await reader.readline()
        if not line:
            return None
        header += line
        if header.endswith(b"\r\n\r\n"):
            break
    length = 0
    for part in header.decode("ascii").split("\r\n"):
        if part.lower().startswith("content-length:"):
            length = int(part.split(":", 1)[1].strip())
    if length == 0:
        return None
    try:
        body = await reader.readexactly(length)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    return json.loads(body)


def _find_primary_server() -> tuple[str, ...]:
    """Find the best available Python type-checking LSP server."""
    candidates = [
        ("pyright-langserver", "--stdio"),
        ("ty", "server"),
        ("pyright", "--stdio"),
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is not None:
            return cmd
    msg = "No Python type checker found. Install ty (uv tool install ty) or pyright."
    raise RuntimeError(msg)


_VENV_MARKERS = (".venv", "venv")
_FALLBACK_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


def _detect_project_root(file_path: str) -> str | None:
    """Walk up from a file path to find the nearest project root.

    Two-pass search to handle monorepos correctly:
    1. First pass: look for .venv (strongest indicator of the actual project root,
       since nested packages in monorepos have pyproject.toml but not .venv)
    2. Second pass: fall back to nearest pyproject.toml, setup.py, .git, etc.
    """
    start = Path(file_path).resolve().parent
    home = Path.home()

    # Pass 1: find nearest .venv
    current = start
    while current != current.parent:
        if current == home:
            break
        for marker in _VENV_MARKERS:
            if (current / marker).is_dir():
                return str(current)
        current = current.parent

    # Pass 2: fall back to nearest project marker
    current = start
    while current != current.parent:
        if current == home:
            break
        for marker in _FALLBACK_MARKERS:
            if (current / marker).exists():
                return str(current)
        current = current.parent

    return None


def _uri_to_path(uri: str) -> str | None:
    """Convert a file:// URI to a filesystem path."""
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return None


def _patch_root_uri(init_msg: dict, project_root: str) -> dict:
    """Return a copy of the initialize message with rootUri/rootPath rewritten."""
    import copy

    patched = copy.deepcopy(init_msg)
    params = patched.get("params", {})
    root_uri = f"file://{project_root}"
    params["rootUri"] = root_uri
    params["rootPath"] = project_root
    # Also patch workspaceFolders if present
    wf = params.get("workspaceFolders")
    if isinstance(wf, list) and wf:
        wf[0]["uri"] = root_uri
        wf[0]["name"] = Path(project_root).name
    patched["params"] = params
    return patched


def _extract_file_path_from_msg(msg: dict) -> str | None:
    """Extract a file path from any LSP message that references a textDocument."""
    params = msg.get("params", {})
    # textDocument/didOpen, didChange, didSave, didClose
    td = params.get("textDocument", {})
    uri = td.get("uri", "")
    if uri:
        return _uri_to_path(uri)
    return None


async def run_mux() -> None:
    """Start the multiplexer, forwarding between client and both servers.

    Server startup is deferred until the first file-referencing message
    (e.g. textDocument/didOpen) so we can detect the actual project root
    from the file path and set cwd accordingly. This lets ty auto-discover
    the project's .venv instead of relying on rootUri (which may be ~ ).
    """
    loop = asyncio.get_event_loop()

    primary_cmd = _find_primary_server()
    primary_name = primary_cmd[0]

    client_reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(client_reader), sys.stdin.buffer
    )

    stdout_lock = asyncio.Lock()
    diag_cache: dict[str, dict[str, list]] = defaultdict(dict)
    jaxtyc_next_id = 100_000
    dual_requests: dict[int, dict] = {}
    jaxtyc_dual_map: dict[int, int] = {}
    jaxtyc_lifecycle_ids: set[int] = set()

    # Mutable state: servers are None until first file is opened
    primary_proc: asyncio.subprocess.Process | None = None
    jaxtyc_proc: asyncio.subprocess.Process | None = None
    servers_started = False

    # Messages buffered before servers start
    buffered_messages: list[dict] = []

    async def send_to_client(msg: dict) -> None:
        async with stdout_lock:
            sys.stdout.buffer.write(encode_message(msg))
            sys.stdout.buffer.flush()

    async def send_to(proc: asyncio.subprocess.Process | None, msg: dict) -> None:
        if proc is not None and proc.stdin is not None:
            proc.stdin.write(encode_message(msg))
            await proc.stdin.drain()

    async def start_servers(project_root: str | None) -> None:
        nonlocal primary_proc, jaxtyc_proc, servers_started, jaxtyc_next_id

        cwd = project_root or os.getcwd()

        primary_proc = await asyncio.create_subprocess_exec(
            *primary_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        jaxtyc_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "jaxtyc.cli.main",
            "lsp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        servers_started = True

        # Start output reader tasks
        asyncio.create_task(handle_primary_output())
        asyncio.create_task(handle_jaxtyc_output())

        # Replay buffered messages, patching rootUri to the detected project root
        for buffered in buffered_messages:
            method = buffered.get("method", "")
            if method == "initialize":
                # Rewrite rootUri/rootPath so ty discovers the correct .venv
                patched = _patch_root_uri(buffered, cwd)
                await send_to(primary_proc, patched)
                jaxtyc_init = dict(patched)
                jaxtyc_id = jaxtyc_next_id
                jaxtyc_next_id += 1
                jaxtyc_init["id"] = jaxtyc_id
                jaxtyc_lifecycle_ids.add(jaxtyc_id)
                await send_to(jaxtyc_proc, jaxtyc_init)
            elif method == "initialized":
                await send_to(primary_proc, buffered)
                await send_to(jaxtyc_proc, buffered)
            else:
                # Any other buffered notification
                await send_to(primary_proc, buffered)
                await send_to(jaxtyc_proc, buffered)
        buffered_messages.clear()

    async def publish_merged_diagnostics(uri: str) -> None:
        merged = []
        for diags in diag_cache[uri].values():
            merged.extend(diags)
        await send_to_client(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": merged},
            }
        )

    async def finalize_dual(client_id: int) -> None:
        dr = dual_requests.get(client_id)
        if dr is None or dr["done"]:
            return
        dr["done"] = True
        timeout_task = dr.get("_timeout_task")
        if timeout_task is not None:
            timeout_task.cancel()

        result = merge_results(dr["method"], dr["primary_response"], dr["jaxtyc_response"])
        await send_to_client(
            {
                "jsonrpc": "2.0",
                "id": client_id,
                "result": result,
            }
        )

        jaxtyc_id = dr.get("jaxtyc_id")
        if jaxtyc_id is not None:
            jaxtyc_dual_map.pop(jaxtyc_id, None)
        dual_requests.pop(client_id, None)

    async def timeout_dual(client_id: int) -> None:
        await asyncio.sleep(MERGE_TIMEOUT)
        await finalize_dual(client_id)

    async def receive_dual(client_id: int, server: str, msg: dict) -> None:
        dr = dual_requests.get(client_id)
        if dr is None or dr["done"]:
            return

        dr[f"{server}_response"] = msg

        if dr["primary_response"] is not None and dr["jaxtyc_response"] is not None:
            await finalize_dual(client_id)
        elif dr.get("_timeout_task") is None:
            dr["_timeout_task"] = asyncio.create_task(timeout_dual(client_id))

    async def handle_primary_output() -> None:
        assert primary_proc is not None and primary_proc.stdout is not None
        while True:
            msg = await read_message(primary_proc.stdout)
            if msg is None:
                break
            method = msg.get("method", "")

            if method == "textDocument/publishDiagnostics":
                uri = msg["params"]["uri"]
                diag_cache[uri][primary_name] = msg["params"].get("diagnostics", [])
                await publish_merged_diagnostics(uri)
                continue

            msg_id = msg.get("id")
            if msg_id is not None and msg_id in dual_requests:
                await receive_dual(msg_id, "primary", msg)
                continue

            await send_to_client(msg)

    async def handle_jaxtyc_output() -> None:
        assert jaxtyc_proc is not None and jaxtyc_proc.stdout is not None
        while True:
            msg = await read_message(jaxtyc_proc.stdout)
            if msg is None:
                break
            method = msg.get("method", "")

            if method == "textDocument/publishDiagnostics":
                uri = msg["params"]["uri"]
                diag_cache[uri]["jaxtyc"] = msg["params"].get("diagnostics", [])
                await publish_merged_diagnostics(uri)
                continue

            msg_id = msg.get("id")

            if msg_id is not None and msg_id in jaxtyc_lifecycle_ids:
                jaxtyc_lifecycle_ids.discard(msg_id)
                continue

            if msg_id is not None and msg_id in jaxtyc_dual_map:
                client_id = jaxtyc_dual_map[msg_id]
                await receive_dual(client_id, "jaxtyc", msg)
                continue

    async def handle_client_input() -> None:
        nonlocal jaxtyc_next_id
        while True:
            msg = await read_message(client_reader)
            if msg is None:
                break

            method = msg.get("method", "")
            has_id = "id" in msg

            # Before servers start, buffer lifecycle messages and
            # wait for the first file-referencing message to detect project root
            if not servers_started:
                if method in ("initialize", "initialized"):
                    buffered_messages.append(msg)
                    # For initialize, send back a synthetic response immediately
                    # so the client doesn't hang waiting
                    if method == "initialize" and has_id:
                        await send_to_client(
                            {
                                "jsonrpc": "2.0",
                                "id": msg["id"],
                                "result": {
                                    "capabilities": {
                                        "textDocumentSync": 1,
                                        "hoverProvider": True,
                                        "completionProvider": {},
                                        "definitionProvider": True,
                                        "referencesProvider": True,
                                        "documentSymbolProvider": True,
                                        "codeActionProvider": True,
                                        "codeLensProvider": {},
                                        "renameProvider": True,
                                        "prepareRenameProvider": True,
                                        "foldingRangeProvider": True,
                                        "inlayHintProvider": True,
                                        "callHierarchyProvider": True,
                                        "implementationProvider": True,
                                        "workspaceSymbolProvider": True,
                                        "signatureHelpProvider": {
                                            "triggerCharacters": ["(", ","],
                                        },
                                        "semanticTokensProvider": {
                                            "legend": {
                                                "tokenTypes": ["variable"],
                                                "tokenModifiers": ["definition"],
                                            },
                                            "full": True,
                                        },
                                        "linkedEditingRangeProvider": True,
                                        "documentHighlightProvider": True,
                                        "diagnosticProvider": {
                                            "interFileDependencies": False,
                                            "workspaceDiagnostics": False,
                                        },
                                    },
                                    "serverInfo": {
                                        "name": "jaxtyc-mux",
                                        "version": _pkg_version("jaxtyc"),
                                    },
                                },
                            }
                        )
                    continue

                # First file-referencing message: detect project root and start servers
                file_path = _extract_file_path_from_msg(msg)
                project_root = _detect_project_root(file_path) if file_path else None
                await start_servers(project_root)

                # Fall through to normal message handling below

            # Normal message handling (servers are running)
            if method == "initialized":
                await send_to(primary_proc, msg)
                await send_to(jaxtyc_proc, msg)

            elif method == "shutdown":
                await send_to(primary_proc, msg)
                jaxtyc_msg = dict(msg)
                jaxtyc_id = jaxtyc_next_id
                jaxtyc_next_id += 1
                jaxtyc_msg["id"] = jaxtyc_id
                jaxtyc_lifecycle_ids.add(jaxtyc_id)
                await send_to(jaxtyc_proc, jaxtyc_msg)

            elif method == "exit":
                await send_to(primary_proc, msg)
                await send_to(jaxtyc_proc, msg)
                break

            elif method == "$/cancelRequest":
                cancel_id = msg.get("params", {}).get("id")
                await send_to(primary_proc, msg)
                if cancel_id is not None and cancel_id in dual_requests:
                    dr = dual_requests[cancel_id]
                    jaxtyc_id = dr.get("jaxtyc_id")
                    if jaxtyc_id is not None:
                        await send_to(
                            jaxtyc_proc,
                            {
                                "jsonrpc": "2.0",
                                "method": "$/cancelRequest",
                                "params": {"id": jaxtyc_id},
                            },
                        )
                    await finalize_dual(cancel_id)
                else:
                    await send_to(jaxtyc_proc, msg)

            elif has_id and method in DUAL_METHODS:
                client_id = msg["id"]
                jaxtyc_id = jaxtyc_next_id
                jaxtyc_next_id += 1

                dual_requests[client_id] = {
                    "method": method,
                    "jaxtyc_id": jaxtyc_id,
                    "primary_response": None,
                    "jaxtyc_response": None,
                    "done": False,
                    "_timeout_task": None,
                }
                jaxtyc_dual_map[jaxtyc_id] = client_id

                jaxtyc_msg = dict(msg)
                jaxtyc_msg["id"] = jaxtyc_id

                await send_to(primary_proc, msg)
                await send_to(jaxtyc_proc, jaxtyc_msg)

            elif has_id:
                await send_to(primary_proc, msg)

            else:
                await send_to(primary_proc, msg)
                await send_to(jaxtyc_proc, msg)

    client_task = asyncio.create_task(handle_client_input())
    await client_task

    for proc in (primary_proc, jaxtyc_proc):
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
