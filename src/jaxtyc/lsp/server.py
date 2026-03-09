"""pygls-based LSP server for jaxtyc shape checking."""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.types import IntermediateShape

logger = logging.getLogger(__name__)

server = LanguageServer("jaxtyc", "v0.1.0")

# Cache of analysis results per URI
_analysis_cache: dict[str, list[IntermediateShape]] = {}

# Cache of CodeLens data per URI: list of (0-indexed line, title text)
_codelens_cache: dict[str, list[tuple[int, str]]] = {}

# Debounce state: pending timers per URI
_debounce_timers: dict[str, threading.Timer] = {}
_debounce_lock = threading.Lock()

DEBOUNCE_SECONDS = 0.5


def _uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    parsed = urlparse(uri)
    return unquote(parsed.path)


def _analyze_and_publish(ls: LanguageServer, uri: str, source: str | None = None) -> None:
    """Analyze a file and publish diagnostics.

    If source is provided, write it to a temp file for analysis (used for didChange
    when the editor hasn't saved yet). Otherwise, analyze from disk.
    """
    file_path = _uri_to_path(uri)

    if not file_path.endswith(".py"):
        return

    # Start progress if client supports it
    token = None
    try:
        caps = ls.client_capabilities
        if caps and getattr(getattr(caps, "window", None), "work_done_progress", False):
            token = str(uuid.uuid4())
            ls.work_done_progress.create(token)
            ls.work_done_progress.begin(
                token,
                types.WorkDoneProgressBegin(
                    title="jaxtyc",
                    message=f"Analyzing {Path(file_path).name}...",
                ),
            )
    except Exception:
        token = None

    if source is not None:
        # Write in-memory content to temp file for analysis
        suffix = Path(file_path).suffix
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            result = analyze_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        # Override the file path in diagnostics to point to the real file
        patched_diags = []
        for d in result.diagnostics:
            patched_diags.append(
                type(d)(
                    file=file_path,
                    line=d.line,
                    col=d.col,
                    severity=d.severity,
                    message=d.message,
                    rule=d.rule,
                )
            )
        from jaxtyc.types import FileResult

        result = FileResult(
            file_path=file_path,
            functions_checked=result.functions_checked,
            diagnostics=patched_diags,
            trace_results=result.trace_results,
        )
    else:
        result = analyze_file(file_path)

    # Convert to LSP diagnostics
    lsp_diagnostics: list[types.Diagnostic] = []
    for diag in result.diagnostics:
        if diag.severity == "error":
            severity = types.DiagnosticSeverity.Error
        elif diag.severity == "warning":
            severity = types.DiagnosticSeverity.Warning
        else:
            severity = types.DiagnosticSeverity.Information

        lsp_diagnostics.append(
            types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=max(0, diag.line - 1), character=diag.col),
                    end=types.Position(line=max(0, diag.line - 1), character=diag.col + 1),
                ),
                message=diag.message,
                severity=severity,
                source="jaxtyc",
                code=diag.rule,
            )
        )

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=lsp_diagnostics,
        )
    )

    # Cache intermediates for hover
    all_intermediates: list[IntermediateShape] = []
    for trace in result.trace_results:
        all_intermediates.extend(trace.intermediates)
    _analysis_cache[uri] = all_intermediates

    # Build CodeLens cache from function specs + trace results
    try:
        source_text = Path(file_path).read_text(encoding="utf-8") if source is None else source
        func_specs = extract_function_specs(source_text, file_path)
        lenses: list[tuple[int, str]] = []
        trace_by_name = {t.function_name: t for t in result.trace_results}
        from jaxtyc.analyzer.dim_env import DimEnv

        for spec in func_specs:
            trace = trace_by_name.get(spec.name)
            if trace is None or not trace.success:
                continue
            param_parts = []
            for pname, pspec in spec.params.items():
                dim_names = ", ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
                param_parts.append(f"{pname}: ({dim_names})")
            ret_part = ""
            if trace.output_shape is not None:
                env_for_names = DimEnv()
                # Rebuild env with the same dim names as the spec
                for pspec in spec.params.values():
                    env_for_names.make_shape(pspec)
                out_names = env_for_names.shape_to_names(trace.output_shape)
                named_out = ", ".join(
                    n or str(s) for n, s in zip(out_names, trace.output_shape, strict=True)
                )
                ret_part = f" -> ({named_out})"
            title = f"shapes: {', '.join(param_parts)}{ret_part}"
            lenses.append((spec.lineno - 1, title))  # Convert to 0-indexed
        _codelens_cache[uri] = lenses
    except Exception:
        _codelens_cache[uri] = []

    # End progress
    if token is not None:
        with contextlib.suppress(Exception):
            ls.work_done_progress.end(
                token,
                types.WorkDoneProgressEnd(message="Analysis complete"),
            )


def _schedule_debounced_analysis(ls: LanguageServer, uri: str, source: str) -> None:
    """Schedule a debounced analysis for a URI. Cancels any pending timer."""
    with _debounce_lock:
        existing = _debounce_timers.pop(uri, None)
        if existing is not None:
            existing.cancel()

        timer = threading.Timer(
            DEBOUNCE_SECONDS,
            _analyze_and_publish,
            args=(ls, uri, source),
        )
        timer.daemon = True
        _debounce_timers[uri] = timer
        timer.start()


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    """Analyze on open (runs in thread pool to avoid blocking event loop)."""
    _analyze_and_publish(ls, params.text_document.uri)


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    """Re-analyze on save (runs in thread pool to avoid blocking event loop)."""
    # Cancel any pending debounced analysis — save takes priority
    with _debounce_lock:
        existing = _debounce_timers.pop(params.text_document.uri, None)
        if existing is not None:
            existing.cancel()
    _analyze_and_publish(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    """Debounced analysis on edit. Cancels pending, schedules new after 500ms."""
    uri = params.text_document.uri
    # Get the latest content from the change events (full sync)
    if params.content_changes:
        source = params.content_changes[-1].text
        _schedule_debounced_analysis(ls, uri, source)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Show intermediate shape at cursor position."""
    uri = params.text_document.uri
    pos = params.position

    intermediates = _analysis_cache.get(uri, [])
    if not intermediates:
        return None

    # Find intermediates matching the cursor line
    line = pos.line + 1  # LSP is 0-indexed, our source_line is 1-indexed
    matching = [i for i in intermediates if i.source_line == line]

    if not matching:
        return None

    # Build hover content
    lines: list[str] = []
    for inter in matching:
        named = ", ".join(n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True))
        lines.append(f"`{inter.op_name}`: ({named})  `{inter.dtype}`")

    content = "\n\n".join(lines)

    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=content,
        ),
    )


@server.feature(types.TEXT_DOCUMENT_CODE_LENS)
def code_lens(ls: LanguageServer, params: types.CodeLensParams) -> list[types.CodeLens]:
    """Return shape annotations as CodeLens items above function definitions."""
    uri = params.text_document.uri
    lenses = _codelens_cache.get(uri, [])

    return [
        types.CodeLens(
            range=types.Range(
                start=types.Position(line=line, character=0),
                end=types.Position(line=line, character=0),
            ),
            command=types.Command(
                title=title,
                command="",
            ),
        )
        for line, title in lenses
    ]


def start_lsp() -> None:
    """Start the LSP server on stdio."""
    server.start_io()
