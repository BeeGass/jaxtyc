"""Document sync handlers for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server


@server.feature(
    types.TEXT_DOCUMENT_DIAGNOSTIC,
    types.DiagnosticOptions(
        inter_file_dependencies=True,
        workspace_diagnostics=False,
    ),
)
def text_document_diagnostic(
    ls: LanguageServer, params: types.DocumentDiagnosticParams
) -> types.RelatedFullDocumentDiagnosticReport:
    """Pull model: return cached diagnostics for a file."""
    uri = params.text_document.uri
    with _state.cache_lock:
        cached = _state.diagnostics_cache.get(uri, [])
    return types.RelatedFullDocumentDiagnosticReport(
        kind=types.DocumentDiagnosticReportKind.Full,
        items=cached,
    )


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    """Analyze on open (runs in thread pool to avoid blocking event loop)."""
    from jaxtyc.lsp.server import _analyze_and_publish

    _analyze_and_publish(ls, params.text_document.uri)


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    """Re-analyze on save (runs in thread pool to avoid blocking event loop)."""
    from jaxtyc.lsp.server import _analyze_and_publish

    # Cancel any pending debounced analysis -- save takes priority
    with _state.debounce_lock:
        existing = _state.debounce_timers.pop(params.text_document.uri, None)
        if existing is not None:
            existing.cancel()
    _analyze_and_publish(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    """Debounced analysis on edit. Cancels pending, schedules new after delay."""
    from jaxtyc.lsp.server import _schedule_debounced_analysis

    uri = params.text_document.uri
    # Get the latest content from the change events (full sync)
    if params.content_changes:
        source = params.content_changes[-1].text
        _schedule_debounced_analysis(ls, uri, source)


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
    """Clear cached data when a document is closed."""
    uri = params.text_document.uri
    with _state.cache_lock:
        _state.analysis_cache.pop(uri, None)
        _state.codelens_cache.pop(uri, None)
        _state.diagnostics_cache.pop(uri, None)
        _state.dim_env_cache.pop(uri, None)
        _state.error_hints_cache.pop(uri, None)
        _state.source_cache.pop(uri, None)
        _state.trace_results_cache.pop(uri, None)
        _state.content_hash_cache.pop(uri, None)
    # Cancel any pending debounce timer
    with _state.debounce_lock:
        existing = _state.debounce_timers.pop(uri, None)
        if existing is not None:
            existing.cancel()
    # Remove from workspace index
    _state.workspace_index.remove_file(uri)
    # Clear published diagnostics
    ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=[]))
