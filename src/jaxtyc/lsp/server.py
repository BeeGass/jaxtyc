"""pygls-based LSP server for jaxtyc shape checking."""

from __future__ import annotations

import logging
from urllib.parse import unquote
from urllib.parse import urlparse

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.types import IntermediateShape

logger = logging.getLogger(__name__)

server = LanguageServer("jaxtyc", "v0.1.0")

# Cache of analysis results per URI
_analysis_cache: dict[str, list[IntermediateShape]] = {}


def _uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    parsed = urlparse(uri)
    return unquote(parsed.path)


def _analyze_and_publish(ls: LanguageServer, uri: str) -> None:
    """Analyze a file and publish diagnostics."""
    file_path = _uri_to_path(uri)

    if not file_path.endswith(".py"):
        return

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


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    """Analyze on open."""
    _analyze_and_publish(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    """Re-analyze on save."""
    _analyze_and_publish(ls, params.text_document.uri)


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


def start_lsp() -> None:
    """Start the LSP server on stdio."""
    server.start_io()
