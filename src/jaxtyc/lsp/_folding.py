"""Folding range handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server


@server.feature(types.TEXT_DOCUMENT_FOLDING_RANGE)
def folding_range(
    ls: LanguageServer, params: types.FoldingRangeParams
) -> list[types.FoldingRange] | None:
    """Provide folding ranges for multi-parameter shape-annotated functions."""
    uri = params.text_document.uri
    file_index = _state.workspace_index.get_file(uri)
    if file_index is None:
        return None

    ranges: list[types.FoldingRange] = []
    for spec in file_index.function_specs:
        # Only fold functions with 3+ parameters that span multiple lines
        if len(spec.params) < 3:
            continue
        start_line = spec.lineno - 1  # 0-indexed
        end_line = spec.end_lineno - 1 if spec.end_lineno > 0 else start_line
        if end_line <= start_line:
            continue
        ranges.append(
            types.FoldingRange(
                start_line=start_line,
                end_line=end_line,
                kind=types.FoldingRangeKind.Region,
            )
        )

    return ranges or None
