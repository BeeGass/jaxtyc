"""Linked editing range handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp._util import dim_range
from jaxtyc.lsp.server import server


@server.feature(types.TEXT_DOCUMENT_LINKED_EDITING_RANGE)
def linked_editing_range(
    ls: LanguageServer, params: types.LinkedEditingRangeParams
) -> types.LinkedEditingRanges | None:
    """Return linked editing ranges for dimension names within a function."""
    uri = params.text_document.uri
    line = params.position.line + 1  # Convert to 1-based
    col = params.position.character

    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    refs = _state.workspace_index.find_dim_references_in_function(
        dim.dim_name, dim.function_name, uri
    )
    if len(refs) < 2:
        return None

    ranges = [dim_range(r) for r in refs]
    return types.LinkedEditingRanges(
        ranges=ranges,
        word_pattern=r"[a-zA-Z_][a-zA-Z0-9_]*",
    )
