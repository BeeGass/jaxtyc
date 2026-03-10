"""Semantic tokens handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server

TOKEN_TYPES = ["variable"]
TOKEN_MODIFIERS = ["definition"]

LEGEND = types.SemanticTokensRegistrationOptions(
    legend=types.SemanticTokensLegend(
        token_types=TOKEN_TYPES,
        token_modifiers=TOKEN_MODIFIERS,
    ),
    full=True,
)


@server.feature(types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL, LEGEND)
def semantic_tokens_full(
    ls: LanguageServer, params: types.SemanticTokensParams
) -> types.SemanticTokens | None:
    """Provide semantic tokens for dimension names in jaxtyping annotations."""
    uri = params.text_document.uri
    file_index = _state.workspace_index.get_file(uri)
    if file_index is None:
        return None

    dim_locs = file_index.dim_locations
    if not dim_locs:
        return None

    # Sort by (lineno, col_start) for relative encoding
    sorted_dims = sorted(dim_locs, key=lambda d: (d.lineno, d.col_start))

    # Track first occurrence of each dim name per function for definition modifier
    first_seen: set[tuple[str, str]] = set()
    data: list[int] = []
    prev_line = 0
    prev_start = 0

    for dim in sorted_dims:
        # Skip internal dimension names
        if dim.dim_name.startswith("_"):
            continue

        line = dim.lineno - 1  # Convert to 0-indexed
        start = dim.col_start
        length = dim.col_end - dim.col_start

        # Relative encoding
        delta_line = line - prev_line
        delta_start = start - prev_start if delta_line == 0 else start

        # Token type: 0 = variable
        token_type = 0

        # Modifier: 1 = definition (bit 0) if first occurrence in function
        key = (dim.function_name, dim.dim_name)
        if key not in first_seen:
            first_seen.add(key)
            modifiers = 1  # definition
        else:
            modifiers = 0

        data.extend([delta_line, delta_start, length, token_type, modifiers])
        prev_line = line
        prev_start = start

    return types.SemanticTokens(data=data) if data else None
