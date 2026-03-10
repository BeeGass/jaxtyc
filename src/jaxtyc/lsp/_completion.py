"""Dimension name completion handler for the jaxtyc LSP server."""

from __future__ import annotations

import re

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server

# Pattern to detect if cursor is inside a jaxtyping shape string.
# Matches: Float[Array, "..., Float32[Array, "..., Shaped[Array, "..., etc.
_JAXTYPING_OPEN = re.compile(
    r"(?:Float|Float16|Float32|Float64|BFloat16|Int|Int8|Int16|Int32|Int64"
    r"|UInt|UInt8|UInt16|UInt32|UInt64|Bool|Complex|Complex64|Complex128"
    r'|Num|Shaped|Key|Scalar)\s*\[\s*\w+\s*,\s*"'
)


def _is_in_shape_string(line_text: str, col: int) -> bool:
    """Check if the column position is inside a jaxtyping shape string."""
    for m in _JAXTYPING_OPEN.finditer(line_text):
        quote_pos = m.end() - 1  # Position of the opening quote
        # Find closing quote
        closing = line_text.find('"', quote_pos + 1)
        if closing == -1:
            # Unclosed string, cursor could be inside
            if col > quote_pos:
                return True
        elif quote_pos < col <= closing:
            return True
    return False


def _get_partial_word(line_text: str, col: int) -> str:
    """Extract the partial word being typed at the cursor position."""
    # Walk backwards from cursor to find word start
    start = col
    while start > 0 and (line_text[start - 1].isalnum() or line_text[start - 1] == "_"):
        start -= 1
    return line_text[start:col]


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=[" ", '"']),
)
def completion(ls: LanguageServer, params: types.CompletionParams) -> types.CompletionList | None:
    """Complete dimension names inside jaxtyping shape strings."""
    uri = params.text_document.uri
    pos = params.position

    doc = ls.workspace.get_text_document(uri)
    lines = doc.source.split("\n") if doc.source else []
    if pos.line >= len(lines):
        return None

    line_text = lines[pos.line]
    if not _is_in_shape_string(line_text, pos.character):
        return None

    prefix = _get_partial_word(line_text, pos.character)

    # Gather all known dimension names from the workspace index
    known_dims: set[str] = set()
    for file_index in _state.workspace_index.all_files():
        for dim_loc in file_index.dim_locations:
            # Skip internal dim names (variadic/ellipsis/anonymous)
            if dim_loc.dim_name.startswith("_"):
                continue
            known_dims.add(dim_loc.dim_name)

    # Filter by prefix
    matching = (
        sorted(d for d in known_dims if d.startswith(prefix)) if prefix else sorted(known_dims)
    )

    if not matching:
        return None

    items = [
        types.CompletionItem(
            label=dim_name,
            kind=types.CompletionItemKind.Variable,
            detail="dimension name",
            insert_text=dim_name,
        )
        for dim_name in matching
    ]

    return types.CompletionList(is_incomplete=False, items=items)
