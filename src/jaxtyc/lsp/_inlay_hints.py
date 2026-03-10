"""Inlay hint handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hint(ls: LanguageServer, params: types.InlayHintParams) -> list[types.InlayHint] | None:
    """Show resolved shapes as inline hints next to operations."""
    uri = params.text_document.uri
    with _state.cache_lock:
        intermediates = _state.analysis_cache.get(uri, [])
    if not intermediates:
        return None

    start_line = params.range.start.line + 1  # Convert to 1-based
    end_line = params.range.end.line + 1

    hints: list[types.InlayHint] = []
    seen_lines: set[int] = set()

    for inter in intermediates:
        if inter.source_line < start_line or inter.source_line > end_line:
            continue
        if inter.source_line in seen_lines:
            continue
        seen_lines.add(inter.source_line)

        named = ", ".join(n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True))
        label = f"  ({named})  {inter.dtype}"

        hints.append(
            types.InlayHint(
                position=types.Position(line=inter.source_line - 1, character=999),
                label=label,
                kind=types.InlayHintKind.Type,
                padding_left=True,
            )
        )

    return hints or None
