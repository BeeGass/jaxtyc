"""Shared utility functions for LSP handlers."""

from __future__ import annotations

from urllib.parse import unquote
from urllib.parse import urlparse

from lsprotocol import types

from jaxtyc.lsp import _state
from jaxtyc.types import DimLocation
from jaxtyc.types import FunctionShapeSpec


def uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    parsed = urlparse(uri)
    return unquote(parsed.path)


def dim_range(dim: DimLocation) -> types.Range:
    """Build an LSP Range for a DimLocation."""
    line = max(0, dim.lineno - 1)
    return types.Range(
        start=types.Position(line=line, character=dim.col_start),
        end=types.Position(line=line, character=dim.col_end),
    )


def spec_range(spec: FunctionShapeSpec) -> types.Range:
    """Build an LSP Range for a FunctionShapeSpec definition line."""
    line = max(0, spec.lineno - 1)
    return types.Range(
        start=types.Position(line=line, character=spec.col_offset),
        end=types.Position(line=line, character=spec.name_col_offset + len(spec.name)),
    )


def spec_selection_range(spec: FunctionShapeSpec) -> types.Range:
    """Build an LSP selection Range for the function name only."""
    line = max(0, spec.lineno - 1)
    return types.Range(
        start=types.Position(line=line, character=spec.name_col_offset),
        end=types.Position(line=line, character=spec.name_col_offset + len(spec.name)),
    )


def shape_summary(spec: FunctionShapeSpec) -> str:
    """Build a shape summary string like '(batch, seq) -> (batch, hidden)'."""
    parts = []
    for pname, pspec in spec.params.items():
        dim_names = ", ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
        parts.append(f"{pname}: ({dim_names})")
    ret = ""
    if spec.return_spec is not None:
        ret_dims = ", ".join(d.name or str(d.size) or d.kind for d in spec.return_spec.dims)
        ret = f" -> ({ret_dims})"
    return f"{', '.join(parts)}{ret}"


def debounce_seconds() -> float:
    """Get the debounce delay in seconds from config."""
    return _state.config.debounce_ms / 1000.0
