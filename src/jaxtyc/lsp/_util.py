"""Shared utility functions for LSP handlers."""

from __future__ import annotations

from urllib.parse import unquote
from urllib.parse import urlparse

from lsprotocol import types

from jaxtyc.lsp import _state
from jaxtyc.types import DimLocation
from jaxtyc.types import DimSpec
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


def dim_label(d: DimSpec) -> str:
    """Build a display label for a single dimension, including mesh_axis if present.

    Returns ``"name|axis"`` when the dim has a ``mesh_axis``, otherwise just
    the plain name/size/kind fallback.
    """
    base = d.name or str(d.size) or d.kind
    if d.mesh_axis is not None:
        return f"{base}|{d.mesh_axis}"
    return base


def shape_summary(spec: FunctionShapeSpec) -> str:
    """Build a shape summary string like '(batch, seq) -> (batch, hidden)'."""
    parts = []
    for pname, pspec in spec.params.items():
        dim_names = ", ".join(dim_label(d) for d in pspec.dims)
        parts.append(f"{pname}: ({dim_names})")
    ret = ""
    if spec.return_spec is not None:
        ret_dims = ", ".join(dim_label(d) for d in spec.return_spec.dims)
        ret = f" -> ({ret_dims})"
    return f"{', '.join(parts)}{ret}"


def debounce_seconds() -> float:
    """Get the debounce delay in seconds from config."""
    return _state.config.debounce_ms / 1000.0


_NUMPY_ABBREV: dict[str, str] = {
    # Standard dtypes
    "float32": "f32",
    "float64": "f64",
    "float16": "f16",
    "bfloat16": "bf16",
    "int32": "i32",
    "int64": "i64",
    "int16": "i16",
    "int8": "i8",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "complex64": "c64",
    "complex128": "c128",
    "bool": "bool",
    # Sub-byte integers (ml_dtypes)
    "int2": "i2",
    "int4": "i4",
    "uint2": "u2",
    "uint4": "u4",
    # FP4 / FP6 (ml_dtypes)
    "float4_e2m1fn": "f4e2m1fn",
    "float6_e2m3fn": "f6e2m3fn",
    "float6_e3m2fn": "f6e3m2fn",
    # FP8 variants (ml_dtypes)
    "float8_e3m4": "f8e3m4",
    "float8_e4m3": "f8e4m3",
    "float8_e4m3b11fnuz": "f8e4m3b11fnuz",
    "float8_e4m3fn": "f8e4m3fn",
    "float8_e4m3fnuz": "f8e4m3fnuz",
    "float8_e5m2": "f8e5m2",
    "float8_e5m2fnuz": "f8e5m2fnuz",
    "float8_e8m0fnu": "f8e8m0fnu",
}

_JAXTYPING_NAMES: dict[str, str] = {
    # Standard dtypes
    "float32": "Float32",
    "float64": "Float64",
    "float16": "Float16",
    "bfloat16": "BFloat16",
    "int32": "Int32",
    "int64": "Int64",
    "int16": "Int16",
    "int8": "Int8",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "complex64": "Complex64",
    "complex128": "Complex128",
    "bool": "Bool",
    # Sub-byte integers
    "int2": "Int2",
    "int4": "Int4",
    "uint2": "UInt2",
    "uint4": "UInt4",
    # FP4 / FP6
    "float4_e2m1fn": "Float4E2M1FN",
    "float6_e2m3fn": "Float6E2M3FN",
    "float6_e3m2fn": "Float6E3M2FN",
    # FP8 variants
    "float8_e3m4": "Float8E3M4",
    "float8_e4m3": "Float8E4M3",
    "float8_e4m3b11fnuz": "Float8E4M3B11FNUZ",
    "float8_e4m3fn": "Float8E4M3FN",
    "float8_e4m3fnuz": "Float8E4M3FNUZ",
    "float8_e5m2": "Float8E5M2",
    "float8_e5m2fnuz": "Float8E5M2FNUZ",
    "float8_e8m0fnu": "Float8E8M0FNU",
}


def format_dtype(dtype: str, style: str) -> str:
    """Format a dtype string according to the display style."""
    if style == "numpy":
        return _NUMPY_ABBREV.get(dtype, dtype)
    if style == "jaxtyping":
        return _JAXTYPING_NAMES.get(dtype, dtype)
    # "jax" style — return as-is
    return dtype
