"""Core data types for jaxtyc shape analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DimSpec:
    """A single dimension in a shape specification."""

    kind: Literal["named", "fixed", "variadic", "anonymous", "ellipsis"]
    name: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class ShapeSpec:
    """Parsed shape specification from a jaxtyping annotation."""

    dims: tuple[DimSpec, ...]
    dtype: str
    is_scalar: bool = False
    is_any_shape: bool = False


@dataclass(frozen=True)
class FunctionShapeSpec:
    """Shape annotations extracted from a function signature."""

    name: str
    file_path: str
    lineno: int
    col_offset: int
    params: dict[str, ShapeSpec]
    return_spec: ShapeSpec | None
    is_method: bool = False
    class_name: str | None = None


@dataclass(frozen=True)
class Diagnostic:
    """A shape-checking diagnostic (error, warning, or info)."""

    file: str
    line: int
    col: int
    severity: Literal["error", "warning", "info"]
    message: str
    rule: str


@dataclass(frozen=True)
class IntermediateShape:
    """Shape of an intermediate value at a specific source location."""

    shape: tuple[int, ...]
    dtype: str
    source_file: str
    source_line: int
    source_col: int
    named_shape: tuple[str | None, ...]
    op_name: str


@dataclass(frozen=True)
class TraceResult:
    """Result of tracing a single function with jax.eval_shape."""

    function_name: str
    output_shape: tuple[int, ...] | None
    output_dtype: str | None
    intermediates: list[IntermediateShape]
    error: str | None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class FileResult:
    """Result of analyzing all functions in a file."""

    file_path: str
    functions_checked: int
    diagnostics: list[Diagnostic]
    trace_results: list[TraceResult]
