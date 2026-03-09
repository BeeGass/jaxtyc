"""Core data types for jaxtyc shape analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DimSpec:
    """A single dimension in a shape specification.

    Attributes:
        kind: Dimension kind — "named" for symbolic dims, "fixed" for literal
            integers, "variadic" for ``*batch``-style dims, "anonymous" for ``_``,
            and "ellipsis" for ``...``.
        name: Symbolic name for named/variadic dims. None for fixed/anonymous/ellipsis.
        size: Literal integer size for fixed dims. None otherwise.
    """

    kind: Literal["named", "fixed", "variadic", "anonymous", "ellipsis"]
    name: str | None = None
    size: int | None = None


@dataclass(frozen=True)
class ShapeSpec:
    """Parsed shape specification from a jaxtyping annotation.

    Attributes:
        dims: Ordered tuple of dimension specs comprising the shape.
        dtype: Dtype string (e.g. "float32", "bfloat16") resolved from the
            jaxtyping dtype class.
        is_scalar: True if the annotation describes a scalar (empty shape string).
        is_any_shape: True if the annotation is ``...`` (skip shape checking).
    """

    dims: tuple[DimSpec, ...]
    dtype: str
    is_scalar: bool = False
    is_any_shape: bool = False


@dataclass(frozen=True)
class FunctionShapeSpec:
    """Shape annotations extracted from a function signature.

    Attributes:
        name: Function (or method) name.
        file_path: Absolute path to the source file containing the function.
        lineno: 1-based line number of the ``def`` statement.
        col_offset: 0-based column offset of the ``def`` statement.
        params: Map of parameter name to its parsed ShapeSpec. Only parameters
            with jaxtyping annotations are included.
        return_spec: Parsed ShapeSpec for the return annotation, or None if the
            return type is not a jaxtyping annotation.
        is_method: True if the function is defined inside a class body.
        class_name: Enclosing class name if ``is_method`` is True, else None.
    """

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
    """A shape-checking diagnostic (error, warning, or info).

    Attributes:
        file: Absolute path to the file where the diagnostic was raised.
        line: 1-based line number (0 if file-level).
        col: 0-based column offset (0 if not available).
        severity: Diagnostic severity level.
        message: Human-readable description of the issue.
        rule: Machine-readable rule identifier (e.g. "shape-mismatch",
            "rank-mismatch", "trace-error").
    """

    file: str
    line: int
    col: int
    severity: Literal["error", "warning", "info"]
    message: str
    rule: str


@dataclass(frozen=True)
class IntermediateShape:
    """Shape of an intermediate value at a specific source location.

    Attributes:
        shape: Concrete shape tuple from JAX tracing.
        dtype: Dtype string (e.g. "float32").
        source_file: Path to the user source file that produced this value.
            Empty string if no user frame was found.
        source_line: 1-based line number in ``source_file`` (0 if unknown).
        source_col: 0-based column offset (always 0; JAX does not expose columns).
        named_shape: Shape with dimension names resolved via DimEnv where
            possible; entries are None for unrecognised sizes.
        op_name: Name of the JAX primitive that produced this value
            (e.g. "dot_general", "add").
    """

    shape: tuple[int, ...]
    dtype: str
    source_file: str
    source_line: int
    source_col: int
    named_shape: tuple[str | None, ...]
    op_name: str


@dataclass(frozen=True)
class TraceResult:
    """Result of tracing a single function with jax.eval_shape.

    Attributes:
        function_name: Name of the traced function.
        output_shape: Concrete output shape from eval_shape, or None on failure.
        output_dtype: Output dtype string, or None on failure.
        intermediates: Shapes of all intermediate values extracted via make_jaxpr.
        error: Error message string if tracing failed, None on success.
    """

    function_name: str
    output_shape: tuple[int, ...] | None
    output_dtype: str | None
    intermediates: list[IntermediateShape]
    error: str | None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class DimLocation:
    """Source location of a dimension name token within a jaxtyping annotation.

    Attributes:
        dim_name: The dimension name as written (e.g. "batch", "seq").
        param_name: Parameter name this dim belongs to, or "__return__" for return.
        function_name: Name of the enclosing function.
        file_path: Absolute path to the source file.
        lineno: 1-based line number of the shape string constant.
        col_start: 0-based column of the first character of this dim token.
        col_end: 0-based column one past the last character of this dim token.
    """

    dim_name: str
    param_name: str
    function_name: str
    file_path: str
    lineno: int
    col_start: int
    col_end: int


@dataclass(frozen=True)
class CallSite:
    """A call from one function to another within a source file.

    Attributes:
        caller_name: Fully qualified name of the calling function.
        callee_name: Name of the called function (may be unqualified).
        file_path: File containing the call.
        lineno: 1-based line of the call expression.
        col_offset: 0-based column of the call expression start.
        end_col_offset: 0-based column one past the end of the callee name.
    """

    caller_name: str
    callee_name: str
    file_path: str
    lineno: int
    col_offset: int
    end_col_offset: int


@dataclass(frozen=True)
class FileResult:
    """Result of analyzing all functions in a file.

    Attributes:
        file_path: Absolute path to the analyzed file.
        functions_checked: Number of annotated functions that were successfully
            traced and checked.
        diagnostics: All diagnostics produced during analysis (errors, warnings,
            info messages).
        trace_results: Per-function trace results for every function that was traced.
    """

    file_path: str
    functions_checked: int
    diagnostics: list[Diagnostic]
    trace_results: list[TraceResult]
