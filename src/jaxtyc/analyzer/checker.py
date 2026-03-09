"""Compare expected shapes (from annotations) vs actual shapes (from tracing)."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import Diagnostic
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult


def _format_named_shape(shape: tuple[int, ...], env: DimEnv) -> str:
    """Format a shape tuple with dimension names where known."""
    parts: list[str] = []
    for s in shape:
        name = env.resolve_name(s)
        parts.append(name if name else str(s))
    return "(" + ", ".join(parts) + ")"


def check_function(
    func_spec: FunctionShapeSpec,
    trace: TraceResult,
    env: DimEnv,
) -> list[Diagnostic]:
    """Check a function's trace result against its annotated shape spec.

    Compares the traced output shape against the expected return annotation.
    Reports rank mismatches, dimension mismatches, and trace errors.

    Args:
        func_spec: Parsed shape annotations for the function.
        trace: TraceResult from ``trace_function``.
        env: DimEnv used during tracing (for resolving dimension names in
            diagnostic messages).

    Returns:
        List of Diagnostic objects. Empty if the shapes match or if
        no return annotation exists.
    """
    diagnostics: list[Diagnostic] = []

    # If tracing itself failed, report the error
    if not trace.success:
        diagnostics.append(
            Diagnostic(
                file=func_spec.file_path,
                line=func_spec.lineno,
                col=func_spec.col_offset,
                severity="error",
                message=f"Trace error in `{func_spec.name}`: {trace.error}",
                rule="trace-error",
            )
        )
        return diagnostics

    # Check return shape if annotated
    if func_spec.return_spec is not None and trace.output_shape is not None:
        _check_shape(
            func_spec.return_spec,
            trace.output_shape,
            func_spec,
            env,
            diagnostics,
            context="return",
        )

    return diagnostics


def _check_shape(
    expected_spec: ShapeSpec,
    actual_shape: tuple[int, ...],
    func_spec: FunctionShapeSpec,
    env: DimEnv,
    diagnostics: list[Diagnostic],
    context: str,
) -> None:
    """Compare an expected ShapeSpec against an actual shape tuple."""
    # Skip checking for any-shape specs
    if expected_spec.is_any_shape:
        return

    # Build expected shape from spec
    expected_shape = env.make_shape(expected_spec)

    # Rank check
    if len(expected_shape) != len(actual_shape):
        expected_named = _format_named_shape(expected_shape, env)
        actual_named = _format_named_shape(actual_shape, env)
        diagnostics.append(
            Diagnostic(
                file=func_spec.file_path,
                line=func_spec.lineno,
                col=func_spec.col_offset,
                severity="error",
                message=(
                    f"Rank mismatch in {context} of `{func_spec.name}`\n"
                    f"  Expected: {expected_named} (rank {len(expected_shape)})\n"
                    f"  Got:      {actual_named} (rank {len(actual_shape)})"
                ),
                rule="rank-mismatch",
            )
        )
        return

    # Dimension-by-dimension check
    if expected_shape != actual_shape:
        expected_named = _format_named_shape(expected_shape, env)
        actual_named = _format_named_shape(actual_shape, env)
        diagnostics.append(
            Diagnostic(
                file=func_spec.file_path,
                line=func_spec.lineno,
                col=func_spec.col_offset,
                severity="error",
                message=(
                    f"Shape mismatch in {context} of `{func_spec.name}`\n"
                    f"  Expected: {expected_named}\n"
                    f"  Got:      {actual_named}"
                ),
                rule="shape-mismatch",
            )
        )
