"""Compare expected shapes (from annotations) vs actual shapes (from tracing)."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import CallSite
from jaxtyc.types import Diagnostic
from jaxtyc.types import DiagnosticData
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

    # Check parameter shapes for consistency
    if trace.input_shapes:
        for pname, pspec in func_spec.params.items():
            if pspec.is_any_shape or pname not in trace.input_shapes:
                continue
            expected_param_shape = env.make_shape(pspec)
            actual_param_shape = trace.input_shapes[pname]
            if expected_param_shape != actual_param_shape:
                expected_named = _format_named_shape(expected_param_shape, env)
                actual_named = _format_named_shape(actual_param_shape, env)
                diagnostics.append(
                    Diagnostic(
                        file=func_spec.file_path,
                        line=func_spec.lineno,
                        col=func_spec.col_offset,
                        severity="error",
                        message=(
                            f"Parameter `{pname}` shape inconsistency in `{func_spec.name}`\n"
                            f"  Annotated: {expected_named}\n"
                            f"  Resolved:  {actual_named}"
                        ),
                        rule="param-inconsistency",
                        data=DiagnosticData(
                            expected_shape=expected_param_shape,
                            actual_shape=actual_param_shape,
                            expected_named=_named_shape_tuple(expected_param_shape, env),
                            actual_named=_named_shape_tuple(actual_param_shape, env),
                            dim_name_mapping=env.name_size_mapping(),
                            rule="param-inconsistency",
                        ),
                    )
                )

    # Check multi-output return (tuple[Float[...], Float[...]])
    if func_spec.return_specs is not None and trace.output_shapes is not None:
        if len(func_spec.return_specs) != len(trace.output_shapes):
            diagnostics.append(
                Diagnostic(
                    file=func_spec.file_path,
                    line=func_spec.lineno,
                    col=func_spec.col_offset,
                    severity="error",
                    message=(
                        f"Return count mismatch in `{func_spec.name}`\n"
                        f"  Expected {len(func_spec.return_specs)} outputs, "
                        f"got {len(trace.output_shapes)}"
                    ),
                    rule="return-count-mismatch",
                )
            )
        else:
            for i, (rspec, actual_shape) in enumerate(
                zip(func_spec.return_specs, trace.output_shapes, strict=True)
            ):
                _check_shape(
                    rspec,
                    actual_shape,
                    func_spec,
                    env,
                    diagnostics,
                    context=f"return[{i}]",
                )
    elif func_spec.return_spec is not None and trace.output_shape is not None:
        # Single return check
        _check_shape(
            func_spec.return_spec,
            trace.output_shape,
            func_spec,
            env,
            diagnostics,
            context="return",
        )

    return diagnostics


def _named_shape_tuple(shape: tuple[int, ...], env: DimEnv) -> tuple[str, ...]:
    """Convert a shape to a tuple of name strings for DiagnosticData."""
    return tuple(env.resolve_name(s) or str(s) for s in shape)


def _suggest_fix(
    expected_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
    env: DimEnv,
    rule: str,
) -> str:
    """Generate a human-readable suggested fix."""
    if rule == "rank-mismatch":
        diff = len(actual_shape) - len(expected_shape)
        if diff > 0:
            return f"Remove {diff} dimension(s) — try jnp.squeeze or indexing"
        return f"Add {-diff} dimension(s) — try jnp.expand_dims"

    # shape-mismatch: detect transposition
    if sorted(expected_shape) == sorted(actual_shape):
        exp_names = _named_shape_tuple(expected_shape, env)
        act_names = _named_shape_tuple(actual_shape, env)
        return f"Transpose: rearrange from ({', '.join(act_names)}) to ({', '.join(exp_names)})"

    mismatched = [
        i for i, (e, a) in enumerate(zip(expected_shape, actual_shape, strict=True)) if e != a
    ]
    dim_desc = ", ".join(
        f"dim {i} (expected {env.resolve_name(expected_shape[i]) or expected_shape[i]}, "
        f"got {env.resolve_name(actual_shape[i]) or actual_shape[i]})"
        for i in mismatched
    )
    return f"Fix {dim_desc}"


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
                data=DiagnosticData(
                    expected_shape=expected_shape,
                    actual_shape=actual_shape,
                    expected_named=_named_shape_tuple(expected_shape, env),
                    actual_named=_named_shape_tuple(actual_shape, env),
                    dim_name_mapping=env.name_size_mapping(),
                    suggested_fix=_suggest_fix(expected_shape, actual_shape, env, "rank-mismatch"),
                    rule="rank-mismatch",
                ),
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
                data=DiagnosticData(
                    expected_shape=expected_shape,
                    actual_shape=actual_shape,
                    expected_named=_named_shape_tuple(expected_shape, env),
                    actual_named=_named_shape_tuple(actual_shape, env),
                    dim_name_mapping=env.name_size_mapping(),
                    suggested_fix=_suggest_fix(expected_shape, actual_shape, env, "shape-mismatch"),
                    rule="shape-mismatch",
                ),
            )
        )


def check_call_site(
    call: CallSite,
    caller_spec: FunctionShapeSpec,
    callee_spec: FunctionShapeSpec,
    callee_trace: TraceResult,
    env: DimEnv,
) -> list[Diagnostic]:
    """Check that a callee's output shape is consistent with caller expectations.

    Args:
        call: The call site linking caller to callee.
        caller_spec: Shape spec of the calling function.
        callee_spec: Shape spec of the called function.
        callee_trace: Trace result of the called function.
        env: Shared DimEnv for dimension resolution.

    Returns:
        List of diagnostics for cross-function shape mismatches.
    """
    diagnostics: list[Diagnostic] = []

    if not callee_trace.success or callee_trace.output_shape is None:
        return diagnostics

    if callee_spec.return_spec is None:
        return diagnostics

    expected_shape = env.make_shape(callee_spec.return_spec)
    actual_shape = callee_trace.output_shape

    if expected_shape != actual_shape:
        expected_named = _format_named_shape(expected_shape, env)
        actual_named = _format_named_shape(actual_shape, env)
        diagnostics.append(
            Diagnostic(
                file=call.file_path,
                line=call.lineno,
                col=call.col_offset,
                severity="error",
                message=(
                    f"Cross-function shape mismatch: `{call.callee_name}` "
                    f"called from `{call.caller_name}`\n"
                    f"  Annotated return: {expected_named}\n"
                    f"  Actual return:    {actual_named}"
                ),
                rule="cross-function-mismatch",
                data=DiagnosticData(
                    expected_shape=expected_shape,
                    actual_shape=actual_shape,
                    expected_named=_named_shape_tuple(expected_shape, env),
                    actual_named=_named_shape_tuple(actual_shape, env),
                    dim_name_mapping=env.name_size_mapping(),
                    rule="cross-function-mismatch",
                ),
            )
        )

    return diagnostics
