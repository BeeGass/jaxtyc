"""Divergence detection: find where intermediate shapes first deviate from expected."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import ErrorHintInfo
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import TraceResult


def _matches_expected(
    inter_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> bool:
    """Check if an intermediate shape is compatible with the expected shape."""
    if len(inter_shape) != len(expected_shape):
        return False
    return all(i == e for i, e in zip(inter_shape, expected_shape, strict=True))


def _build_dim_message(
    inter: IntermediateShape,
    expected_shape: tuple[int, ...],
    env: DimEnv,
) -> tuple[str, str]:
    """Build a divergence message and rule for a mismatched intermediate."""
    if len(inter.shape) != len(expected_shape):
        return (
            f"Rank changed to {len(inter.shape)} (expected {len(expected_shape)}) "
            f"at {inter.op_name}",
            "rank-mismatch",
        )

    mismatched_dims: list[str] = []
    for i, (actual, exp) in enumerate(zip(inter.shape, expected_shape, strict=True)):
        if actual != exp:
            actual_name = env.resolve_name(actual) or str(actual)
            expected_name = env.resolve_name(exp) or str(exp)
            mismatched_dims.append(f"dim {i}: expected {expected_name}, got {actual_name}")

    msg = "; ".join(mismatched_dims)
    return msg, "shape-mismatch"


def find_divergence_points(
    func_spec: FunctionShapeSpec,
    trace: TraceResult,
    env: DimEnv,
) -> list[ErrorHintInfo]:
    """Find the first intermediate shape that diverges from the expected return shape.

    Args:
        func_spec: Function annotations with expected return shape.
        trace: Trace result with intermediate shapes.
        env: DimEnv for resolving dimension names.

    Returns:
        List with at most one ErrorHintInfo at the first divergence point,
        or empty list if all intermediates match.
    """
    if func_spec.return_spec is None:
        return []
    if func_spec.return_spec.is_any_shape:
        return []

    expected_shape = env.make_shape(func_spec.return_spec)

    # Filter out intermediates with no source info, sort by line
    valid: list[IntermediateShape] = [
        inter for inter in trace.intermediates if inter.source_file and inter.source_line > 0
    ]
    valid.sort(key=lambda x: x.source_line)

    for inter in valid:
        if not _matches_expected(inter.shape, expected_shape):
            message, rule = _build_dim_message(inter, expected_shape, env)
            named = env.shape_to_names(inter.shape)
            expected_named = env.shape_to_names(expected_shape)
            return [
                ErrorHintInfo(
                    source_line=inter.source_line,
                    message=message,
                    rule=rule,
                    function_name=func_spec.name,
                    expected_named=tuple(
                        n or str(s) for n, s in zip(expected_named, expected_shape, strict=True)
                    ),
                    actual_named=tuple(
                        n or str(s) for n, s in zip(named, inter.shape, strict=True)
                    ),
                )
            ]

    return []
