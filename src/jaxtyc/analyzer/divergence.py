"""Divergence detection: find where intermediate shapes first deviate from expected."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import ErrorHintInfo
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult


def _expand_expected_dims(
    return_spec: ShapeSpec,
) -> list[tuple[str, str | int | None]]:
    """Expand a return ShapeSpec into (kind, identifier) pairs.

    For named dims the identifier is the dim name string. For fixed dims
    it is the literal int size. For anonymous dims it is None (matches
    anything). Variadic and ellipsis dims expand to two named entries
    using their synthetic names.
    """
    result: list[tuple[str, str | int | None]] = []
    for dim in return_spec.dims:
        match dim.kind:
            case "named":
                result.append(("named", dim.name))
            case "fixed":
                result.append(("fixed", dim.size))
            case "variadic":
                result.append(("named", f"_var_{dim.name}_0"))
                result.append(("named", f"_var_{dim.name}_1"))
            case "ellipsis":
                result.append(("named", "_ellipsis_0"))
                result.append(("named", "_ellipsis_1"))
            case "anonymous":
                result.append(("anonymous", None))
    return result


def _matches_expected(
    inter: IntermediateShape,
    expected_dims: list[tuple[str, str | int | None]],
) -> bool:
    """Check if an intermediate shape matches expected dims.

    Uses ``named_shape`` for named dims (avoiding cross-scope symbolic
    comparison issues) and raw ``shape`` values for fixed dims.
    """
    if len(inter.named_shape) != len(expected_dims):
        return False
    for i, (kind, ident) in enumerate(expected_dims):
        if kind == "anonymous":
            continue
        if kind == "fixed" and i < len(inter.shape) and inter.shape[i] != ident:
            return False
        if kind == "named":
            # Skip comparison when named_shape is None — this happens for
            # broadcast/constant intermediates with literal sizes (e.g. 1)
            # that the DimEnv couldn't resolve to a name.
            if inter.named_shape[i] is None:
                continue
            if inter.named_shape[i] != ident:
                return False
    return True


def _build_dim_message(
    inter: IntermediateShape,
    expected_dims: list[tuple[str, str | int | None]],
) -> tuple[str, str]:
    """Build a divergence message and rule for a mismatched intermediate."""
    if len(inter.named_shape) != len(expected_dims):
        return (
            f"Rank changed to {len(inter.shape)} (expected {len(expected_dims)}) "
            f"at {inter.op_name}",
            "rank-mismatch",
        )

    mismatched_dims: list[str] = []
    for i, (kind, ident) in enumerate(expected_dims):
        if kind == "anonymous":
            continue
        actual_name = inter.named_shape[i] or str(inter.shape[i])
        is_fixed_mismatch = kind == "fixed" and i < len(inter.shape) and inter.shape[i] != ident
        is_named_mismatch = (
            kind == "named" and inter.named_shape[i] is not None and inter.named_shape[i] != ident
        )
        if is_fixed_mismatch or is_named_mismatch:
            mismatched_dims.append(f"dim {i}: expected {ident}, got {actual_name}")

    msg = "; ".join(mismatched_dims)
    return msg, "shape-mismatch"


def find_divergence_points(
    func_spec: FunctionShapeSpec,
    trace: TraceResult,
    env: DimEnv | None = None,
) -> list[ErrorHintInfo]:
    """Find the first intermediate shape that diverges from the expected return shape.

    Compares using ``named_shape`` from each intermediate (resolved by the
    tracing DimEnv) against the return spec's dimension names. This avoids
    cross-scope symbolic comparison issues and concrete-size mismatches
    from sharding adjustments.

    Args:
        func_spec: Function annotations with expected return shape.
        trace: Trace result with intermediate shapes.
        env: Unused. Kept for backward compatibility.

    Returns:
        List with at most one ErrorHintInfo at the first divergence point,
        or empty list if all intermediates match.
    """
    specs_to_check: list[ShapeSpec] = []
    if func_spec.return_specs is not None:
        specs_to_check = [s for s in func_spec.return_specs if not s.is_any_shape]
    elif func_spec.return_spec is not None and not func_spec.return_spec.is_any_shape:
        specs_to_check = [func_spec.return_spec]
    if not specs_to_check:
        return []

    expected_dims = _expand_expected_dims(specs_to_check[0])

    # Filter out intermediates with no source info, sort by line
    valid: list[IntermediateShape] = [
        inter for inter in trace.intermediates if inter.source_file and inter.source_line > 0
    ]
    valid.sort(key=lambda x: x.source_line)

    for inter in valid:
        if not _matches_expected(inter, expected_dims):
            message, rule = _build_dim_message(inter, expected_dims)
            expected_named = tuple(
                str(ident) if ident is not None else "_" for _, ident in expected_dims
            )
            actual_named = tuple(
                n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True)
            )
            return [
                ErrorHintInfo(
                    source_line=inter.source_line,
                    message=message,
                    rule=rule,
                    function_name=func_spec.name,
                    expected_named=expected_named,
                    actual_named=actual_named,
                )
            ]

    return []
