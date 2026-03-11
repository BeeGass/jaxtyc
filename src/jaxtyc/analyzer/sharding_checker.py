"""Sharding validation: check PartitionSpec consistency against shapes and mesh."""

from __future__ import annotations

from collections import defaultdict

from jaxtyc.types import Diagnostic
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape


def check_sharding(
    intermediates: list[IntermediateShape],
    func_spec: FunctionShapeSpec,
    file_path: str,
) -> list[Diagnostic]:
    """Check sharding constraints for consistency.

    Implements four rules:
    - sharding-rank-mismatch: PartitionSpec length != array rank
    - sharding-axis-unknown: PartitionSpec references axis not in mesh
    - sharding-conflict: conflicting specs on same shape at same line
    - sharding-io-mismatch: jit out_shardings contradict inner constraints

    Args:
        intermediates: All traced intermediate shapes (may include non-sharded).
        func_spec: Function spec for diagnostic context.
        file_path: File path for diagnostics.

    Returns:
        List of sharding diagnostics.
    """
    diagnostics: list[Diagnostic] = []
    sharded = [i for i in intermediates if i.sharding is not None]
    if not sharded:
        return diagnostics

    for inter in sharded:
        info = inter.sharding
        assert info is not None  # guarded above

        # Rule: sharding-rank-mismatch
        if len(info.partition_spec) != len(inter.shape):
            diagnostics.append(
                Diagnostic(
                    file=file_path,
                    line=info.source_line or inter.source_line,
                    col=inter.source_col,
                    severity="error",
                    message=(
                        f"Sharding rank mismatch in `{func_spec.name}`: "
                        f"PartitionSpec has {len(info.partition_spec)} entries "
                        f"but array has rank {len(inter.shape)}"
                    ),
                    rule="sharding-rank-mismatch",
                )
            )

        # Rule: sharding-axis-unknown
        mesh_axes = set(info.mesh_axis_names)
        for axis in info.partition_spec:
            if axis is not None and axis not in mesh_axes:
                diagnostics.append(
                    Diagnostic(
                        file=file_path,
                        line=info.source_line or inter.source_line,
                        col=inter.source_col,
                        severity="error",
                        message=(
                            f"Unknown mesh axis `{axis}` in `{func_spec.name}`: "
                            f"mesh has axes {info.mesh_axis_names}"
                        ),
                        rule="sharding-axis-unknown",
                    )
                )

    # Rule: sharding-conflict — group by (source_line, shape)
    groups: dict[tuple[int, tuple[int, ...]], list[IntermediateShape]] = defaultdict(list)
    for inter in sharded:
        assert inter.sharding is not None
        key = (inter.source_line, inter.shape)
        groups[key].append(inter)

    for (line, _shape), group in groups.items():
        specs = set()
        for inter in group:
            assert inter.sharding is not None
            specs.add(inter.sharding.partition_spec)
        if len(specs) > 1:
            diagnostics.append(
                Diagnostic(
                    file=file_path,
                    line=line,
                    col=0,
                    severity="error",
                    message=(
                        f"Conflicting sharding specs in `{func_spec.name}` at line {line}: "
                        f"{', '.join(str(s) for s in specs)}"
                    ),
                    rule="sharding-conflict",
                )
            )

    # Rule: sharding-io-mismatch — jit vs sharding_constraint
    jit_sharded = [
        i for i in sharded if i.sharding is not None and i.sharding.source_primitive == "jit"
    ]
    constraint_sharded = [
        i
        for i in sharded
        if i.sharding is not None and i.sharding.source_primitive == "sharding_constraint"
    ]
    for jit_inter in jit_sharded:
        assert jit_inter.sharding is not None
        for c_inter in constraint_sharded:
            assert c_inter.sharding is not None
            if (
                jit_inter.shape == c_inter.shape
                and jit_inter.sharding.partition_spec != c_inter.sharding.partition_spec
            ):
                diagnostics.append(
                    Diagnostic(
                        file=file_path,
                        line=c_inter.source_line,
                        col=0,
                        severity="warning",
                        message=(
                            f"Sharding I/O mismatch in `{func_spec.name}`: "
                            f"jit specifies {jit_inter.sharding.partition_spec} "
                            f"but inner constraint specifies {c_inter.sharding.partition_spec}"
                        ),
                        rule="sharding-io-mismatch",
                    )
                )

    return diagnostics
