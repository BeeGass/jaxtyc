"""Sharding validation: check PartitionSpec consistency against shapes and mesh."""

from __future__ import annotations

from collections import defaultdict

from jaxtyc.types import Diagnostic
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec

_SHARDABLE_KINDS: frozenset[str] = frozenset({"named", "fixed", "variadic", "anonymous"})


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


def check_sharding_propagation(
    propagated_sharding: object | None,
    return_spec: ShapeSpec | None,
    func_spec: FunctionShapeSpec,
    file_path: str,
) -> list[Diagnostic]:
    """Compare JAX's propagated output sharding against return annotation.

    Args:
        propagated_sharding: NamedSharding from eval_shape output, or None.
        return_spec: Return annotation ShapeSpec with mesh_axis info.
        func_spec: Function spec for diagnostic context.
        file_path: File path for diagnostics.

    Returns:
        List with at most one diagnostic if propagated sharding differs
        from annotated sharding.
    """
    if propagated_sharding is None or return_spec is None:
        return []
    if not return_spec.has_sharding:
        return []

    propagated = tuple(getattr(propagated_sharding, "spec", ()))
    expected = tuple(d.mesh_axis for d in return_spec.dims)

    if propagated == expected:
        return []

    return [
        Diagnostic(
            file=file_path,
            line=func_spec.lineno,
            col=func_spec.col_offset,
            severity="error",
            message=(
                f"Sharding propagation mismatch in `{func_spec.name}`: "
                f"JAX propagated P{propagated} but annotation expects P{expected}"
            ),
            rule="sharding-propagation-mismatch",
        )
    ]


def check_mesh_axes(
    func_spec: FunctionShapeSpec,
    file_path: str,
    mesh_config: dict[str, int],
    axis_rules: dict[str, str],
) -> list[Diagnostic]:
    """Check that all mesh_axis references resolve to known axes.

    A mesh_axis is "known" if it appears as a key in mesh_config (physical axis)
    OR as a key in axis_rules (logical axis that maps to a physical one).

    Args:
        func_spec: Function spec with param and return annotations.
        file_path: File path for diagnostics.
        mesh_config: Physical mesh axis name -> device count.
        axis_rules: Logical axis name -> physical axis name.

    Returns:
        List of sharding-mesh-undefined diagnostics.
    """
    diagnostics: list[Diagnostic] = []
    known_axes = set(mesh_config.keys()) | set(axis_rules.keys())

    all_specs: list[ShapeSpec] = list(func_spec.params.values())
    if func_spec.return_spec is not None:
        all_specs.append(func_spec.return_spec)
    if func_spec.return_specs:
        all_specs.extend(func_spec.return_specs)

    seen: set[str] = set()
    for spec in all_specs:
        for dim in spec.dims:
            if (
                dim.mesh_axis is not None
                and dim.mesh_axis not in known_axes
                and dim.mesh_axis not in seen
            ):
                seen.add(dim.mesh_axis)
                diagnostics.append(
                    Diagnostic(
                        file=file_path,
                        line=func_spec.lineno,
                        col=func_spec.col_offset,
                        severity="error",
                        message=(
                            f"Undefined mesh axis `{dim.mesh_axis}` on dim "
                            f"`{dim.name or '_'}` in `{func_spec.name}`: "
                            f"not found in mesh axes {sorted(mesh_config.keys())} "
                            f"or axis_rules {sorted(axis_rules.keys())}"
                        ),
                        rule="sharding-mesh-undefined",
                    )
                )
    return diagnostics


def check_annotation_sharding(
    func_spec: FunctionShapeSpec,
    file_path: str,
    strict: bool = True,
) -> list[Diagnostic]:
    """Check sharding annotations on a function's parameters and return.

    Implements:
    - sharding-annotation-incomplete: In strict mode, if any dim in a shape
      has mesh_axis set, ALL named dims must have mesh_axis set.
    - sharding-dim-conflict: Same dim name sharded on different axes across params.

    Args:
        func_spec: Function spec with param and return annotations.
        file_path: File path for diagnostics.
        strict: Whether to enforce strict annotation completeness.

    Returns:
        List of sharding annotation diagnostics.
    """
    diagnostics: list[Diagnostic] = []

    # Collect all specs (params + return)
    all_specs: list[tuple[str, ShapeSpec]] = []
    for pname, pspec in func_spec.params.items():
        all_specs.append((f"param `{pname}`", pspec))
    if func_spec.return_spec is not None:
        all_specs.append(("return", func_spec.return_spec))
    if func_spec.return_specs:
        for i, rspec in enumerate(func_spec.return_specs):
            all_specs.append((f"return[{i}]", rspec))

    # Rule: sharding-annotation-incomplete (strict mode only)
    if strict:
        for context, spec in all_specs:
            if not spec.has_sharding:
                continue
            bare_dims = [
                d
                for d in spec.dims
                if d.kind in _SHARDABLE_KINDS and d.mesh_axis is None and not d.sharding_annotated
            ]
            if bare_dims:
                names = ", ".join(d.name or "_" for d in bare_dims)
                diagnostics.append(
                    Diagnostic(
                        file=file_path,
                        line=func_spec.lineno,
                        col=func_spec.col_offset,
                        severity="error",
                        message=(
                            f"Incomplete sharding annotation in {context} of "
                            f"`{func_spec.name}`: dims [{names}] lack |axis or |None"
                        ),
                        rule="sharding-annotation-incomplete",
                    )
                )

    # Rule: sharding-dim-conflict
    # Collect all mesh_axis assignments for each dim name across all specs
    dim_axes: dict[str, set[str]] = defaultdict(set)
    for _context, spec in all_specs:
        for dim in spec.dims:
            if dim.name is not None and dim.mesh_axis is not None:
                dim_axes[dim.name].add(dim.mesh_axis)

    for dim_name, axes in dim_axes.items():
        if len(axes) > 1:
            diagnostics.append(
                Diagnostic(
                    file=file_path,
                    line=func_spec.lineno,
                    col=func_spec.col_offset,
                    severity="warning",
                    message=(
                        f"Dim `{dim_name}` sharded on conflicting axes in "
                        f"`{func_spec.name}`: {', '.join(sorted(axes))}"
                    ),
                    rule="sharding-dim-conflict",
                )
            )

    return diagnostics
