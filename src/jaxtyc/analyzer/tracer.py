"""Trace functions with jax.eval_shape and jax.make_jaxpr to extract shapes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import jax
from jax.typing import DTypeLike

from jaxtyc.analyzer._errors import truncate_error
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import ShardingInfo
from jaxtyc.types import TraceResult

logger = logging.getLogger(__name__)

_SHARDING_PRIMITIVES: frozenset[str] = frozenset(
    {
        "sharding_constraint",
        "shard_map",
        "jit",
    }
)

# JAX dtype string mapping
_DTYPE_MAP: dict[str, DTypeLike] = {
    "float16": jax.numpy.float16,
    "float32": jax.numpy.float32,
    "float64": jax.numpy.float64,
    "bfloat16": jax.numpy.bfloat16,
    "int": jax.numpy.int32,
    "int8": jax.numpy.int8,
    "int16": jax.numpy.int16,
    "int32": jax.numpy.int32,
    "int64": jax.numpy.int64,
    "uint": jax.numpy.uint32,
    "uint8": jax.numpy.uint8,
    "uint16": jax.numpy.uint16,
    "uint32": jax.numpy.uint32,
    "uint64": jax.numpy.uint64,
    "bool": jax.numpy.bool_,
    "complex": jax.numpy.complex64,
    "complex64": jax.numpy.complex64,
    "complex128": jax.numpy.complex128,
    "numeric": jax.numpy.float32,
    "shaped": jax.numpy.float32,
    "key": jax.numpy.uint32,
    "scalar": jax.numpy.float32,
}


def _resolve_jax_dtype(dtype_str: str) -> DTypeLike:
    """Convert a jaxtyc dtype string to a JAX dtype."""
    return _DTYPE_MAP.get(dtype_str, jax.numpy.float32)


def _build_abstract_input(spec: ShapeSpec, env: DimEnv) -> jax.ShapeDtypeStruct:
    """Build a jax.ShapeDtypeStruct from a ShapeSpec using the DimEnv."""
    shape = env.make_shape(spec)
    dtype = _resolve_jax_dtype(spec.dtype)
    return jax.ShapeDtypeStruct(shape, dtype)


_INTERNAL_PATHS: tuple[str, ...] = (
    "jax/",
    "jaxlib/",
    "flax/",
    "equinox/",
    "einops/",
    "site-packages/jax",
    "site-packages/jaxlib",
    "site-packages/flax",
    "site-packages/equinox",
    "site-packages/einops",
)


def _is_internal_frame(file_name: str) -> bool:
    """Check if a file path is from JAX/Flax/Equinox internals."""
    return any(part in file_name for part in _INTERNAL_PATHS)


def _extract_sharding_info(eqn: Any, source_line: int) -> ShardingInfo | None:
    """Extract ShardingInfo from a jaxpr equation if it is a sharding primitive."""
    try:
        prim_name = eqn.primitive.name
        if prim_name == "sharding_constraint":
            sharding_obj = eqn.params.get("sharding")
            if sharding_obj is None:
                return None
            spec = getattr(sharding_obj, "spec", None)
            if spec is None:
                return None
            # PartitionSpec is a tuple-like of axis names or None
            partition_spec = tuple(spec)
            mesh = getattr(sharding_obj, "mesh", None)
            mesh_axis_names: tuple[str, ...] = ()
            if mesh is not None:
                mesh_axis_names = tuple(mesh.axis_names)
            return ShardingInfo(
                partition_spec=partition_spec,
                mesh_axis_names=mesh_axis_names,
                source_primitive="sharding_constraint",
                source_line=source_line,
            )
        if prim_name == "shard_map":
            mesh = eqn.params.get("mesh")
            out_specs = eqn.params.get("out_names_thunk")
            mesh_axis_names = ()
            if mesh is not None:
                mesh_axis_names = tuple(mesh.axis_names)
            if callable(out_specs):
                out_specs = out_specs()
            if out_specs and len(out_specs) > 0:
                first_spec = out_specs[0]
                partition_spec = tuple(
                    name if isinstance(name, str) else None for name in first_spec
                )
            else:
                partition_spec = ()
            return ShardingInfo(
                partition_spec=partition_spec,
                mesh_axis_names=mesh_axis_names,
                source_primitive="shard_map",
                source_line=source_line,
            )
    except Exception:
        logger.debug("Failed to extract sharding info from %s", eqn.primitive.name, exc_info=True)
    return None


def _extract_intermediates(
    fn: Callable[..., Any],
    abstract_inputs: dict[str, jax.ShapeDtypeStruct],
    env: DimEnv,
) -> list[IntermediateShape]:
    """Extract intermediate shapes from jax.make_jaxpr output."""
    intermediates: list[IntermediateShape] = []

    try:

        def wrapper(**kwargs: Any) -> Any:
            return fn(**kwargs)

        closed_jaxpr = jax.make_jaxpr(wrapper)(**abstract_inputs)
        jaxpr = closed_jaxpr.jaxpr

        for eqn in jaxpr.eqns:
            # Extract source info if available
            source_file = ""
            source_line = 0
            source_col = 0
            if eqn.source_info and eqn.source_info.traceback:
                frames = eqn.source_info.traceback.frames
                # Find the innermost user frame (skip JAX/Flax/Equinox internals).
                # Frames are ordered innermost-first, so the first non-library frame
                # is the user's code — before reaching jaxtyc wrapper frames.
                for frame in frames:
                    if not _is_internal_frame(frame.file_name):
                        source_file = frame.file_name
                        source_line = frame.line_num
                        source_col = 0  # JAX frames don't expose column
                        break

            # Extract sharding info if this is a sharding primitive
            sharding: ShardingInfo | None = None
            if eqn.primitive.name in _SHARDING_PRIMITIVES:
                sharding = _extract_sharding_info(eqn, source_line)

            for outvar in eqn.outvars:
                if hasattr(outvar, "aval") and hasattr(outvar.aval, "shape"):
                    aval = outvar.aval
                    named_shape = env.shape_to_names(aval.shape)
                    intermediates.append(
                        IntermediateShape(
                            shape=aval.shape,
                            dtype=str(aval.dtype),
                            source_file=source_file,
                            source_line=source_line,
                            source_col=source_col,
                            named_shape=named_shape,
                            op_name=eqn.primitive.name,
                            sharding=sharding,
                        )
                    )
    except Exception:
        logger.debug("make_jaxpr failed for intermediates extraction", exc_info=True)

    return intermediates


def _build_abstract_mesh(mesh_config: dict[str, int]) -> Any:
    """Build an AbstractMesh with Explicit axis types from mesh config.

    Args:
        mesh_config: Map of physical axis name to device count.

    Returns:
        AbstractMesh suitable for use with jax.set_mesh().
    """
    from jax.sharding import AbstractMesh
    from jax.sharding import AxisType

    axis_names = tuple(mesh_config.keys())
    axis_sizes = tuple(mesh_config.values())
    axis_types = tuple(AxisType.Explicit for _ in axis_names)
    return AbstractMesh(axis_sizes, axis_names, axis_types=axis_types)


def _build_sharded_abstract_input(
    spec: ShapeSpec,
    env: DimEnv,
    mesh_config: dict[str, int],
    axis_rules: dict[str, str] | None = None,
) -> jax.ShapeDtypeStruct:
    """Build a ShapeDtypeStruct with NamedSharding for a sharded spec.

    Uses concrete sizes that are divisible by the mesh partition size for each
    sharded dimension. Unsharded dims use standard concrete sizes.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    shape: list[int] = []
    pspec_entries: list[str | None] = []

    for i, dim in enumerate(spec.dims):
        axis = dim.mesh_axis
        # Resolve logical -> physical axis name
        if axis is not None and axis_rules:
            axis = axis_rules.get(axis, axis)
        pspec_entries.append(axis)

        if dim.kind == "named":
            if dim.name is None:
                msg = f"DimSpec(kind='named') requires name, got None at index {i}"
                raise ValueError(msg)
            base_size = env.get_concrete_size(dim.name)
            if axis is not None and axis in mesh_config:
                # Make size divisible by partition count
                partition = mesh_config[axis]
                if base_size % partition != 0:
                    base_size = base_size + (partition - base_size % partition)
                    # Update DimEnv so checker's make_concrete_shape matches
                    env._concrete[dim.name] = base_size
            shape.append(base_size)
        elif dim.kind == "fixed":
            if dim.size is None:
                msg = f"DimSpec(kind='fixed') requires size, got None at index {i}"
                raise ValueError(msg)
            shape.append(dim.size)
        elif dim.kind == "variadic":
            if dim.name is None:
                msg = f"DimSpec(kind='variadic') requires name, got None at index {i}"
                raise ValueError(msg)
            shape.extend(
                [
                    env.get_concrete_size(f"_var_{dim.name}_0"),
                    env.get_concrete_size(f"_var_{dim.name}_1"),
                ]
            )
            pspec_entries.append(None)  # second variadic dim unsharded
        elif dim.kind == "ellipsis":
            shape.extend(
                [
                    env.get_concrete_size("_ellipsis_0"),
                    env.get_concrete_size("_ellipsis_1"),
                ]
            )
            pspec_entries.append(None)
        elif dim.kind == "anonymous":
            env._anon_counter += 1
            shape.append(env.get_concrete_size(f"_anon_{env._anon_counter}"))

    dtype = _resolve_jax_dtype(spec.dtype)

    abstract_mesh = _build_abstract_mesh(mesh_config)
    named_sharding = NamedSharding(abstract_mesh, P(*pspec_entries))
    return jax.ShapeDtypeStruct(tuple(shape), dtype, sharding=named_sharding)


def _trace_fallback_unsharded(
    fn: Callable[..., Any],
    params: dict[str, ShapeSpec],
    env: DimEnv,
    original_error: str,
) -> TraceResult:
    """Retry tracing without sharding after a sharded trace failure.

    Args:
        fn: The function to trace.
        params: Map of parameter name to ShapeSpec.
        env: DimEnv for symbolic sizing.
        original_error: Error message from the failed sharded trace.

    Returns:
        TraceResult from unsharded tracing, with sharding_fallback_reason set.
    """
    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for name, spec in params.items():
        if spec.is_any_shape:
            continue
        abstract_inputs[name] = _build_abstract_input(spec, env)

    try:

        def wrapper(**kwargs: Any) -> Any:
            return fn(**kwargs)

        output_struct = jax.eval_shape(wrapper, **abstract_inputs)
    except Exception as e:
        return TraceResult(
            function_name=getattr(fn, "__name__", "<unknown>"),
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=truncate_error(e),
            sharding_fallback_reason=original_error,
        )

    if hasattr(output_struct, "shape"):
        output_shape = output_struct.shape
        output_dtype = str(output_struct.dtype)
        output_shapes: list[tuple[int, ...]] | None = [output_shape]
        output_dtypes: list[str] | None = [output_dtype]
    else:
        leaves = jax.tree.leaves(output_struct)
        shaped_leaves = [lf for lf in leaves if hasattr(lf, "shape")]
        if shaped_leaves:
            output_shape = shaped_leaves[0].shape
            output_dtype = str(shaped_leaves[0].dtype)
            output_shapes = [lf.shape for lf in shaped_leaves]
            output_dtypes = [str(lf.dtype) for lf in shaped_leaves]
        else:
            output_shape = None
            output_dtype = None
            output_shapes = None
            output_dtypes = None

    intermediates = _extract_intermediates(fn, abstract_inputs, env)

    return TraceResult(
        function_name=getattr(fn, "__name__", "<unknown>"),
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=intermediates,
        error=None,
        input_shapes={name: struct.shape for name, struct in abstract_inputs.items()},
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        sharding_fallback_reason=original_error,
    )


def trace_function(
    fn: Callable[..., Any],
    params: dict[str, ShapeSpec],
    env: DimEnv,
    mesh_config: dict[str, int] | None = None,
    axis_rules: dict[str, str] | None = None,
) -> TraceResult:
    """Trace a function using jax.eval_shape to get output shapes.

    Args:
        fn: The function to trace.
        params: Map of parameter name to ShapeSpec (from jaxtyping annotations).
        env: DimEnv for symbolic sizing.
        mesh_config: Optional mesh axis name -> size mapping for sharded tracing.
        axis_rules: Optional logical -> physical axis name mapping.

    Returns:
        TraceResult with output shape, intermediates, and any errors.
    """
    has_sharding = bool(mesh_config and any(spec.has_sharding for spec in params.values()))

    # Build mesh once for reuse in both input construction and tracing context
    abstract_mesh = _build_abstract_mesh(mesh_config) if has_sharding else None  # type: ignore[arg-type]

    # Build abstract inputs
    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for name, spec in params.items():
        if spec.is_any_shape:
            continue
        if has_sharding:
            abstract_inputs[name] = _build_sharded_abstract_input(
                spec,
                env,
                mesh_config,  # type: ignore[arg-type]
                axis_rules,
            )
        else:
            abstract_inputs[name] = _build_abstract_input(spec, env)

    # Run eval_shape — with mesh context when sharding is active
    try:

        def wrapper(**kwargs: Any) -> Any:
            return fn(**kwargs)

        if abstract_mesh is not None:
            from jax._src.mesh import use_abstract_mesh

            with use_abstract_mesh(abstract_mesh):
                output_struct = jax.eval_shape(wrapper, **abstract_inputs)
        else:
            output_struct = jax.eval_shape(wrapper, **abstract_inputs)
    except Exception as e:
        if has_sharding:
            return _trace_fallback_unsharded(fn, params, env, truncate_error(e))
        return TraceResult(
            function_name=getattr(fn, "__name__", "<unknown>"),
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=truncate_error(e),
        )

    # Extract output shape(s)
    output_shapes: list[tuple[int, ...]] | None = None
    output_dtypes: list[str] | None = None
    output_sharding = None

    if hasattr(output_struct, "shape"):
        output_shape = output_struct.shape
        output_dtype = str(output_struct.dtype)
        output_shapes = [output_shape]
        output_dtypes = [output_dtype]
        if has_sharding:
            output_sharding = getattr(output_struct, "sharding", None)
    else:
        leaves = jax.tree.leaves(output_struct)
        shaped_leaves = [lf for lf in leaves if hasattr(lf, "shape")]
        if shaped_leaves:
            output_shape = shaped_leaves[0].shape
            output_dtype = str(shaped_leaves[0].dtype)
            output_shapes = [lf.shape for lf in shaped_leaves]
            output_dtypes = [str(lf.dtype) for lf in shaped_leaves]
            if has_sharding:
                output_sharding = getattr(shaped_leaves[0], "sharding", None)
        else:
            output_shape = None
            output_dtype = None

    # Extract intermediates via make_jaxpr
    if abstract_mesh is not None:
        from jax._src.mesh import use_abstract_mesh

        with use_abstract_mesh(abstract_mesh):
            intermediates = _extract_intermediates(fn, abstract_inputs, env)
    else:
        intermediates = _extract_intermediates(fn, abstract_inputs, env)

    return TraceResult(
        function_name=getattr(fn, "__name__", "<unknown>"),
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=intermediates,
        error=None,
        input_shapes={name: struct.shape for name, struct in abstract_inputs.items()},
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        output_sharding=output_sharding,
    )
