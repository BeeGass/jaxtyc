"""Trace functions with jax.eval_shape and jax.make_jaxpr to extract shapes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jax.typing import DTypeLike

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult

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


_JAX_INTERNAL_PATHS: tuple[str, ...] = (
    "jax/",
    "jaxlib/",
    "site-packages/jax",
    "site-packages/jaxlib",
)


def _is_jax_internal(file_name: str) -> bool:
    """Check if a file path is from JAX internals."""
    return any(part in file_name for part in _JAX_INTERNAL_PATHS)


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
                # Find the last user frame (skip JAX internals)
                for frame in reversed(frames):
                    if not _is_jax_internal(frame.file_name):
                        source_file = frame.file_name
                        source_line = frame.line_num
                        source_col = 0  # JAX frames don't expose column
                        break

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
                        )
                    )
    except Exception:
        # If make_jaxpr fails, we still have eval_shape results
        pass

    return intermediates


def trace_function(
    fn: Callable[..., Any],
    params: dict[str, ShapeSpec],
    env: DimEnv,
) -> TraceResult:
    """Trace a function using jax.eval_shape to get output shapes.

    Args:
        fn: The function to trace.
        params: Map of parameter name to ShapeSpec (from jaxtyping annotations).
        env: DimEnv for prime-based symbolic sizing.

    Returns:
        TraceResult with output shape, intermediates, and any errors.

    Example:
        >>> env = DimEnv()
        >>> spec = parse_shape_string("batch seq d_model", "float32")
        >>> result = trace_function(my_fn, {"x": spec}, env)
        >>> result.output_shape
        (2, 3, 5)
    """
    # Build abstract inputs
    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for name, spec in params.items():
        if spec.is_any_shape:
            continue
        abstract_inputs[name] = _build_abstract_input(spec, env)

    # Run eval_shape
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
            error=str(e),
        )

    # Extract output shape(s)
    output_shapes: list[tuple[int, ...]] | None = None
    output_dtypes: list[str] | None = None

    if hasattr(output_struct, "shape"):
        output_shape = output_struct.shape
        output_dtype = str(output_struct.dtype)
        output_shapes = [output_shape]
        output_dtypes = [output_dtype]
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

    # Extract intermediates via make_jaxpr
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
    )
