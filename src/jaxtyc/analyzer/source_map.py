"""Extract source-mapped intermediate shapes from jax.make_jaxpr output."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import IntermediateShape

_JAX_INTERNAL_MARKERS = (
    "site-packages/jax",
    "site-packages/jaxlib",
    "jax/_src/",
    "jaxlib/",
)


def _is_jax_internal(file_name: str) -> bool:
    """Check if a file path belongs to JAX internals."""
    return any(marker in file_name for marker in _JAX_INTERNAL_MARKERS)


def _find_user_frame(frames: Any) -> tuple[str, int, int]:
    """Find the most specific user frame from a jaxpr traceback.

    Returns (file_name, line_num, col) where col is always 0
    since JAX frames don't expose column info.
    """
    for frame in reversed(frames):
        if not _is_jax_internal(frame.file_name):
            return frame.file_name, frame.line_num, 0
    return "", 0, 0


def extract_source_mapped_intermediates(
    fn: Callable[..., Any],
    abstract_inputs: dict[str, jax.ShapeDtypeStruct],
    env: DimEnv,
) -> list[IntermediateShape]:
    """Run make_jaxpr on a function and extract all intermediate shapes with source locations.

    Args:
        fn: The function to trace.
        abstract_inputs: Map of parameter name to ShapeDtypeStruct.
        env: DimEnv for resolving shapes back to dimension names.

    Returns:
        List of IntermediateShape with source file/line info for each operation.
    """
    intermediates: list[IntermediateShape] = []

    try:

        def wrapper(**kwargs: Any) -> Any:
            return fn(**kwargs)

        closed_jaxpr = jax.make_jaxpr(wrapper)(**abstract_inputs)
        jaxpr = closed_jaxpr.jaxpr

        for eqn in jaxpr.eqns:
            source_file, source_line, source_col = "", 0, 0
            if eqn.source_info and eqn.source_info.traceback:
                frames = eqn.source_info.traceback.frames
                if frames:
                    source_file, source_line, source_col = _find_user_frame(frames)

            for outvar in eqn.outvars:
                if hasattr(outvar, "aval") and hasattr(outvar.aval, "shape"):
                    aval = outvar.aval
                    intermediates.append(
                        IntermediateShape(
                            shape=aval.shape,
                            dtype=str(aval.dtype),
                            source_file=source_file,
                            source_line=source_line,
                            source_col=source_col,
                            named_shape=env.shape_to_names(aval.shape),
                            op_name=eqn.primitive.name,
                        )
                    )
    except Exception:
        pass

    return intermediates
