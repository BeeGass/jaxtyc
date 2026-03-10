"""End-to-end analysis pipeline: parse annotations, import, trace, check."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import jax

from jax.typing import DTypeLike

from jaxtyc.analyzer.annotations import extract_call_sites
from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.checker import check_call_site
from jaxtyc.analyzer.checker import check_function
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.importer import import_module_from_path
from jaxtyc.analyzer.suppressions import extract_suppressions
from jaxtyc.analyzer.suppressions import filter_inline_suppressions
from jaxtyc.analyzer.tracer import trace_function
from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import TraceResult


def analyze_file(file_path: str) -> FileResult:
    """Analyze all jaxtyping-annotated functions in a Python file.

    Pipeline steps:
        1. Parse the file's AST to extract jaxtyping annotations
        2. Import the module to get live function objects
        3. Trace each function with jax.eval_shape
        4. Compare actual vs expected shapes
        5. Return diagnostics for any mismatches

    Args:
        file_path: Absolute path to the Python source file to analyze.

    Returns:
        FileResult containing per-function trace results and any diagnostics
        (shape mismatches, rank errors, import/read failures).

    Example:
        >>> result = analyze_file("/path/to/model.py")
        >>> for d in result.diagnostics:
        ...     print(f"{d.severity}: {d.message}")
    """
    path = Path(file_path)
    diagnostics: list[Diagnostic] = []
    trace_results: list[TraceResult] = []

    # Read source
    if not path.exists():
        diagnostics.append(
            Diagnostic(
                file=file_path,
                line=0,
                col=0,
                severity="info",
                message=f"File not found: {file_path}",
                rule="file-not-found",
            )
        )
        return FileResult(
            file_path=file_path,
            functions_checked=0,
            diagnostics=diagnostics,
            trace_results=trace_results,
        )

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        diagnostics.append(
            Diagnostic(
                file=file_path,
                line=0,
                col=0,
                severity="info",
                message=f"Could not read file: {e}",
                rule="read-error",
            )
        )
        return FileResult(
            file_path=file_path,
            functions_checked=0,
            diagnostics=diagnostics,
            trace_results=trace_results,
        )

    # Parse annotations from AST
    func_specs = extract_function_specs(source, file_path)
    if not func_specs:
        return FileResult(
            file_path=file_path,
            functions_checked=0,
            diagnostics=diagnostics,
            trace_results=trace_results,
        )

    # Import module to get live function objects
    try:
        module = import_module_from_path(file_path)
    except Exception as e:
        diagnostics.append(
            Diagnostic(
                file=file_path,
                line=0,
                col=0,
                severity="info",
                message=f"Could not import module: {e}",
                rule="import-error",
            )
        )
        return FileResult(
            file_path=file_path,
            functions_checked=0,
            diagnostics=diagnostics,
            trace_results=trace_results,
        )

    # Shared DimEnv across all functions in the file — same dim name always maps
    # to the same prime, enabling cross-function consistency checking.
    env = DimEnv()

    # Trace and check each annotated function
    functions_checked = 0
    for func_spec in func_specs:
        # Resolve the function object from the module
        fn = _resolve_function(module, func_spec.name, func_spec.class_name)
        if fn is None:
            diagnostics.append(
                Diagnostic(
                    file=file_path,
                    line=func_spec.lineno,
                    col=func_spec.col_offset,
                    severity="info",
                    message=f"Could not resolve function `{func_spec.name}`",
                    rule="resolve-error",
                )
            )
            continue

        # Skip if no params to trace with, or all params are any-shape
        traceable_params = {k: v for k, v in func_spec.params.items() if not v.is_any_shape}
        if not traceable_params and not func_spec.params:
            continue
        if not traceable_params:
            # All params are any-shape — nothing to trace
            continue

        # Detect NNX/equinox module methods and use specialized tracing
        if func_spec.is_method and func_spec.class_name is not None:
            cls = getattr(module, func_spec.class_name, None)
            if cls is not None and _is_nnx_module(cls):
                trace = _trace_nnx_method(cls, func_spec, env)
                trace_results.append(trace)
                functions_checked += 1
                diags = check_function(func_spec, trace, env)
                diagnostics.extend(diags)
                continue
            if cls is not None and _is_eqx_module(cls):
                trace = _trace_eqx_method(cls, func_spec, env)
                trace_results.append(trace)
                functions_checked += 1
                diags = check_function(func_spec, trace, env)
                diagnostics.extend(diags)
                continue

        trace = trace_function(fn, func_spec.params, env)
        trace_results.append(trace)
        functions_checked += 1

        # Check shapes against annotations
        diags = check_function(func_spec, trace, env)
        diagnostics.extend(diags)

    # Cross-function shape propagation
    traced: dict[str, tuple[FunctionShapeSpec, TraceResult]] = {}
    for spec, trace in zip(func_specs, trace_results, strict=False):
        traced[spec.name] = (spec, trace)

    known_functions = {s.name for s in func_specs}
    call_sites = extract_call_sites(source, file_path, known_functions)
    for call in call_sites:
        callee_entry = traced.get(call.callee_name)
        if callee_entry is None:
            continue
        callee_spec, callee_trace = callee_entry
        caller_entry = traced.get(call.caller_name.split(".")[-1])
        if caller_entry is None:
            continue
        caller_spec, _ = caller_entry
        cross_diags = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        diagnostics.extend(cross_diags)

    # Apply inline suppression comments
    suppressions = extract_suppressions(source)
    if suppressions:
        diagnostics = filter_inline_suppressions(diagnostics, suppressions)

    return FileResult(
        file_path=file_path,
        functions_checked=functions_checked,
        diagnostics=diagnostics,
        trace_results=trace_results,
    )


def _resolve_function(
    module: object, name: str, class_name: str | None
) -> Callable[..., Any] | None:
    """Resolve a function object from a module, optionally within a class."""
    if class_name is not None:
        cls = getattr(module, class_name, None)
        if cls is None:
            return None
        return getattr(cls, name, None)
    return getattr(module, name, None)


def _is_nnx_module(cls: type) -> bool:
    """Check if a class is a Flax NNX module."""
    try:
        from flax import nnx

        return isinstance(cls, type) and issubclass(cls, nnx.Module)
    except ImportError:
        return False


def _is_eqx_module(cls: type) -> bool:
    """Check if a class is an equinox module."""
    try:
        import equinox as eqx

        return isinstance(cls, type) and issubclass(cls, eqx.Module)
    except ImportError:
        return False


def _trace_nnx_method(
    cls: type,
    func_spec: FunctionShapeSpec,
    env: DimEnv,
) -> TraceResult:
    """Trace a Flax NNX module method using nnx.eval_shape."""
    from flax import nnx

    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for pname, pspec in func_spec.params.items():
        if pspec.is_any_shape:
            continue
        import jax

        shape = env.make_shape(pspec)
        dtype = _resolve_jax_dtype(pspec.dtype)
        abstract_inputs[pname] = jax.ShapeDtypeStruct(shape, dtype)

    try:
        # Create abstract module instance via nnx.eval_shape
        abstract_model = nnx.eval_shape(lambda: cls(d_in=2, d_out=3, rngs=nnx.Rngs(0)))
    except Exception:
        # Fallback: try common constructor patterns
        try:
            abstract_model = nnx.eval_shape(lambda: cls(rngs=nnx.Rngs(0)))
        except Exception as e:
            return TraceResult(
                function_name=func_spec.name,
                output_shape=None,
                output_dtype=None,
                intermediates=[],
                error=f"Could not instantiate NNX module {cls.__name__}: {e}",
            )

    # Trace the method with nnx.eval_shape
    try:
        method = getattr(abstract_model, func_spec.name)
        output_struct = nnx.eval_shape(lambda: method(**abstract_inputs))
    except Exception as e:
        return TraceResult(
            function_name=func_spec.name,
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=str(e),
        )

    # Extract output shape
    if hasattr(output_struct, "shape"):
        output_shape = output_struct.shape
        output_dtype = str(output_struct.dtype)
    else:
        import jax

        leaves = jax.tree.leaves(output_struct)
        if leaves and hasattr(leaves[0], "shape"):
            output_shape = leaves[0].shape
            output_dtype = str(leaves[0].dtype)
        else:
            output_shape = None
            output_dtype = None

    return TraceResult(
        function_name=func_spec.name,
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=[],
        error=None,
    )


def _resolve_jax_dtype(dtype_str: str) -> DTypeLike:
    """Resolve a jaxtyc dtype string to a JAX dtype."""
    from jaxtyc.analyzer.tracer import _resolve_jax_dtype as _resolve

    return _resolve(dtype_str)


def _trace_eqx_method(
    cls: type,
    func_spec: FunctionShapeSpec,
    env: DimEnv,
) -> TraceResult:
    """Trace an equinox module method using jax.eval_shape with a bound method."""
    import jax

    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for pname, pspec in func_spec.params.items():
        if pspec.is_any_shape:
            continue
        shape = env.make_shape(pspec)
        dtype = _resolve_jax_dtype(pspec.dtype)
        abstract_inputs[pname] = jax.ShapeDtypeStruct(shape, dtype)

    # Equinox modules are pytrees — instantiate one with concrete params,
    # then trace __call__ with jax.eval_shape passing the model as a static arg
    try:
        key = jax.random.key(0)
        # Try common constructor patterns
        try:
            model = cls(d_in=2, d_out=3, key=key)
        except Exception:
            try:
                model = cls(key=key)
            except Exception as e:
                return TraceResult(
                    function_name=func_spec.name,
                    output_shape=None,
                    output_dtype=None,
                    intermediates=[],
                    error=f"Could not instantiate equinox module {cls.__name__}: {e}",
                )

        method = getattr(model, func_spec.name)

        def wrapper(**kwargs: Any) -> Any:
            return method(**kwargs)

        output_struct = jax.eval_shape(wrapper, **abstract_inputs)
    except Exception as e:
        return TraceResult(
            function_name=func_spec.name,
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=str(e),
        )

    if hasattr(output_struct, "shape"):
        output_shape = output_struct.shape
        output_dtype = str(output_struct.dtype)
    else:
        leaves = jax.tree.leaves(output_struct)
        if leaves and hasattr(leaves[0], "shape"):
            output_shape = leaves[0].shape
            output_dtype = str(leaves[0].dtype)
        else:
            output_shape = None
            output_dtype = None

    return TraceResult(
        function_name=func_spec.name,
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=[],
        error=None,
    )
