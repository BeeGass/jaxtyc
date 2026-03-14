"""End-to-end analysis pipeline: parse annotations, import, trace, check."""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import jax

from jax.typing import DTypeLike

from jaxtyc.analyzer._errors import truncate_error
from jaxtyc.analyzer.annotations import extract_call_sites
from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.checker import check_call_site
from jaxtyc.analyzer.checker import check_function
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.importer import import_module_from_path
from jaxtyc.analyzer.mesh_resolver import MeshInfo
from jaxtyc.analyzer.mesh_resolver import resolve_mesh
from jaxtyc.analyzer.sharding_checker import check_annotation_sharding
from jaxtyc.analyzer.sharding_checker import check_mesh_axes
from jaxtyc.analyzer.sharding_checker import check_sharding
from jaxtyc.analyzer.sharding_checker import check_sharding_propagation
from jaxtyc.analyzer.suppressions import extract_suppressions
from jaxtyc.analyzer.suppressions import filter_inline_suppressions
from jaxtyc.analyzer.tracer import trace_function
from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import TraceResult

logger = logging.getLogger(__name__)


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
                message=f"Could not read file: {truncate_error(e)}",
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

    # Infer mesh shape and axis rules from source AST (consumed by sharded tracing)
    try:
        tree = ast.parse(source)
        mesh_info = resolve_mesh(tree)
    except SyntaxError:
        mesh_info = MeshInfo()

    mesh_config = mesh_info.mesh or None
    axis_rules = dict(mesh_info.axis_rules) if mesh_info.axis_rules else {}

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
                message=f"Could not import module: {truncate_error(e)}",
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
    # to the same symbolic value, enabling cross-function consistency checking.
    env = DimEnv()

    # Trace and check each annotated function
    traced: dict[str, tuple[FunctionShapeSpec, TraceResult]] = {}
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
                trace = _trace_nnx_method(
                    cls, func_spec, env, mesh_config=mesh_config, axis_rules=axis_rules
                )
                trace_results.append(trace)
                traced[func_spec.name] = (func_spec, trace)
                functions_checked += 1
                diags = check_function(func_spec, trace, env)
                diagnostics.extend(diags)
                diagnostics.extend(check_sharding(trace.intermediates, func_spec, file_path))
                continue
            if cls is not None and _is_eqx_module(cls):
                trace = _trace_eqx_method(
                    cls, func_spec, env, mesh_config=mesh_config, axis_rules=axis_rules
                )
                trace_results.append(trace)
                traced[func_spec.name] = (func_spec, trace)
                functions_checked += 1
                diags = check_function(func_spec, trace, env)
                diagnostics.extend(diags)
                diagnostics.extend(check_sharding(trace.intermediates, func_spec, file_path))
                continue

        # Check annotation-level sharding consistency (no tracing needed)
        annotation_sharding_diags = check_annotation_sharding(func_spec, file_path)
        diagnostics.extend(annotation_sharding_diags)

        # Check mesh axis references
        if mesh_config:
            mesh_diags = check_mesh_axes(func_spec, file_path, mesh_config, axis_rules)
            diagnostics.extend(mesh_diags)

        # If annotation-level sharding has errors, skip sharded tracing to avoid
        # confusing trace-error from conflicting concrete sizes. Fall back to
        # unsharded tracing so shape checks still run.
        effective_mesh = mesh_config
        if annotation_sharding_diags and mesh_config:
            effective_mesh = None

        trace = trace_function(
            fn, func_spec.params, env, mesh_config=effective_mesh, axis_rules=axis_rules
        )

        # Emit warning if sharded tracing fell back to unsharded
        if trace.sharding_fallback_reason is not None:
            diagnostics.append(
                Diagnostic(
                    file=file_path,
                    line=func_spec.lineno,
                    col=func_spec.col_offset,
                    severity="warning",
                    message=(
                        f"Sharded tracing failed for `{func_spec.name}`, fell back to "
                        f"unsharded: {trace.sharding_fallback_reason}"
                    ),
                    rule="trace-error",
                )
            )

        trace_results.append(trace)
        traced[func_spec.name] = (func_spec, trace)
        functions_checked += 1

        # Check shapes against annotations
        diags = check_function(func_spec, trace, env)
        diagnostics.extend(diags)

        # Check sharding constraints from make_jaxpr intermediates
        sharding_diags = check_sharding(trace.intermediates, func_spec, file_path)
        diagnostics.extend(sharding_diags)

        # Check propagated output sharding against return annotation
        if trace.output_sharding is not None:
            diagnostics.extend(
                check_sharding_propagation(
                    trace.output_sharding, func_spec.return_spec, func_spec, file_path
                )
            )

    # Cross-function shape propagation

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


_SKIP_INIT_PARAMS: frozenset[str] = frozenset({"self", "rngs", "key", "rng", "parent", "name"})


def _collect_dim_kwargs(
    cls: type,
    func_spec: FunctionShapeSpec,
    env: DimEnv,
) -> dict[str, int]:
    """Build constructor kwargs by combining annotation dims with constructor int params.

    Two sources of dimension information:

    1. **Annotation dims** — named dimensions from the method's param and return
       annotations.  These get their prime from *env* and are used for shape
       checking.
    2. **Constructor int params** — ``int``-typed parameters in ``__init__`` that
       don't appear in any annotation.  These also get primes via *env* so the
       model can be constructed and the primes can be reverse-mapped in error
       messages.

    This ensures construction succeeds even when annotations don't reference
    every constructor dimension (e.g. a buggy return annotation that omits
    ``d_out``).
    """
    import inspect

    # 1. Collect named dims from method annotations (concrete ints for construction)
    annotation_dims: dict[str, int] = {}
    specs = list(func_spec.params.values())
    if func_spec.return_spec is not None:
        specs.append(func_spec.return_spec)
    if func_spec.return_specs:
        specs.extend(func_spec.return_specs)
    for spec in specs:
        for dim in spec.dims:
            if dim.kind == "named" and dim.name is not None:
                annotation_dims.setdefault(dim.name, env.get_concrete_size(dim.name))

    # 2. Inspect constructor for all int-typed params
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return annotation_dims

    result: dict[str, int] = {}
    for pname, param in sig.parameters.items():
        if pname in _SKIP_INIT_PARAMS:
            continue
        if param.kind not in (
            param.POSITIONAL_OR_KEYWORD,
            param.KEYWORD_ONLY,
            param.POSITIONAL_ONLY,
        ):
            continue

        if pname in annotation_dims:
            # Matches an annotation dim — use its prime
            result[pname] = annotation_dims[pname]
        elif (
            param.annotation is int or param.annotation == "int"
        ) and param.default is inspect.Parameter.empty:
            # Required int param not in annotations — assign a concrete size for reverse-mapping.
            # Optional int params (with defaults) keep their defaults to avoid
            # breaking divisibility assertions like `assert features % num_head == 0`.
            result[pname] = env.get_concrete_size(pname)

    return result


def _trace_nnx_method(
    cls: type,
    func_spec: FunctionShapeSpec,
    env: DimEnv,
    mesh_config: dict[str, int] | None = None,
    axis_rules: dict[str, str] | None = None,
) -> TraceResult:
    """Trace a Flax NNX module method using a concrete model with split/merge."""
    import jax
    from flax import nnx

    from jaxtyc.analyzer.tracer import _extract_intermediates

    # NNX modules need concrete int sizes for construction. Use concrete
    # shapes for abstract inputs so they match the module's internal weights.
    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for pname, pspec in func_spec.params.items():
        if pspec.is_any_shape:
            continue
        shape = env.make_concrete_shape(pspec)
        dtype = _resolve_jax_dtype(pspec.dtype)
        abstract_inputs[pname] = jax.ShapeDtypeStruct(shape, dtype)

    # Build constructor kwargs from dimension names in annotations.
    # The model's internal params must match the concrete abstract inputs
    # so that eval_shape and make_jaxpr produce correct shapes.
    dim_sizes = _collect_dim_kwargs(cls, func_spec, env)

    # Create concrete model instance (nnx.eval_shape abstract models fail
    # on self.kernel[...] in newer Flax versions)
    try:
        concrete_model = cls(**dim_sizes, rngs=nnx.Rngs(0))
    except Exception:
        try:
            concrete_model = cls(rngs=nnx.Rngs(0))
        except Exception as e:
            return TraceResult(
                function_name=func_spec.name,
                output_shape=None,
                output_dtype=None,
                intermediates=[],
                error=f"Could not instantiate NNX module {cls.__name__}: {truncate_error(e)}",
            )

    # Build pure function via split/merge for jax.eval_shape and make_jaxpr
    graphdef, state = nnx.split(concrete_model)

    def pure_fn(**kwargs: Any) -> Any:
        model = nnx.merge(graphdef, state)
        return getattr(model, func_spec.name)(**kwargs)

    # Build mesh context if sharding is active
    has_sharding = bool(
        mesh_config and any(spec.has_sharding for spec in func_spec.params.values())
    )
    abstract_mesh = None
    if has_sharding and mesh_config:
        from jaxtyc.analyzer.tracer import _build_abstract_mesh

        abstract_mesh = _build_abstract_mesh(mesh_config)

    # Trace for output shape
    try:
        if abstract_mesh is not None:
            from jax._src.mesh import use_abstract_mesh

            with use_abstract_mesh(abstract_mesh):
                output_struct = jax.eval_shape(pure_fn, **abstract_inputs)
        else:
            output_struct = jax.eval_shape(pure_fn, **abstract_inputs)
    except Exception as e:
        return TraceResult(
            function_name=func_spec.name,
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=truncate_error(e),
        )

    # Extract output shape
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

    # Extract intermediates via make_jaxpr (graceful degradation)
    if abstract_mesh is not None:
        from jax._src.mesh import use_abstract_mesh

        with use_abstract_mesh(abstract_mesh):
            intermediates = _extract_intermediates(pure_fn, abstract_inputs, env)
    else:
        intermediates = _extract_intermediates(pure_fn, abstract_inputs, env)

    return TraceResult(
        function_name=func_spec.name,
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=intermediates,
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
    mesh_config: dict[str, int] | None = None,
    axis_rules: dict[str, str] | None = None,
) -> TraceResult:
    """Trace an equinox module method using jax.eval_shape with a bound method."""
    import jax

    from jaxtyc.analyzer.tracer import _extract_intermediates

    # Equinox modules need concrete int sizes — use make_concrete_shape
    abstract_inputs: dict[str, jax.ShapeDtypeStruct] = {}
    for pname, pspec in func_spec.params.items():
        if pspec.is_any_shape:
            continue
        shape = env.make_concrete_shape(pspec)
        dtype = _resolve_jax_dtype(pspec.dtype)
        abstract_inputs[pname] = jax.ShapeDtypeStruct(shape, dtype)

    # Build constructor kwargs from annotation dims so model params
    # match the concrete abstract inputs
    dim_sizes = _collect_dim_kwargs(cls, func_spec, env)

    # Build mesh context if sharding is active
    has_sharding = bool(
        mesh_config and any(spec.has_sharding for spec in func_spec.params.values())
    )
    abstract_mesh = None
    if has_sharding and mesh_config:
        from jaxtyc.analyzer.tracer import _build_abstract_mesh

        abstract_mesh = _build_abstract_mesh(mesh_config)

    # Equinox modules are pytrees — instantiate with dimension-matched params
    try:
        key = jax.random.key(0)
        try:
            model = cls(**dim_sizes, key=key)
        except Exception:
            try:
                model = cls(key=key)
            except Exception as e:
                return TraceResult(
                    function_name=func_spec.name,
                    output_shape=None,
                    output_dtype=None,
                    intermediates=[],
                    error=f"Could not instantiate equinox module {cls.__name__}: {truncate_error(e)}",
                )

        method = getattr(model, func_spec.name)

        def wrapper(**kwargs: Any) -> Any:
            return method(**kwargs)

        if abstract_mesh is not None:
            from jax._src.mesh import use_abstract_mesh

            with use_abstract_mesh(abstract_mesh):
                output_struct = jax.eval_shape(wrapper, **abstract_inputs)
        else:
            output_struct = jax.eval_shape(wrapper, **abstract_inputs)
    except Exception as e:
        return TraceResult(
            function_name=func_spec.name,
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error=truncate_error(e),
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

    # Extract intermediates via make_jaxpr (graceful degradation)
    if abstract_mesh is not None:
        from jax._src.mesh import use_abstract_mesh

        with use_abstract_mesh(abstract_mesh):
            intermediates = _extract_intermediates(wrapper, abstract_inputs, env)
    else:
        intermediates = _extract_intermediates(wrapper, abstract_inputs, env)

    return TraceResult(
        function_name=func_spec.name,
        output_shape=output_shape,
        output_dtype=output_dtype,
        intermediates=intermediates,
        error=None,
    )
