"""End-to-end analysis pipeline: parse annotations, import, trace, check."""

from __future__ import annotations

from pathlib import Path

from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.checker import check_function
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.importer import import_module_from_path
from jaxtyc.analyzer.tracer import trace_function
from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult
from jaxtyc.types import TraceResult


def analyze_file(file_path: str) -> FileResult:
    """Analyze all jaxtyping-annotated functions in a Python file.

    1. Parse the file's AST to extract jaxtyping annotations
    2. Import the module to get live function objects
    3. Trace each function with jax.eval_shape
    4. Compare actual vs expected shapes
    5. Return diagnostics for any mismatches
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

        env = DimEnv()
        trace = trace_function(fn, func_spec.params, env)
        trace_results.append(trace)
        functions_checked += 1

        # Check shapes against annotations
        diags = check_function(func_spec, trace, env)
        diagnostics.extend(diags)

    return FileResult(
        file_path=file_path,
        functions_checked=functions_checked,
        diagnostics=diagnostics,
        trace_results=trace_results,
    )


def _resolve_function(module: object, name: str, class_name: str | None) -> object | None:
    """Resolve a function object from a module, optionally within a class."""
    if class_name is not None:
        cls = getattr(module, class_name, None)
        if cls is None:
            return None
        return getattr(cls, name, None)
    return getattr(module, name, None)
