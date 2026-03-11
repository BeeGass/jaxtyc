"""pygls-based LSP server for jaxtyc shape checking."""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
import uuid
from importlib.metadata import version as _pkg_version
from pathlib import Path

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.config import filter_diagnostics
from jaxtyc.config import load_config
from jaxtyc.lsp import _state
from jaxtyc.lsp._util import debounce_seconds
from jaxtyc.lsp._util import uri_to_path
from jaxtyc.lsp.index import build_file_index
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape

logger: logging.Logger = logging.getLogger(__name__)

server: LanguageServer = LanguageServer(
    "jaxtyc",
    _pkg_version("jaxtyc"),
    text_document_sync_kind=types.TextDocumentSyncKind.Full,
)


def _collect_literal_dims(func_specs: list[FunctionShapeSpec]) -> frozenset[int]:
    """Pre-scan function specs to collect literal dimension values for reservation."""
    reserved: set[int] = set()
    for spec in func_specs:
        for pspec in spec.params.values():
            for dim in pspec.dims:
                if dim.kind == "fixed" and dim.size is not None:
                    reserved.add(dim.size)
        if spec.return_spec:
            for dim in spec.return_spec.dims:
                if dim.kind == "fixed" and dim.size is not None:
                    reserved.add(dim.size)
        if spec.return_specs:
            for rspec in spec.return_specs:
                for dim in rspec.dims:
                    if dim.kind == "fixed" and dim.size is not None:
                        reserved.add(dim.size)
    return frozenset(reserved)


def _check_cross_file_calls(
    uri: str,
    func_specs: list[FunctionShapeSpec],
) -> list[types.Diagnostic]:
    """Check calls from this file to functions in other already-analyzed files."""
    from jaxtyc.analyzer.checker import check_call_site
    from jaxtyc.analyzer.dim_env import DimEnv

    file_index = _state.workspace_index.get_file(uri)
    if file_index is None:
        return []

    local_names = {s.name for s in func_specs}
    if file_index is not None:
        local_names |= {d.name for d in file_index.function_defs}
    extra_diags: list[types.Diagnostic] = []

    for call in file_index.call_sites:
        # Skip calls to functions in the same file (already checked by pipeline)
        if call.callee_name in local_names:
            continue

        callee_specs = _state.workspace_index.find_function_by_name(call.callee_name)
        if not callee_specs:
            continue

        callee_spec = callee_specs[0]
        callee_uri = _state.workspace_index.uri_for_file(callee_spec.file_path)
        if callee_uri is None:
            continue

        with _state.cache_lock:
            callee_traces = _state.trace_results_cache.get(callee_uri, {})
            callee_env_obj = _state.dim_env_cache.get(callee_uri)

        callee_trace = callee_traces.get(call.callee_name)
        if callee_trace is None or not callee_trace.success:
            continue
        if not isinstance(callee_env_obj, DimEnv):
            continue

        # Find the caller spec for this call site
        caller_name_short = call.caller_name.split(".")[-1]
        caller_spec = next((s for s in func_specs if s.name == caller_name_short), None)
        if caller_spec is None:
            continue

        cross_diags = check_call_site(call, caller_spec, callee_spec, callee_trace, callee_env_obj)
        for diag in cross_diags:
            severity = types.DiagnosticSeverity.Error
            if diag.severity == "warning":
                severity = types.DiagnosticSeverity.Warning
            elif diag.severity == "info":
                severity = types.DiagnosticSeverity.Information

            lsp_data = None
            if diag.data is not None:
                lsp_data = {
                    "expected_shape": list(diag.data.expected_shape)
                    if diag.data.expected_shape
                    else None,
                    "actual_shape": list(diag.data.actual_shape)
                    if diag.data.actual_shape
                    else None,
                    "expected_named": list(diag.data.expected_named)
                    if diag.data.expected_named
                    else None,
                    "actual_named": list(diag.data.actual_named)
                    if diag.data.actual_named
                    else None,
                    "dim_name_mapping": diag.data.dim_name_mapping,
                    "suggested_fix": diag.data.suggested_fix,
                    "rule": diag.data.rule,
                }

            related_info: list[types.DiagnosticRelatedInformation] | None = None
            if diag.data is not None and diag.data.related_locations:
                from pathlib import PurePosixPath

                related_info = [
                    types.DiagnosticRelatedInformation(
                        location=types.Location(
                            uri=PurePosixPath(rl.file_path).as_uri(),
                            range=types.Range(
                                start=types.Position(line=max(0, rl.line - 1), character=rl.col),
                                end=types.Position(line=max(0, rl.line - 1), character=rl.end_col),
                            ),
                        ),
                        message=rl.message,
                    )
                    for rl in diag.data.related_locations
                ]

            extra_diags.append(
                types.Diagnostic(
                    range=types.Range(
                        start=types.Position(line=max(0, diag.line - 1), character=diag.col),
                        end=types.Position(
                            line=max(0, diag.line - 1),
                            character=diag.col + 1,
                        ),
                    ),
                    message=diag.message,
                    severity=severity,
                    source="jaxtyc",
                    code=diag.rule,
                    data=lsp_data,
                    related_information=related_info,
                )
            )

    return extra_diags


def _analyze_and_publish(ls: LanguageServer, uri: str, source: str | None = None) -> None:
    """Analyze a file and publish diagnostics.

    If source is provided, write it to a temp file for analysis (used for didChange
    when the editor hasn't saved yet). Otherwise, analyze from disk.
    """
    file_path = uri_to_path(uri)

    if not file_path.endswith(".py"):
        return

    # Start progress if client supports it
    token = None
    try:
        caps = ls.client_capabilities
        if caps and getattr(getattr(caps, "window", None), "work_done_progress", False):
            token = str(uuid.uuid4())
            ls.work_done_progress.create(token)
            ls.work_done_progress.begin(
                token,
                types.WorkDoneProgressBegin(
                    title="jaxtyc",
                    message=f"Analyzing {Path(file_path).name}...",
                ),
            )
    except Exception:
        token = None

    if source is not None:
        # Write in-memory content to temp file for analysis
        suffix = Path(file_path).suffix
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            result = analyze_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        # Override the file path in diagnostics to point to the real file
        patched_diags = []
        for d in result.diagnostics:
            patched_diags.append(
                type(d)(
                    file=file_path,
                    line=d.line,
                    col=d.col,
                    severity=d.severity,
                    message=d.message,
                    rule=d.rule,
                )
            )
        from jaxtyc.types import FileResult

        result = FileResult(
            file_path=file_path,
            functions_checked=result.functions_checked,
            diagnostics=patched_diags,
            trace_results=result.trace_results,
        )
    else:
        result = analyze_file(file_path)

    # Apply config-based filtering (severity threshold, ignore_rules)
    filtered_diags = filter_diagnostics(result.diagnostics, _state.config)

    # Convert to LSP diagnostics
    lsp_diagnostics: list[types.Diagnostic] = []
    for diag in filtered_diags:
        if diag.severity == "error":
            severity = types.DiagnosticSeverity.Error
        elif diag.severity == "warning":
            severity = types.DiagnosticSeverity.Warning
        else:
            severity = types.DiagnosticSeverity.Information

        lsp_data = None
        if diag.data is not None:
            lsp_data = {
                "expected_shape": list(diag.data.expected_shape)
                if diag.data.expected_shape
                else None,
                "actual_shape": list(diag.data.actual_shape) if diag.data.actual_shape else None,
                "expected_named": list(diag.data.expected_named)
                if diag.data.expected_named
                else None,
                "actual_named": list(diag.data.actual_named) if diag.data.actual_named else None,
                "dim_name_mapping": diag.data.dim_name_mapping,
                "suggested_fix": diag.data.suggested_fix,
                "rule": diag.data.rule,
            }

        related_info: list[types.DiagnosticRelatedInformation] | None = None
        if diag.data is not None and diag.data.related_locations:
            from pathlib import PurePosixPath

            related_info = []
            for rl in diag.data.related_locations:
                rl_uri = PurePosixPath(rl.file_path).as_uri()
                related_info.append(
                    types.DiagnosticRelatedInformation(
                        location=types.Location(
                            uri=rl_uri,
                            range=types.Range(
                                start=types.Position(line=max(0, rl.line - 1), character=rl.col),
                                end=types.Position(line=max(0, rl.line - 1), character=rl.end_col),
                            ),
                        ),
                        message=rl.message,
                    )
                )

        lsp_diagnostics.append(
            types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=max(0, diag.line - 1), character=diag.col),
                    end=types.Position(line=max(0, diag.line - 1), character=diag.col + 1),
                ),
                message=diag.message,
                severity=severity,
                source="jaxtyc",
                code=diag.rule,
                data=lsp_data,
                related_information=related_info,
            )
        )

    # Cache intermediates for hover
    all_intermediates: list[IntermediateShape] = []
    for trace in result.trace_results:
        all_intermediates.extend(trace.intermediates)

    # Extract function specs once -- reused for CodeLens and navigation index
    try:
        source_text = Path(file_path).read_text(encoding="utf-8") if source is None else source
        func_specs = extract_function_specs(source_text, file_path)
    except Exception:
        func_specs = []
        source_text = ""

    # Collect all known function names across workspace for cross-file call detection
    all_known: set[str] = set()
    for fi in _state.workspace_index.all_files():
        all_known.update(s.name for s in fi.function_specs)
        all_known.update(d.name for d in fi.function_defs)
    all_known.update(s.name for s in func_specs)

    # Build navigation index
    try:
        file_index = build_file_index(
            source_text,
            file_path,
            uri,
            func_specs=func_specs,
            extra_known_names=frozenset(all_known),
        )
        _state.workspace_index.update_file(file_index)
    except Exception:
        logger.debug("Failed to build navigation index for %s", file_path, exc_info=True)

    # Import DimEnv once for both CodeLens and hover enhancement
    from jaxtyc.analyzer.dim_env import DimEnv

    # Collect literal dim values to reserve them from prime assignment
    reserved = _collect_literal_dims(func_specs)

    # Build CodeLens cache from function specs + trace results
    try:
        lenses: list[tuple[int, str]] = []
        trace_by_name = {t.function_name: t for t in result.trace_results}

        for spec in func_specs:
            trace = trace_by_name.get(spec.name)
            if trace is None or not trace.success:
                continue
            param_parts = []
            for pname, pspec in spec.params.items():
                dim_names = ", ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
                param_parts.append(f"{pname}: ({dim_names})")
            ret_part = ""
            if trace.output_shape is not None:
                env_for_names = DimEnv(reserved=reserved)
                # Rebuild env with the same dim names as the spec
                for pspec in spec.params.values():
                    env_for_names.make_shape(pspec)
                out_names = env_for_names.shape_to_names(trace.output_shape)
                named_out = ", ".join(
                    n or str(s) for n, s in zip(out_names, trace.output_shape, strict=True)
                )
                ret_part = f" -> ({named_out})"
            title = f"shapes: {', '.join(param_parts)}{ret_part}"
            lenses.append((spec.lineno - 1, title))  # Convert to 0-indexed
    except Exception:
        lenses = []

    # Build DimEnv for hover enhancement
    hover_env = None
    try:
        hover_env = DimEnv(reserved=reserved)
        for spec in func_specs:
            for pspec in spec.params.values():
                hover_env.make_shape(pspec)
            if spec.return_spec is not None:
                hover_env.make_shape(spec.return_spec)
    except Exception:
        hover_env = None

    # Build error hints via divergence detection
    from jaxtyc.analyzer.divergence import find_divergence_points
    from jaxtyc.types import ErrorHintInfo

    error_hints: list[ErrorHintInfo] = []
    if hover_env is not None:
        trace_by_name = {t.function_name: t for t in result.trace_results}
        for spec in func_specs:
            trace = trace_by_name.get(spec.name)
            if trace is None or not trace.success:
                continue
            try:
                hints = find_divergence_points(spec, trace, hover_env)
                error_hints.extend(hints)
            except Exception:
                logger.debug("Failed to find divergence points for %s", spec.name, exc_info=True)

    # Synthesize return-line intermediates for functions whose trace succeeded
    # but produced no jaxpr-based intermediates (identity/passthrough functions,
    # or functions where make_jaxpr failed).  This ensures inlay hints appear
    # on the return line showing the resolved output shape.
    if hover_env is not None and source_text:
        import re

        _return_re = re.compile(r"^\s*return\b")
        source_lines = source_text.splitlines()
        covered_lines = {i.source_line for i in all_intermediates}
        trace_by_name_synth = {t.function_name: t for t in result.trace_results}

        for spec in func_specs:
            trace = trace_by_name_synth.get(spec.name)
            if trace is None or not trace.success or trace.output_shape is None:
                continue
            # Find return lines within this function's body
            start = spec.lineno  # 1-based def line
            end = spec.end_lineno if spec.end_lineno > 0 else len(source_lines)
            for line_idx in range(start, min(end, len(source_lines))):
                line_num = line_idx + 1  # 1-based
                if line_num in covered_lines:
                    continue
                if _return_re.match(source_lines[line_idx]):
                    named = hover_env.shape_to_names(trace.output_shape)
                    all_intermediates.append(
                        IntermediateShape(
                            shape=trace.output_shape,
                            dtype=trace.output_dtype or "float32",
                            source_file=file_path,
                            source_line=line_num,
                            source_col=0,
                            named_shape=named,
                            op_name="return",
                        )
                    )
                    covered_lines.add(line_num)

    # Build trace results index for cross-file and call-site features
    trace_results_by_name = {t.function_name: t for t in result.trace_results}

    # Batch-write all caches atomically
    with _state.cache_lock:
        _state.diagnostics_cache[uri] = lsp_diagnostics
        _state.analysis_cache[uri] = all_intermediates
        _state.codelens_cache[uri] = lenses
        _state.error_hints_cache[uri] = error_hints
        _state.source_cache[uri] = source_text
        _state.trace_results_cache[uri] = trace_results_by_name
        if hover_env is not None:
            _state.dim_env_cache[uri] = hover_env

    # Cross-file checking phase
    cross_file_diags = _check_cross_file_calls(uri, func_specs)
    if cross_file_diags:
        lsp_diagnostics.extend(cross_file_diags)
        with _state.cache_lock:
            _state.diagnostics_cache[uri] = lsp_diagnostics

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=lsp_diagnostics,
        )
    )

    # End progress
    if token is not None:
        with contextlib.suppress(Exception):
            ls.work_done_progress.end(
                token,
                types.WorkDoneProgressEnd(message="Analysis complete"),
            )


def _schedule_debounced_analysis(ls: LanguageServer, uri: str, source: str) -> None:
    """Schedule a debounced analysis for a URI. Cancels any pending timer."""
    with _state.debounce_lock:
        existing = _state.debounce_timers.pop(uri, None)
        if existing is not None:
            existing.cancel()

        timer = threading.Timer(
            debounce_seconds(),
            _analyze_and_publish,
            args=(ls, uri, source),
        )
        timer.daemon = True
        _state.debounce_timers[uri] = timer
        timer.start()


@server.feature(types.INITIALIZED)
def on_initialized(ls: LanguageServer, params: types.InitializedParams) -> None:
    """Load config from workspace root on initialization."""
    root_uri = ls.workspace.root_uri
    if root_uri:
        root_path = uri_to_path(root_uri)
        _state.config = load_config(root_path)
        logger.info("Loaded config from %s: debounce_ms=%d", root_path, _state.config.debounce_ms)


def start_lsp() -> None:
    """Start the LSP server on stdio."""
    server.start_io()


# Import handler modules to register their @server.feature() decorators.
# These must be at the bottom to avoid circular imports.
import jaxtyc.lsp._code_actions  # noqa: F401, E402
import jaxtyc.lsp._completion  # noqa: F401, E402
import jaxtyc.lsp._configuration  # noqa: F401, E402
import jaxtyc.lsp._diagnostics  # noqa: F401, E402
import jaxtyc.lsp._folding  # noqa: F401, E402
import jaxtyc.lsp._inlay_hints  # noqa: F401, E402
import jaxtyc.lsp._linked_editing  # noqa: F401, E402
import jaxtyc.lsp._navigation  # noqa: F401, E402
import jaxtyc.lsp._semantic_tokens  # noqa: F401, E402
import jaxtyc.lsp._signature_help  # noqa: F401, E402
