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
from jaxtyc.config import load_config
from jaxtyc.lsp import _state
from jaxtyc.lsp._util import debounce_seconds
from jaxtyc.lsp._util import uri_to_path
from jaxtyc.lsp.index import build_file_index
from jaxtyc.types import IntermediateShape

logger: logging.Logger = logging.getLogger(__name__)

server: LanguageServer = LanguageServer("jaxtyc", _pkg_version("jaxtyc"))


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

    # Convert to LSP diagnostics
    lsp_diagnostics: list[types.Diagnostic] = []
    for diag in result.diagnostics:
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
            )
        )

    _state.diagnostics_cache[uri] = lsp_diagnostics

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=uri,
            diagnostics=lsp_diagnostics,
        )
    )

    # Cache intermediates for hover
    all_intermediates: list[IntermediateShape] = []
    for trace in result.trace_results:
        all_intermediates.extend(trace.intermediates)
    _state.analysis_cache[uri] = all_intermediates

    # Extract function specs once -- reused for CodeLens and navigation index
    try:
        source_text = Path(file_path).read_text(encoding="utf-8") if source is None else source
        func_specs = extract_function_specs(source_text, file_path)
    except Exception:
        func_specs = []
        source_text = ""

    # Build navigation index
    try:
        file_index = build_file_index(source_text, file_path, uri, func_specs=func_specs)
        _state.workspace_index.update_file(file_index)
    except Exception:
        logger.debug("Failed to build navigation index for %s", file_path, exc_info=True)

    # Import DimEnv once for both CodeLens and hover enhancement
    from jaxtyc.analyzer.dim_env import DimEnv

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
                env_for_names = DimEnv()
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
        _state.codelens_cache[uri] = lenses
    except Exception:
        _state.codelens_cache[uri] = []

    # Cache DimEnv for hover enhancement
    try:
        hover_env = DimEnv()
        for spec in func_specs:
            for pspec in spec.params.values():
                hover_env.make_shape(pspec)
            if spec.return_spec is not None:
                hover_env.make_shape(spec.return_spec)
        _state.dim_env_cache[uri] = hover_env
    except Exception:
        pass

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
