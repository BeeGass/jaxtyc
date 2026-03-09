"""pygls-based LSP server for jaxtyc shape checking."""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.analyzer.annotations import extract_function_specs
from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.config import JaxtycConfig
from jaxtyc.config import load_config
from jaxtyc.lsp.index import WorkspaceIndex
from jaxtyc.lsp.index import build_file_index
from jaxtyc.types import DimLocation
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape

logger = logging.getLogger(__name__)

server = LanguageServer("jaxtyc", "v0.1.0")

# Cache of analysis results per URI
_analysis_cache: dict[str, list[IntermediateShape]] = {}

# Cache of CodeLens data per URI: list of (0-indexed line, title text)
_codelens_cache: dict[str, list[tuple[int, str]]] = {}

# Debounce state: pending timers per URI
_debounce_timers: dict[str, threading.Timer] = {}
_debounce_lock = threading.Lock()

# Server config — loaded from workspace root on initialize
_config: JaxtycConfig = JaxtycConfig()

# Workspace-level index for navigation features
_workspace_index = WorkspaceIndex()


def _debounce_seconds() -> float:
    """Get the debounce delay in seconds from config."""
    return _config.debounce_ms / 1000.0


def _uri_to_path(uri: str) -> str:
    """Convert a file URI to a filesystem path."""
    parsed = urlparse(uri)
    return unquote(parsed.path)


def _analyze_and_publish(ls: LanguageServer, uri: str, source: str | None = None) -> None:
    """Analyze a file and publish diagnostics.

    If source is provided, write it to a temp file for analysis (used for didChange
    when the editor hasn't saved yet). Otherwise, analyze from disk.
    """
    file_path = _uri_to_path(uri)

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
            )
        )

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
    _analysis_cache[uri] = all_intermediates

    # Extract function specs once — reused for CodeLens and navigation index
    try:
        source_text = Path(file_path).read_text(encoding="utf-8") if source is None else source
        func_specs = extract_function_specs(source_text, file_path)
    except Exception:
        func_specs = []
        source_text = ""

    # Build navigation index
    try:
        file_index = build_file_index(source_text, file_path, uri, func_specs=func_specs)
        _workspace_index.update_file(file_index)
    except Exception:
        logger.debug("Failed to build navigation index for %s", file_path, exc_info=True)

    # Build CodeLens cache from function specs + trace results
    try:
        lenses: list[tuple[int, str]] = []
        trace_by_name = {t.function_name: t for t in result.trace_results}
        from jaxtyc.analyzer.dim_env import DimEnv

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
        _codelens_cache[uri] = lenses
    except Exception:
        _codelens_cache[uri] = []

    # End progress
    if token is not None:
        with contextlib.suppress(Exception):
            ls.work_done_progress.end(
                token,
                types.WorkDoneProgressEnd(message="Analysis complete"),
            )


def _schedule_debounced_analysis(ls: LanguageServer, uri: str, source: str) -> None:
    """Schedule a debounced analysis for a URI. Cancels any pending timer."""
    with _debounce_lock:
        existing = _debounce_timers.pop(uri, None)
        if existing is not None:
            existing.cancel()

        timer = threading.Timer(
            _debounce_seconds(),
            _analyze_and_publish,
            args=(ls, uri, source),
        )
        timer.daemon = True
        _debounce_timers[uri] = timer
        timer.start()


@server.feature(types.INITIALIZED)
def on_initialized(ls: LanguageServer, params: types.InitializedParams) -> None:
    """Load config from workspace root on initialization."""
    global _config  # noqa: PLW0603
    root_uri = ls.workspace.root_uri
    if root_uri:
        root_path = _uri_to_path(root_uri)
        _config = load_config(root_path)
        logger.info("Loaded config from %s: debounce_ms=%d", root_path, _config.debounce_ms)


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    """Analyze on open (runs in thread pool to avoid blocking event loop)."""
    _analyze_and_publish(ls, params.text_document.uri)


@server.thread()
@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    """Re-analyze on save (runs in thread pool to avoid blocking event loop)."""
    # Cancel any pending debounced analysis — save takes priority
    with _debounce_lock:
        existing = _debounce_timers.pop(params.text_document.uri, None)
        if existing is not None:
            existing.cancel()
    _analyze_and_publish(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    """Debounced analysis on edit. Cancels pending, schedules new after 500ms."""
    uri = params.text_document.uri
    # Get the latest content from the change events (full sync)
    if params.content_changes:
        source = params.content_changes[-1].text
        _schedule_debounced_analysis(ls, uri, source)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Show intermediate shape at cursor position."""
    uri = params.text_document.uri
    pos = params.position

    intermediates = _analysis_cache.get(uri, [])
    if not intermediates:
        return None

    # Find intermediates matching the cursor line
    line = pos.line + 1  # LSP is 0-indexed, our source_line is 1-indexed
    matching = [i for i in intermediates if i.source_line == line]

    if not matching:
        return None

    # Build hover content
    lines: list[str] = []
    for inter in matching:
        named = ", ".join(n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True))
        lines.append(f"`{inter.op_name}`: ({named})  `{inter.dtype}`")

    content = "\n\n".join(lines)

    return types.Hover(
        contents=types.MarkupContent(
            kind=types.MarkupKind.Markdown,
            value=content,
        ),
    )


@server.feature(types.TEXT_DOCUMENT_CODE_LENS)
def code_lens(ls: LanguageServer, params: types.CodeLensParams) -> list[types.CodeLens]:
    """Return shape annotations as CodeLens items above function definitions."""
    uri = params.text_document.uri
    lenses = _codelens_cache.get(uri, [])

    return [
        types.CodeLens(
            range=types.Range(
                start=types.Position(line=line, character=0),
                end=types.Position(line=line, character=0),
            ),
            command=types.Command(
                title=title,
                command="",
            ),
        )
        for line, title in lenses
    ]


def _shape_summary(spec: FunctionShapeSpec) -> str:
    """Build a shape summary string like '(batch, seq) -> (batch, hidden)'."""
    parts = []
    for pname, pspec in spec.params.items():
        dim_names = ", ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
        parts.append(f"{pname}: ({dim_names})")
    ret = ""
    if spec.return_spec is not None:
        ret_dims = ", ".join(d.name or str(d.size) or d.kind for d in spec.return_spec.dims)
        ret = f" -> ({ret_dims})"
    return f"{', '.join(parts)}{ret}"


def _spec_range(spec: FunctionShapeSpec) -> types.Range:
    """Build an LSP Range for a FunctionShapeSpec definition line."""
    line = max(0, spec.lineno - 1)
    return types.Range(
        start=types.Position(line=line, character=spec.col_offset),
        end=types.Position(
            line=line, character=spec.col_offset + len(spec.name) + 4
        ),  # "def " + name
    )


def _spec_selection_range(spec: FunctionShapeSpec) -> types.Range:
    """Build an LSP selection Range for the function name only."""
    line = max(0, spec.lineno - 1)
    # "def " is 4 chars before the name
    name_start = spec.col_offset + 4
    return types.Range(
        start=types.Position(line=line, character=name_start),
        end=types.Position(line=line, character=name_start + len(spec.name)),
    )


def _dim_range(dim: DimLocation) -> types.Range:
    """Build an LSP Range for a DimLocation."""
    line = max(0, dim.lineno - 1)
    return types.Range(
        start=types.Position(line=line, character=dim.col_start),
        end=types.Position(line=line, character=dim.col_end),
    )


# ---------------------------------------------------------------------------
# Navigation handlers
# ---------------------------------------------------------------------------


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: LanguageServer, params: types.DocumentSymbolParams
) -> list[types.DocumentSymbol] | None:
    """Return shape-annotated functions as document symbols."""
    uri = params.text_document.uri
    file_index = _workspace_index.get_file(uri)
    if file_index is None:
        return None

    # Group methods by class
    class_methods: dict[str, list[types.DocumentSymbol]] = {}
    top_level: list[types.DocumentSymbol] = []

    for spec in file_index.function_specs:
        kind = types.SymbolKind.Method if spec.is_method else types.SymbolKind.Function
        sym = types.DocumentSymbol(
            name=spec.name,
            kind=kind,
            range=_spec_range(spec),
            selection_range=_spec_selection_range(spec),
            detail=_shape_summary(spec),
        )
        if spec.is_method and spec.class_name is not None:
            class_methods.setdefault(spec.class_name, []).append(sym)
        else:
            top_level.append(sym)

    # Wrap class methods in class symbols
    for class_name, methods in class_methods.items():
        # Use the first method's line as approximate class range
        first_line = max(0, methods[0].range.start.line - 1) if methods else 0
        class_sym = types.DocumentSymbol(
            name=class_name,
            kind=types.SymbolKind.Class,
            range=types.Range(
                start=types.Position(line=first_line, character=0),
                end=methods[-1].range.end
                if methods
                else types.Position(line=first_line, character=0),
            ),
            selection_range=types.Range(
                start=types.Position(line=first_line, character=0),
                end=types.Position(line=first_line, character=len(class_name)),
            ),
            children=methods,
        )
        top_level.append(class_sym)

    return top_level or None


@server.feature(types.TEXT_DOCUMENT_DEFINITION)
def go_to_definition(
    ls: LanguageServer, params: types.DefinitionParams
) -> types.Location | list[types.Location] | None:
    """Navigate to dimension name definition or function definition."""
    uri = params.text_document.uri
    line = params.position.line + 1  # Convert to 1-based
    col = params.position.character

    # Try dimension name first
    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        defn = _workspace_index.find_dim_definition(dim.dim_name, dim.function_name, uri)
        if defn is not None and (defn.lineno != dim.lineno or defn.col_start != dim.col_start):
            return types.Location(uri=uri, range=_dim_range(defn))
        return None  # Already at definition

    # Try function name
    spec = _workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        return types.Location(uri=uri, range=_spec_selection_range(spec))

    return None


@server.feature(types.TEXT_DOCUMENT_REFERENCES)
def find_references(
    ls: LanguageServer, params: types.ReferenceParams
) -> list[types.Location] | None:
    """Find all references to a dimension name or function."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    # Try dimension name
    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        refs = _workspace_index.find_all_dim_references(dim.dim_name, uri=uri)
        return [types.Location(uri=uri, range=_dim_range(r)) for r in refs] or None

    # Try function name
    spec = _workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        locations: list[types.Location] = []
        if params.context.include_declaration:
            locations.append(types.Location(uri=uri, range=_spec_selection_range(spec)))
        # Find call sites across workspace
        callers = _workspace_index.get_callers_of(spec.name, uri)
        for call in callers:
            call_uri = uri  # TODO: cross-file URIs
            call_line = max(0, call.lineno - 1)
            locations.append(
                types.Location(
                    uri=call_uri,
                    range=types.Range(
                        start=types.Position(line=call_line, character=call.col_offset),
                        end=types.Position(line=call_line, character=call.end_col_offset),
                    ),
                )
            )
        return locations or None

    return None


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(
    ls: LanguageServer, params: types.DocumentHighlightParams
) -> list[types.DocumentHighlight] | None:
    """Highlight all occurrences of a dimension name in the file."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    refs = _workspace_index.find_all_dim_references(dim.dim_name, uri=uri)
    return [
        types.DocumentHighlight(
            range=_dim_range(r),
            kind=types.DocumentHighlightKind.Read,
        )
        for r in refs
    ] or None


@server.feature(types.TEXT_DOCUMENT_PREPARE_RENAME)
def prepare_rename(
    ls: LanguageServer, params: types.PrepareRenameParams
) -> types.PrepareRenamePlaceholder | None:
    """Check if rename is valid at cursor position (dim names only)."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    return types.PrepareRenamePlaceholder(
        range=_dim_range(dim),
        placeholder=dim.dim_name,
    )


@server.feature(types.TEXT_DOCUMENT_RENAME)
def rename(ls: LanguageServer, params: types.RenameParams) -> types.WorkspaceEdit | None:
    """Rename a dimension name across all annotations in the file."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    refs = _workspace_index.find_all_dim_references(dim.dim_name, uri=uri)
    if not refs:
        return None

    edits = [types.TextEdit(range=_dim_range(r), new_text=params.new_name) for r in refs]
    return types.WorkspaceEdit(changes={uri: edits})


@server.feature(types.WORKSPACE_SYMBOL)
def workspace_symbol(
    ls: LanguageServer, params: types.WorkspaceSymbolParams
) -> list[types.SymbolInformation] | None:
    """Search shape-annotated functions across the workspace."""
    results = _workspace_index.search_symbols(params.query)
    if not results:
        return None

    symbols: list[types.SymbolInformation] = []
    for spec in results:
        kind = types.SymbolKind.Method if spec.is_method else types.SymbolKind.Function
        # Convert file path to URI
        spec_uri = f"file://{spec.file_path}"
        symbols.append(
            types.SymbolInformation(
                name=spec.name,
                kind=kind,
                location=types.Location(uri=spec_uri, range=_spec_selection_range(spec)),
                container_name=spec.class_name,
            )
        )
    return symbols or None


@server.feature(types.TEXT_DOCUMENT_IMPLEMENTATION)
def go_to_implementation(
    ls: LanguageServer, params: types.ImplementationParams
) -> types.Location | list[types.Location] | None:
    """Navigate to function implementation (delegates to definition logic)."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    dim = _workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        defn = _workspace_index.find_dim_definition(dim.dim_name, dim.function_name, uri)
        if defn is not None and (defn.lineno != dim.lineno or defn.col_start != dim.col_start):
            return types.Location(uri=uri, range=_dim_range(defn))
        return None

    spec = _workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        return types.Location(uri=uri, range=_spec_selection_range(spec))

    return None


@server.feature(types.TEXT_DOCUMENT_PREPARE_CALL_HIERARCHY)
def prepare_call_hierarchy(
    ls: LanguageServer, params: types.CallHierarchyPrepareParams
) -> list[types.CallHierarchyItem] | None:
    """Prepare call hierarchy for a shape-annotated function."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    spec = _workspace_index.find_function_at(uri, line, col)
    if spec is None:
        return None

    return [
        types.CallHierarchyItem(
            name=spec.name,
            kind=types.SymbolKind.Method if spec.is_method else types.SymbolKind.Function,
            uri=uri,
            range=_spec_range(spec),
            selection_range=_spec_selection_range(spec),
            detail=_shape_summary(spec),
            data={"function_name": spec.name, "class_name": spec.class_name, "uri": uri},
        )
    ]


@server.feature(types.CALL_HIERARCHY_INCOMING_CALLS)
def incoming_calls(
    ls: LanguageServer, params: types.CallHierarchyIncomingCallsParams
) -> list[types.CallHierarchyIncomingCall] | None:
    """Find functions that call the target function."""
    data = params.item.data or {}
    function_name = data.get("function_name", params.item.name)
    item_uri = data.get("uri", params.item.uri)

    callers = _workspace_index.get_callers_of(function_name, item_uri)
    if not callers:
        return None

    results: list[types.CallHierarchyIncomingCall] = []
    for call in callers:
        caller_specs = _workspace_index.find_function_by_name(call.caller_name)
        if not caller_specs:
            continue
        caller_spec = caller_specs[0]
        caller_uri = f"file://{caller_spec.file_path}"
        call_line = max(0, call.lineno - 1)
        results.append(
            types.CallHierarchyIncomingCall(
                from_=types.CallHierarchyItem(
                    name=caller_spec.name,
                    kind=types.SymbolKind.Method
                    if caller_spec.is_method
                    else types.SymbolKind.Function,
                    uri=caller_uri,
                    range=_spec_range(caller_spec),
                    selection_range=_spec_selection_range(caller_spec),
                    detail=_shape_summary(caller_spec),
                    data={
                        "function_name": caller_spec.name,
                        "class_name": caller_spec.class_name,
                        "uri": caller_uri,
                    },
                ),
                from_ranges=[
                    types.Range(
                        start=types.Position(line=call_line, character=call.col_offset),
                        end=types.Position(line=call_line, character=call.end_col_offset),
                    )
                ],
            )
        )
    return results or None


@server.feature(types.CALL_HIERARCHY_OUTGOING_CALLS)
def outgoing_calls(
    ls: LanguageServer, params: types.CallHierarchyOutgoingCallsParams
) -> list[types.CallHierarchyOutgoingCall] | None:
    """Find functions called by the target function."""
    data = params.item.data or {}
    function_name = data.get("function_name", params.item.name)
    item_uri = data.get("uri", params.item.uri)

    callees = _workspace_index.get_callees_of(function_name, item_uri)
    if not callees:
        return None

    results: list[types.CallHierarchyOutgoingCall] = []
    for call in callees:
        callee_specs = _workspace_index.find_function_by_name(call.callee_name)
        if not callee_specs:
            continue
        callee_spec = callee_specs[0]
        callee_uri = f"file://{callee_spec.file_path}"
        call_line = max(0, call.lineno - 1)
        results.append(
            types.CallHierarchyOutgoingCall(
                to=types.CallHierarchyItem(
                    name=callee_spec.name,
                    kind=types.SymbolKind.Method
                    if callee_spec.is_method
                    else types.SymbolKind.Function,
                    uri=callee_uri,
                    range=_spec_range(callee_spec),
                    selection_range=_spec_selection_range(callee_spec),
                    detail=_shape_summary(callee_spec),
                    data={
                        "function_name": callee_spec.name,
                        "class_name": callee_spec.class_name,
                        "uri": callee_uri,
                    },
                ),
                from_ranges=[
                    types.Range(
                        start=types.Position(line=call_line, character=call.col_offset),
                        end=types.Position(line=call_line, character=call.end_col_offset),
                    )
                ],
            )
        )
    return results or None


def start_lsp() -> None:
    """Start the LSP server on stdio."""
    server.start_io()
