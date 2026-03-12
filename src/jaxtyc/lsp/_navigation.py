"""Navigation handlers for the jaxtyc LSP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp._util import dim_label
from jaxtyc.lsp._util import dim_range
from jaxtyc.lsp._util import format_dtype
from jaxtyc.lsp._util import shape_summary
from jaxtyc.lsp._util import spec_range
from jaxtyc.lsp._util import spec_selection_range
from jaxtyc.lsp.server import server

if TYPE_CHECKING:
    from jaxtyc.types import FunctionShapeSpec
    from jaxtyc.types import ShapeSpec


def _word_at(text: str, col: int) -> str:
    """Extract the identifier at the given 0-based column in *text*."""
    if col >= len(text):
        return ""
    if not (text[col].isalnum() or text[col] == "_"):
        return ""
    start = col
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    end = col
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return text[start:end]


def _param_hover(param_name: str, spec: ShapeSpec) -> str:
    """Build hover markdown for a shape-annotated parameter."""
    dim_parts = []
    for d in spec.dims:
        label = dim_label(d)
        dim_parts.append(f"`{label}`")
    dims_str = ", ".join(dim_parts)
    return f"**`{param_name}`** — `{spec.dtype}[{', '.join(dim_label(d) for d in spec.dims)}]`\n\nDimensions: {dims_str}"


def _function_hover(func_spec: FunctionShapeSpec) -> str:
    """Build hover markdown showing the full shape signature of a function."""
    parts: list[str] = [f"**`{func_spec.name}`** — shape signature\n"]
    # Parameters
    for pname, pspec in func_spec.params.items():
        dim_names = ", ".join(dim_label(d) for d in pspec.dims)
        parts.append(f"- `{pname}`: `{pspec.dtype}[{dim_names}]`")
    # Return
    if func_spec.return_spec is not None:
        ret_dims = ", ".join(dim_label(d) for d in func_spec.return_spec.dims)
        parts.append(f"- **returns**: `{func_spec.return_spec.dtype}[{ret_dims}]`")
    if func_spec.return_specs:
        parts.append("- **returns** (tuple):")
        for i, rspec in enumerate(func_spec.return_specs):
            ret_dims = ", ".join(dim_label(d) for d in rspec.dims)
            parts.append(f"  - `[{i}]`: `{rspec.dtype}[{ret_dims}]`")
    return "\n".join(parts)


@server.feature(types.TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Show shape info for parameters, functions, dimensions, and intermediates."""
    uri = params.text_document.uri
    pos = params.position
    line = pos.line + 1  # LSP is 0-indexed, our source_line is 1-indexed

    # Check if cursor is on a dimension name in an annotation
    dim = _state.workspace_index.find_dim_at(uri, line, pos.character)
    if dim is not None:
        lines_parts: list[str] = [f"**`{dim.dim_name}`** — dimension name"]
        # Show resolved prime size if available
        with _state.cache_lock:
            env = _state.dim_env_cache.get(uri)
        if env is not None:
            from jaxtyc.analyzer.dim_env import DimEnv

            if isinstance(env, DimEnv):
                size = env.get_size(dim.dim_name)
                name_check = env.resolve_name(size)
                if name_check == dim.dim_name:
                    lines_parts.append(f"Symbolic size: `{size}` (prime)")
        # Show all usages in the file
        all_refs = _state.workspace_index.find_all_dim_references(dim.dim_name, uri=uri)
        if all_refs:
            lines_parts.append(f"**Used {len(all_refs)} time(s) in this file:**")
            for ref in all_refs:
                param_label = "return" if ref.param_name == "__return__" else ref.param_name
                lines_parts.append(f"- `{ref.function_name}` / `{param_label}` (line {ref.lineno})")
        content = "\n\n".join(lines_parts)
        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=content,
            ),
        )

    # Check if cursor is on a parameter name or function name in a shape-annotated function
    try:
        doc = ls.workspace.get_text_document(uri)
    except RuntimeError:
        doc = None
    if doc is not None and pos.line < len(doc.lines):
        line_text = doc.lines[pos.line]
        word = _word_at(line_text, pos.character)
        if word:
            # Check function name on the def line
            func_at = _state.workspace_index.find_function_at(uri, line, pos.character)
            if func_at is not None and word == func_at.name:
                return types.Hover(
                    contents=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=_function_hover(func_at),
                    ),
                )

            # Check parameter name within a function signature
            func_containing = _state.workspace_index.find_function_containing(uri, line)
            if func_containing is not None and word in func_containing.params:
                pspec = func_containing.params[word]
                return types.Hover(
                    contents=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=_param_hover(word, pspec),
                    ),
                )

    # Check if cursor is on a call site
    call_site = _state.workspace_index.find_call_site_at(uri, line, pos.character)
    if call_site is not None:
        callee_specs = _state.workspace_index.find_function_by_name(
            call_site.callee_name, preferred_uri=uri
        )
        if callee_specs:
            callee_spec = callee_specs[0]
            call_parts: list[str] = [_function_hover(callee_spec)]

            callee_uri = _state.workspace_index.uri_for_file(callee_spec.file_path)
            if callee_uri is not None:
                with _state.cache_lock:
                    callee_traces = _state.trace_results_cache.get(callee_uri, {})
                    callee_env_obj = _state.dim_env_cache.get(callee_uri)

                callee_trace = callee_traces.get(call_site.callee_name)
                if callee_trace is not None and callee_trace.success:
                    if callee_trace.output_shape is not None and callee_env_obj is not None:
                        from jaxtyc.analyzer.dim_env import DimEnv

                        if isinstance(callee_env_obj, DimEnv):
                            named = callee_env_obj.shape_to_names(callee_trace.output_shape)
                            out_str = ", ".join(
                                n or str(s)
                                for n, s in zip(
                                    named,
                                    callee_trace.output_shape,
                                    strict=True,
                                )
                            )
                            call_parts.append(f"\n**Traced output**: `({out_str})`")

                            if callee_spec.return_spec is not None:
                                expected = callee_env_obj.make_shape(callee_spec.return_spec)
                                if expected != callee_trace.output_shape:
                                    exp_named = callee_env_obj.shape_to_names(expected)
                                    exp_str = ", ".join(
                                        n or str(s)
                                        for n, s in zip(exp_named, expected, strict=True)
                                    )
                                    call_parts.append(
                                        f"\n**Mismatch**: annotated `({exp_str})`, "
                                        f"traced `({out_str})`"
                                    )
                elif callee_trace is not None and not callee_trace.success:
                    call_parts.append(f"\n**Trace error**: {callee_trace.error}")

            return types.Hover(
                contents=types.MarkupContent(
                    kind=types.MarkupKind.Markdown,
                    value="\n".join(call_parts),
                ),
            )

        # External call site — callee not in workspace
        display = call_site.callee_qualified_name or call_site.callee_name
        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=f"**`{display}`** (external)",
            ),
        )

    # Show all intermediate shapes at cursor line (Option E format)
    with _state.cache_lock:
        intermediates = _state.analysis_cache.get(uri, [])

    matching = [i for i in intermediates if i.source_line == line] if intermediates else []

    if not matching:
        # Fallback: show trace error if cursor is inside a function that failed tracing
        func_containing = _state.workspace_index.find_function_containing(uri, line)
        if func_containing is not None:
            with _state.cache_lock:
                traces = _state.trace_results_cache.get(uri, {})
            trace = traces.get(func_containing.name)
            if trace is not None and not trace.success:
                return types.Hover(
                    contents=types.MarkupContent(
                        kind=types.MarkupKind.Markdown,
                        value=f"**Trace error in `{func_containing.name}`**: {trace.error}",
                    ),
                )
        return None

    dtype_style = _state.config.hints.dtype_style

    # Build hover content with full intermediate chain
    lines: list[str] = [f"**Intermediates at line {line}:**"]
    for idx, inter in enumerate(matching):
        named = ", ".join(n or str(s) for n, s in zip(inter.named_shape, inter.shape, strict=True))
        dtype = format_dtype(inter.dtype, dtype_style)
        entry = f"`{inter.op_name}` \u2192 `{dtype}[{named}]`"
        if idx == len(matching) - 1:
            entry += " \u2190 *final*"
        lines.append(entry)

        # Show sharding info inline if present
        if inter.sharding is not None:
            parts = ", ".join(
                repr(a) if a is not None else "None" for a in inter.sharding.partition_spec
            )
            mesh_parts = ", ".join(repr(a) for a in inter.sharding.mesh_axis_names)
            lines.append(f"  Sharding: `P({parts})` | mesh(`{mesh_parts}`)")

    # Check for divergence error at this line
    with _state.cache_lock:
        error_hints = _state.error_hints_cache.get(uri, [])
    divergence = [eh for eh in error_hints if eh.source_line == line]
    if divergence:
        eh = divergence[0]
        lines.append("")
        lines.append("**Shape divergence detected:**")
        if eh.expected_named is not None:
            lines.append(f"- Expected: `({', '.join(eh.expected_named)})`")
        if eh.actual_named is not None:
            lines.append(f"- Actual: `({', '.join(eh.actual_named)})`")
        lines.append(f"- {eh.message}")

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
    with _state.cache_lock:
        lenses = _state.codelens_cache.get(uri, [])

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


# ---------------------------------------------------------------------------
# Navigation handlers
# ---------------------------------------------------------------------------


@server.feature(types.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    ls: LanguageServer, params: types.DocumentSymbolParams
) -> list[types.DocumentSymbol] | None:
    """Return shape-annotated functions as document symbols."""
    uri = params.text_document.uri
    file_index = _state.workspace_index.get_file(uri)
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
            range=spec_range(spec),
            selection_range=spec_selection_range(spec),
            detail=shape_summary(spec),
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
    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        # Try function-scoped first
        defn = _state.workspace_index.find_dim_definition(dim.dim_name, dim.function_name, uri)
        if defn is not None and (defn.lineno != dim.lineno or defn.col_start != dim.col_start):
            return types.Location(uri=uri, range=dim_range(defn))
        # Fall back to file-scoped (first occurrence in the entire file)
        file_defn = _state.workspace_index.find_dim_definition_in_file(dim.dim_name, uri)
        if file_defn is not None:
            return types.Location(uri=uri, range=dim_range(file_defn))
        return None

    # Try function name
    spec = _state.workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        return types.Location(uri=uri, range=spec_selection_range(spec))

    # Try call site -- resolve callee name to its definition
    call_site = _state.workspace_index.find_call_site_at(uri, line, col)
    if call_site is not None:
        callee_specs = _state.workspace_index.find_function_by_name(
            call_site.callee_name, preferred_uri=uri
        )
        if callee_specs:
            cs_spec = callee_specs[0]
            spec_uri = (
                _state.workspace_index.uri_for_file(cs_spec.file_path)
                or f"file://{cs_spec.file_path}"
            )
            return types.Location(uri=spec_uri, range=spec_selection_range(cs_spec))

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
    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        refs = _state.workspace_index.find_all_dim_references(dim.dim_name, uri=None)
        locations = []
        for r in refs:
            ref_uri = _state.workspace_index.uri_for_file(r.file_path) or f"file://{r.file_path}"
            locations.append(types.Location(uri=ref_uri, range=dim_range(r)))
        return locations or None

    # Try function name
    spec = _state.workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        locations: list[types.Location] = []
        if params.context.include_declaration:
            locations.append(types.Location(uri=uri, range=spec_selection_range(spec)))
        # Find call sites based on configured scope
        if _state.config.navigation.references_scope == "workspace":
            callers = _state.workspace_index.get_all_callers_of(spec.name)
        else:
            callers = _state.workspace_index.get_callers_of(spec.name, uri)
        for call in callers:
            call_uri = (
                _state.workspace_index.uri_for_file(call.file_path) or f"file://{call.file_path}"
            )
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

    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    refs = _state.workspace_index.find_all_dim_references(dim.dim_name, uri=uri)
    return [
        types.DocumentHighlight(
            range=dim_range(r),
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

    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    return types.PrepareRenamePlaceholder(
        range=dim_range(dim),
        placeholder=dim.dim_name,
    )


@server.feature(types.TEXT_DOCUMENT_RENAME)
def rename(ls: LanguageServer, params: types.RenameParams) -> types.WorkspaceEdit | None:
    """Rename a dimension name across all annotations in the file."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is None:
        return None

    refs = _state.workspace_index.find_all_dim_references(dim.dim_name, uri=None)
    if not refs:
        return None

    edits_by_uri: dict[str, list[types.TextEdit]] = {}
    for r in refs:
        ref_uri = _state.workspace_index.uri_for_file(r.file_path) or f"file://{r.file_path}"
        edits_by_uri.setdefault(ref_uri, []).append(
            types.TextEdit(range=dim_range(r), new_text=params.new_name)
        )
    return types.WorkspaceEdit(changes=edits_by_uri)


@server.feature(types.WORKSPACE_SYMBOL)
def workspace_symbol(
    ls: LanguageServer, params: types.WorkspaceSymbolParams
) -> list[types.SymbolInformation] | None:
    """Search all workspace functions by name (annotated and non-annotated)."""
    from jaxtyc.types import FunctionShapeSpec

    results = _state.workspace_index.search_symbols(params.query)
    if not results:
        return None

    symbols: list[types.SymbolInformation] = []
    for item in results:
        kind = types.SymbolKind.Method if item.is_method else types.SymbolKind.Function
        item_uri = f"file://{item.file_path}"
        if isinstance(item, FunctionShapeSpec):
            sel_range = spec_selection_range(item)
        else:
            line = max(0, item.lineno - 1)
            sel_range = types.Range(
                start=types.Position(line=line, character=item.name_col_offset),
                end=types.Position(line=line, character=item.name_col_offset + len(item.name)),
            )
        symbols.append(
            types.SymbolInformation(
                name=item.name,
                kind=kind,
                location=types.Location(uri=item_uri, range=sel_range),
                container_name=item.class_name,
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

    dim = _state.workspace_index.find_dim_at(uri, line, col)
    if dim is not None:
        defn = _state.workspace_index.find_dim_definition(dim.dim_name, dim.function_name, uri)
        if defn is not None and (defn.lineno != dim.lineno or defn.col_start != dim.col_start):
            return types.Location(uri=uri, range=dim_range(defn))
        file_defn = _state.workspace_index.find_dim_definition_in_file(dim.dim_name, uri)
        if file_defn is not None:
            return types.Location(uri=uri, range=dim_range(file_defn))
        return None

    spec = _state.workspace_index.find_function_at(uri, line, col)
    if spec is not None:
        return types.Location(uri=uri, range=spec_selection_range(spec))

    # Try call site -- resolve callee name to its definition
    call_site = _state.workspace_index.find_call_site_at(uri, line, col)
    if call_site is not None:
        callee_specs = _state.workspace_index.find_function_by_name(
            call_site.callee_name, preferred_uri=uri
        )
        if callee_specs:
            cs_spec = callee_specs[0]
            spec_uri = (
                _state.workspace_index.uri_for_file(cs_spec.file_path)
                or f"file://{cs_spec.file_path}"
            )
            return types.Location(uri=spec_uri, range=spec_selection_range(cs_spec))

    return None


@server.feature(types.TEXT_DOCUMENT_PREPARE_CALL_HIERARCHY)
def prepare_call_hierarchy(
    ls: LanguageServer, params: types.CallHierarchyPrepareParams
) -> list[types.CallHierarchyItem] | None:
    """Prepare call hierarchy for a shape-annotated function."""
    uri = params.text_document.uri
    line = params.position.line + 1
    col = params.position.character

    spec = _state.workspace_index.find_function_at(uri, line, col)
    if spec is None:
        return None

    return [
        types.CallHierarchyItem(
            name=spec.name,
            kind=types.SymbolKind.Method if spec.is_method else types.SymbolKind.Function,
            uri=uri,
            range=spec_range(spec),
            selection_range=spec_selection_range(spec),
            detail=shape_summary(spec),
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

    if _state.config.navigation.references_scope == "workspace":
        callers = _state.workspace_index.get_all_callers_of(function_name)
    else:
        callers = _state.workspace_index.get_callers_of(function_name, item_uri)
    if not callers:
        return None

    results: list[types.CallHierarchyIncomingCall] = []
    for call in callers:
        caller_uri = _state.workspace_index.uri_for_file(call.file_path)
        caller_specs = _state.workspace_index.find_function_by_name(
            call.caller_name, preferred_uri=caller_uri
        )
        if caller_specs:
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
                        range=spec_range(caller_spec),
                        selection_range=spec_selection_range(caller_spec),
                        detail=shape_summary(caller_spec),
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
            continue

        # Fallback: non-annotated workspace function
        caller_defs = _state.workspace_index.find_function_def_by_name(
            call.caller_name, preferred_uri=caller_uri
        )
        if not caller_defs:
            continue
        fdef = caller_defs[0]
        fdef_uri = _state.workspace_index.uri_for_file(fdef.file_path) or f"file://{fdef.file_path}"
        call_line = max(0, call.lineno - 1)
        fdef_line = max(0, fdef.lineno - 1)
        results.append(
            types.CallHierarchyIncomingCall(
                from_=types.CallHierarchyItem(
                    name=fdef.name,
                    kind=types.SymbolKind.Method if fdef.is_method else types.SymbolKind.Function,
                    uri=fdef_uri,
                    range=types.Range(
                        start=types.Position(line=fdef_line, character=fdef.col_offset),
                        end=types.Position(line=max(0, fdef.end_lineno - 1), character=0),
                    ),
                    selection_range=types.Range(
                        start=types.Position(line=fdef_line, character=fdef.name_col_offset),
                        end=types.Position(
                            line=fdef_line,
                            character=fdef.name_col_offset + len(fdef.name),
                        ),
                    ),
                    data={
                        "function_name": fdef.name,
                        "class_name": fdef.class_name,
                        "uri": fdef_uri,
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

    callees = _state.workspace_index.get_callees_of(function_name, item_uri)
    if not callees:
        return None

    results: list[types.CallHierarchyOutgoingCall] = []
    for call in callees:
        callee_uri = _state.workspace_index.uri_for_file(call.file_path)
        callee_specs = _state.workspace_index.find_function_by_name(
            call.callee_name, preferred_uri=callee_uri
        )
        if callee_specs:
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
                        range=spec_range(callee_spec),
                        selection_range=spec_selection_range(callee_spec),
                        detail=shape_summary(callee_spec),
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
            continue

        # Fallback: non-annotated workspace function
        callee_defs = _state.workspace_index.find_function_def_by_name(
            call.callee_name, preferred_uri=callee_uri
        )
        if not callee_defs:
            # External/library call -- no workspace definition found
            if _state.config.navigation.include_external_calls:
                call_line = max(0, call.lineno - 1)
                results.append(
                    types.CallHierarchyOutgoingCall(
                        to=types.CallHierarchyItem(
                            name=call.callee_qualified_name or call.callee_name,
                            kind=types.SymbolKind.Function,
                            uri=item_uri,
                            range=types.Range(
                                start=types.Position(line=call_line, character=call.col_offset),
                                end=types.Position(line=call_line, character=call.end_col_offset),
                            ),
                            selection_range=types.Range(
                                start=types.Position(line=call_line, character=call.col_offset),
                                end=types.Position(line=call_line, character=call.end_col_offset),
                            ),
                            detail="(external)",
                        ),
                        from_ranges=[
                            types.Range(
                                start=types.Position(line=call_line, character=call.col_offset),
                                end=types.Position(line=call_line, character=call.end_col_offset),
                            )
                        ],
                    )
                )
            continue
        fdef = callee_defs[0]
        fdef_uri = _state.workspace_index.uri_for_file(fdef.file_path) or f"file://{fdef.file_path}"
        call_line = max(0, call.lineno - 1)
        fdef_line = max(0, fdef.lineno - 1)
        results.append(
            types.CallHierarchyOutgoingCall(
                to=types.CallHierarchyItem(
                    name=fdef.name,
                    kind=types.SymbolKind.Method if fdef.is_method else types.SymbolKind.Function,
                    uri=fdef_uri,
                    range=types.Range(
                        start=types.Position(line=fdef_line, character=fdef.col_offset),
                        end=types.Position(line=max(0, fdef.end_lineno - 1), character=0),
                    ),
                    selection_range=types.Range(
                        start=types.Position(line=fdef_line, character=fdef.name_col_offset),
                        end=types.Position(
                            line=fdef_line,
                            character=fdef.name_col_offset + len(fdef.name),
                        ),
                    ),
                    data={
                        "function_name": fdef.name,
                        "class_name": fdef.class_name,
                        "uri": fdef_uri,
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
