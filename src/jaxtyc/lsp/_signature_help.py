"""Signature help handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server


def _find_call_context(line_text: str, col: int) -> tuple[str | None, int]:
    """Find the function name being called and the active parameter index.

    Returns:
        Tuple of (function_name, active_parameter_index).
    """
    # Walk backwards from cursor to find the opening paren
    depth = 0
    comma_count = 0
    i = col - 1

    while i >= 0:
        ch = line_text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                # Found the opening paren
                # Extract function name before it
                name_end = i
                name_start = i - 1
                while name_start >= 0 and (
                    line_text[name_start].isalnum() or line_text[name_start] in ("_", ".")
                ):
                    name_start -= 1
                name_start += 1
                func_name = line_text[name_start:name_end]
                # Strip module prefix: obj.method -> method
                if "." in func_name:
                    func_name = func_name.rsplit(".", 1)[-1]
                return func_name, comma_count
            depth -= 1
        elif ch == "," and depth == 0:
            comma_count += 1
        i -= 1

    return None, 0


@server.feature(
    types.TEXT_DOCUMENT_SIGNATURE_HELP,
    types.SignatureHelpOptions(trigger_characters=["(", ","]),
)
def signature_help(
    ls: LanguageServer, params: types.SignatureHelpParams
) -> types.SignatureHelp | None:
    """Show shape signatures for jaxtyping-annotated function calls."""
    uri = params.text_document.uri
    pos = params.position

    doc = ls.workspace.get_text_document(uri)
    lines = doc.source.split("\n") if doc.source else []
    if pos.line >= len(lines):
        return None

    line_text = lines[pos.line]
    func_name, active_param = _find_call_context(line_text, pos.character)

    if func_name is None:
        return None

    # Look up function spec
    specs = _state.workspace_index.find_function_by_name(func_name, preferred_uri=uri)
    if not specs:
        return None

    spec = specs[0]

    # Build parameter information
    param_infos: list[types.ParameterInformation] = []
    param_labels: list[str] = []

    for pname, pspec in spec.params.items():
        dim_names = " ".join(d.name or str(d.size) or d.kind for d in pspec.dims)
        label = f"{pname}: ({dim_names}) {pspec.dtype}"
        param_labels.append(label)
        param_infos.append(
            types.ParameterInformation(
                label=label,
                documentation=f"Shape: `({dim_names})`, dtype: `{pspec.dtype}`",
            )
        )

    if not param_infos:
        return None

    # Build signature label
    ret_part = ""
    if spec.return_spec is not None:
        ret_dims = " ".join(d.name or str(d.size) or d.kind for d in spec.return_spec.dims)
        ret_part = f" -> ({ret_dims}) {spec.return_spec.dtype}"

    sig_label = f"{spec.name}({', '.join(param_labels)}){ret_part}"

    return types.SignatureHelp(
        signatures=[
            types.SignatureInformation(
                label=sig_label,
                parameters=param_infos,
                active_parameter=min(active_param, len(param_infos) - 1),
            )
        ],
        active_signature=0,
        active_parameter=min(active_param, len(param_infos) - 1),
    )
