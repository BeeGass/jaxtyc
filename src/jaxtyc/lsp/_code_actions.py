"""Code action handler for the jaxtyc LSP server."""

from __future__ import annotations

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp.server import server
from jaxtyc.lsp.suggestions import suggest_fixes


@server.feature(
    types.TEXT_DOCUMENT_CODE_ACTION,
    types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
)
def code_action(
    ls: LanguageServer, params: types.CodeActionParams
) -> list[types.CodeAction] | None:
    """Generate quick-fix code actions for shape diagnostics."""
    actions: list[types.CodeAction] = []
    uri = params.text_document.uri

    for diag in params.context.diagnostics:
        if diag.source != "jaxtyc" or diag.data is None:
            continue

        data = diag.data
        if not isinstance(data, dict):
            continue

        expected = data.get("expected_shape")
        actual = data.get("actual_shape")
        dim_mapping = data.get("dim_name_mapping")

        if expected is not None and actual is not None and dim_mapping is not None:
            # Build reverse map: size -> name
            dim_names = {v: k for k, v in dim_mapping.items()}

            fixes = suggest_fixes(
                tuple(expected),
                tuple(actual),
                dim_names,
                prefer_einops=_state.config.prefer_einops,
            )

            for fix in fixes:
                actions.append(
                    types.CodeAction(
                        title=fix.title,
                        kind=types.CodeActionKind.QuickFix,
                        diagnostics=[diag],
                        data={
                            "code": fix.code,
                            "kind": fix.kind,
                        },
                    )
                )

        # Always offer suppress action
        actions.append(
            types.CodeAction(
                title="Suppress with # jaxtyc: ignore",
                kind=types.CodeActionKind.QuickFix,
                diagnostics=[diag],
                edit=types.WorkspaceEdit(
                    changes={
                        uri: [
                            types.TextEdit(
                                range=types.Range(
                                    start=types.Position(
                                        line=diag.range.start.line,
                                        character=1000,  # End of line
                                    ),
                                    end=types.Position(
                                        line=diag.range.start.line,
                                        character=1000,
                                    ),
                                ),
                                new_text=f"  # jaxtyc: ignore[{diag.code}]"
                                if diag.code
                                else "  # jaxtyc: ignore",
                            )
                        ]
                    }
                ),
            )
        )

    return actions or None
