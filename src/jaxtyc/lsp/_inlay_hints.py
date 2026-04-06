"""Inlay hint handler for the jaxtyc LSP server."""

from __future__ import annotations

import re

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.lsp import _state
from jaxtyc.lsp._util import format_dtype
from jaxtyc.lsp._util import format_named_shape
from jaxtyc.lsp.server import server
from jaxtyc.types import ErrorHintInfo
from jaxtyc.types import IntermediateShape

# Matches single-variable assignment: "  var = ...", "  self.var = ...", "  var: Type = ..."
# Does NOT match tuple unpacking, augmented assignment (+=), or comparison (==).
_ASSIGN_RE = re.compile(r"^(\s*)([\w]+(?:\.[\w]+)*)\s*(?::[^=]*)?=(?!=)")

# Matches return statements: "  return expr" or bare "  return"
_RETURN_RE = re.compile(r"^\s*return\b")


def _classify_line(line_text: str) -> tuple[int | None, str]:
    """Classify a source line for hint positioning.

    Returns:
        (character_position, kind) where kind is "assign", "return", or "other".
        character_position is set for assignments (after variable name), None otherwise.
    """
    if _RETURN_RE.match(line_text):
        return None, "return"
    m = _ASSIGN_RE.match(line_text)
    if m is not None:
        indent = m.group(1)
        varname = m.group(2)
        return len(indent) + len(varname), "assign"
    return None, "other"


def _find_hint_position(line_text: str) -> int | None:
    """Return the character position right after the variable name for assignments.

    Returns None for return statements, bare expressions, tuple unpacking,
    augmented assignments, and comparisons — caller should fall back to
    end-of-line positioning.
    """
    pos, kind = _classify_line(line_text)
    if kind == "assign":
        return pos
    return None


def _format_shape(inter: IntermediateShape, dtype_style: str) -> str:
    """Format intermediate shape as compact dtype[dim1|axis dim2|axis] string.

    When sharding info is available and not suppressed by config, each
    dimension is annotated with its partition axis using pipe syntax
    (e.g. ``batch|data seq|None``). Scalars (rank 0) use empty brackets.

    Synthetic dim names (_ellipsis_*, _var_*, _anon_*) are collapsed into
    user-friendly display forms by format_named_shape().
    """
    dtype = format_dtype(inter.dtype, dtype_style)
    if not inter.shape:
        return f"{dtype}[]"

    dim_parts = format_named_shape(inter.named_shape, inter.shape)

    sharding_axes = _get_sharding_axes(inter)
    if sharding_axes is not None:
        annotated: list[str] = []
        orig_idx = 0
        for label in dim_parts:
            if label.startswith("...") or label.startswith("*") or label == "_":
                # Skip sharding for collapsed/anonymous dims, advance orig_idx
                if label.startswith("..."):
                    while (
                        orig_idx < len(inter.named_shape)
                        and (elem := inter.named_shape[orig_idx]) is not None
                        and elem.startswith("_ellipsis_")
                    ):
                        orig_idx += 1
                elif label.startswith("*"):
                    var_name = label[1:]
                    while (
                        orig_idx < len(inter.named_shape)
                        and (elem := inter.named_shape[orig_idx]) is not None
                        and elem.startswith(f"_var_{var_name}_")
                    ):
                        orig_idx += 1
                else:
                    orig_idx += 1
                annotated.append(label)
            else:
                if orig_idx < len(sharding_axes):
                    axis = sharding_axes[orig_idx]
                    label = f"{label}|{axis}" if axis is not None else f"{label}|None"
                orig_idx += 1
                annotated.append(label)
        dim_parts = annotated

    return f"{dtype}[{' '.join(dim_parts)}]"


def _get_sharding_axes(inter: IntermediateShape) -> tuple[str | None, ...] | None:
    """Extract per-dimension sharding axes, or None if sharding is suppressed."""
    if inter.sharding is None:
        return None
    display = _state.config.sharding.display
    if display == "off":
        return None
    if display == "constrained_only" and inter.sharding.source_primitive != "sharding_constraint":
        return None
    return inter.sharding.partition_spec


def _format_error(error: ErrorHintInfo, style: str) -> str:
    """Format error info as inline text."""
    sep = " | " if style == "pipe" else " \u26a0 "
    return f"{sep}{error.message}"


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hint(ls: LanguageServer, params: types.InlayHintParams) -> list[types.InlayHint] | None:
    """Show resolved shapes as inline hints next to operations.

    Uses compact dtype[dim1, dim2] format. Takes the last intermediate per
    line (final shape after nested ops). Places hints after variable names
    for assignment lines, falls back to end-of-line for returns and
    expressions.
    """
    uri = params.text_document.uri
    with _state.cache_lock:
        intermediates = list(_state.analysis_cache.get(uri, []))
        error_hints = list(_state.error_hints_cache.get(uri, []))
        source_text = _state.source_cache.get(uri)

    if not intermediates:
        return None

    # Split source into lines for position detection
    source_lines: list[str] | None = None
    if source_text is not None:
        source_lines = source_text.splitlines()

    # Index error hints by line for fast lookup
    error_by_line: dict[int, ErrorHintInfo] = {}
    for eh in error_hints:
        error_by_line.setdefault(eh.source_line, eh)

    config_hints = _state.config.hints
    error_mode = config_hints.error_mode
    error_style = config_hints.error_style
    dtype_style = config_hints.dtype_style

    start_line = params.range.start.line + 1  # Convert to 1-based
    end_line = params.range.end.line + 1

    # Collect LAST intermediate per line (final shape after nested ops)
    last_per_line: dict[int, IntermediateShape] = {}
    for inter in intermediates:
        if inter.source_line < start_line or inter.source_line > end_line:
            continue
        last_per_line[inter.source_line] = inter

    hints: list[types.InlayHint] = []

    for line_num in sorted(last_per_line):
        inter = last_per_line[line_num]

        # Shape part (includes sharding as dim|axis when available)
        shape_text = _format_shape(inter, dtype_style)

        # Error part
        error_text = ""
        error = error_by_line.get(inter.source_line)
        if error is not None:
            error_text = _format_error(error, error_style)

        # Compose label based on error_mode
        if error is not None and error_mode == "replace":
            label = error_text.lstrip()
        else:
            label = f"{shape_text}{error_text}"

        # Determine hint position and prefix based on line kind
        character = 999  # default: end-of-line
        prefix = ""
        pad_left = True
        line_idx = inter.source_line - 1  # 0-indexed
        if source_lines is not None and 0 <= line_idx < len(source_lines):
            var_pos, kind = _classify_line(source_lines[line_idx])
            if kind == "assign" and var_pos is not None:
                character = var_pos
                prefix = ": "
                pad_left = False  # colon sits flush against variable name
            elif kind == "return":
                prefix = " -> "

        hints.append(
            types.InlayHint(
                position=types.Position(line=inter.source_line - 1, character=character),
                label=f"{prefix}{label}",
                kind=types.InlayHintKind.Type,
                padding_left=pad_left,
            )
        )

    return hints or None
