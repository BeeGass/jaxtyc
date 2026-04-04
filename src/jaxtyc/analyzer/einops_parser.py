"""Parse einops pattern strings into structured output dimension names."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\(([^()]+)\)|([a-zA-Z_]\w*)|(\d+)|(_)")


@dataclass(frozen=True)
class EinopsOutputPattern:
    """Parsed output side of an einops pattern string.

    Attributes:
        dim_names: One entry per output dimension. Named dims are strings,
            anonymous dims (``_`` or numeric literals) are None.
            Parenthesized groups are joined with ``*`` (e.g. ``(c h)`` -> ``"c*h"``).
    """

    dim_names: tuple[str | None, ...]


def parse_einops_output(pattern: str) -> EinopsOutputPattern | None:
    """Parse an einops pattern and extract output dimension names.

    Args:
        pattern: Full einops pattern string (e.g. ``"b c h w -> b (c h) w"``).

    Returns:
        Parsed output pattern, or None if the pattern is malformed.
    """
    if "->" not in pattern:
        return None

    _, _, output_side = pattern.partition("->")
    output_side = output_side.strip()

    # Reject unbalanced or nested parentheses
    depth = 0
    for ch in output_side:
        if ch == "(":
            depth += 1
            if depth > 1:
                return None
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
    if depth != 0:
        return None

    # Empty output = scalar result (e.g. full reduction)
    if not output_side:
        return EinopsOutputPattern(dim_names=())

    dims: list[str | None] = []
    for match in _TOKEN_RE.finditer(output_side):
        group_content = match.group(1)
        name = match.group(2)
        numeric = match.group(3)
        anon = match.group(4)

        if group_content is not None:
            inner_names = group_content.split()
            if not inner_names:
                return None
            dims.append("*".join(inner_names))
        elif name is not None:
            # Standalone underscore is anonymous in einops (skip dimension)
            dims.append(None if name == "_" else name)
        elif numeric is not None or anon is not None:
            dims.append(None)

    return EinopsOutputPattern(dim_names=tuple(dims))
