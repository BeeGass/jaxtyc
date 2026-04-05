"""Detect einops operations in Python source via AST analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from jaxtyc.analyzer.einops_parser import parse_einops_output

_EINOPS_OPS: frozenset[str] = frozenset({"rearrange", "reduce", "repeat"})


@dataclass(frozen=True)
class EinopsCallInfo:
    """An einops operation detected in source code.

    Attributes:
        line: 1-based source line of the call.
        operation: Operation name (``rearrange``, ``reduce``, ``repeat``).
        pattern: Raw pattern string from the source.
        output_names: Parsed output dimension names (``None`` for anonymous dims).
    """

    line: int
    operation: str
    pattern: str
    output_names: tuple[str | None, ...]


def extract_einops_calls(source: str) -> list[EinopsCallInfo]:
    """Extract einops calls from Python source code.

    Detects ``einops.rearrange(x, 'pattern')``, ``rearrange(x, 'pattern')``,
    and the same for ``reduce`` and ``repeat``. The pattern must be a string
    constant (variable patterns are skipped).

    Args:
        source: Python source code text.

    Returns:
        List of detected einops calls with parsed output dimension names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Track bare einops imports: `from einops import rearrange`
    bare_imports: set[str] = set()
    has_einops_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "einops" or alias.name.startswith("einops."):
                    has_einops_import = True
        elif isinstance(node, ast.ImportFrom) and node.module == "einops":
            has_einops_import = True
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                if alias.name in _EINOPS_OPS:
                    bare_imports.add(local_name)

    if not has_einops_import and not bare_imports:
        return []

    results: list[EinopsCallInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        op_name = _match_einops_call(node, bare_imports)
        if op_name is None:
            continue

        # Pattern is the second positional argument
        if len(node.args) < 2:
            continue
        pattern_node = node.args[1]
        if not isinstance(pattern_node, ast.Constant) or not isinstance(pattern_node.value, str):
            continue

        pattern = pattern_node.value
        parsed = parse_einops_output(pattern)
        if parsed is None:
            continue

        results.append(
            EinopsCallInfo(
                line=node.lineno,
                operation=op_name,
                pattern=pattern,
                output_names=parsed.dim_names,
            )
        )

    return results


def _match_einops_call(node: ast.Call, bare_imports: set[str]) -> str | None:
    """Return the einops operation name if the call matches, else None."""
    func = node.func

    # einops.rearrange(...)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "einops"
        and func.attr in _EINOPS_OPS
    ):
        return func.attr

    # rearrange(...) via bare import
    if isinstance(func, ast.Name) and func.id in bare_imports:
        return func.id

    return None
