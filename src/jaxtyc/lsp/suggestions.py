"""Shape fix suggestion generation (JAX-native + einops)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeFix:
    """A suggested code fix for a shape mismatch.

    Attributes:
        title: Short description for the code action menu.
        code: Suggested code snippet.
        kind: Fix category (transpose, expand, squeeze, reshape).
    """

    title: str
    code: str
    kind: str


def suggest_fixes(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
    dim_names: dict[int, str],
    prefer_einops: bool = False,
) -> list[ShapeFix]:
    """Generate shape fix suggestions.

    Args:
        expected: Expected shape tuple.
        actual: Actual shape tuple from tracing.
        dim_names: Map of size -> dimension name for readable suggestions.
        prefer_einops: If True, einops suggestions are primary.

    Returns:
        List of ShapeFix suggestions, ordered by relevance.
    """
    fixes: list[ShapeFix] = []

    def _name(size: int) -> str:
        return dim_names.get(size, str(size))

    expected_names = tuple(_name(s) for s in expected)
    actual_names = tuple(_name(s) for s in actual)

    if len(expected) == len(actual) and sorted(str(s) for s in expected) == sorted(
        str(s) for s in actual
    ):
        # Transposition detected
        _suggest_transpose(expected, actual, expected_names, actual_names, prefer_einops, fixes)
    elif len(actual) < len(expected):
        # Missing dimensions
        _suggest_expand(expected, actual, expected_names, actual_names, prefer_einops, fixes)
    elif len(actual) > len(expected):
        # Extra dimensions
        _suggest_squeeze(expected, actual, expected_names, actual_names, prefer_einops, fixes)
    else:
        # Same rank but different sizes — suggest reshape
        _suggest_reshape(expected, actual, expected_names, actual_names, prefer_einops, fixes)

    return fixes


def _suggest_transpose(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
    expected_names: tuple[str, ...],
    actual_names: tuple[str, ...],
    prefer_einops: bool,
    fixes: list[ShapeFix],
) -> None:
    """Add transpose suggestions when dimensions are permuted."""
    # Compute permutation: expected[i] should be at perm[i] in actual
    perm = []
    remaining = list(range(len(actual)))
    for e in expected:
        for j in remaining:
            if actual[j] == e:
                perm.append(j)
                remaining.remove(j)
                break

    jax_fix = ShapeFix(
        title=f"Transpose with jnp.transpose(x, {perm})",
        code=f"jnp.transpose(x, {perm})",
        kind="transpose",
    )

    # Check if it's a simple two-axis swap
    swapped = [i for i, p in enumerate(perm) if i != p]
    if len(swapped) == 2:
        a, b = swapped
        jax_fix = ShapeFix(
            title=f"Swap axes {a} and {b} with jnp.swapaxes",
            code=f"jnp.swapaxes(x, {a}, {b})",
            kind="transpose",
        )

    einops_fix = ShapeFix(
        title=f"Rearrange: ({' '.join(actual_names)}) -> ({' '.join(expected_names)})",
        code=f"einops.rearrange(x, '{' '.join(actual_names)} -> {' '.join(expected_names)}')",
        kind="transpose",
    )

    if prefer_einops:
        fixes.extend([einops_fix, jax_fix])
    else:
        fixes.extend([jax_fix, einops_fix])


def _suggest_expand(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
    expected_names: tuple[str, ...],
    actual_names: tuple[str, ...],
    prefer_einops: bool,
    fixes: list[ShapeFix],
) -> None:
    """Add expand_dims suggestions when rank is too low."""
    diff = len(expected) - len(actual)
    # Simple heuristic: add dims at the end
    axes = list(range(len(actual), len(actual) + diff))

    jax_fix = ShapeFix(
        title=f"Expand {diff} dim(s) with jnp.expand_dims",
        code=f"jnp.expand_dims(x, axis={axes if len(axes) > 1 else axes[0]})",
        kind="expand",
    )

    einops_pattern = " ".join(actual_names) + " -> " + " ".join(expected_names)
    einops_fix = ShapeFix(
        title="Rearrange to add dimension(s)",
        code=f"einops.rearrange(x, '{einops_pattern}')",
        kind="expand",
    )

    if prefer_einops:
        fixes.extend([einops_fix, jax_fix])
    else:
        fixes.extend([jax_fix, einops_fix])


def _suggest_squeeze(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
    expected_names: tuple[str, ...],
    actual_names: tuple[str, ...],
    prefer_einops: bool,
    fixes: list[ShapeFix],
) -> None:
    """Add squeeze suggestions when rank is too high."""
    diff = len(actual) - len(expected)
    # Heuristic: remove trailing dims
    axes = list(range(len(actual) - diff, len(actual)))

    jax_fix = ShapeFix(
        title=f"Remove {diff} dim(s) with jnp.squeeze",
        code=f"jnp.squeeze(x, axis={axes if len(axes) > 1 else axes[0]})",
        kind="squeeze",
    )

    einops_pattern = " ".join(actual_names) + " -> " + " ".join(expected_names)
    einops_fix = ShapeFix(
        title="Reduce to remove dimension(s)",
        code=f"einops.reduce(x, '{einops_pattern}', 'sum')",
        kind="squeeze",
    )

    if prefer_einops:
        fixes.extend([einops_fix, jax_fix])
    else:
        fixes.extend([jax_fix, einops_fix])


def _suggest_reshape(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
    expected_names: tuple[str, ...],
    actual_names: tuple[str, ...],
    prefer_einops: bool,
    fixes: list[ShapeFix],
) -> None:
    """Add reshape suggestions when sizes differ but rank matches."""
    jax_fix = ShapeFix(
        title="Reshape to expected shape",
        code=f"jnp.reshape(x, {expected})",
        kind="reshape",
    )

    einops_pattern = " ".join(actual_names) + " -> " + " ".join(expected_names)
    einops_fix = ShapeFix(
        title="Rearrange to expected shape",
        code=f"einops.rearrange(x, '{einops_pattern}')",
        kind="reshape",
    )

    if prefer_einops:
        fixes.extend([einops_fix, jax_fix])
    else:
        fixes.extend([jax_fix, einops_fix])
