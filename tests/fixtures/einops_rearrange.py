"""Einops rearrange with jaxtyping annotations -- test fixture for einops hints."""

import einops
from jaxtyping import Array
from jaxtyping import Float


def merge_heads(
    x: Float[Array, "batch seq heads d_head"],
) -> Float[Array, "batch seq d_model"]:
    return einops.rearrange(x, "b s h d -> b s (h d)")


def split_heads(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq heads d_head"]:
    return einops.rearrange(x, "b s (h d) -> b s h d", h=8)
