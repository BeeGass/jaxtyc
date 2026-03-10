"""Functions using Int jaxtyping annotations."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Int


def increment_ids(
    ids: Int[Array, "batch seq"],
) -> Int[Array, "batch seq"]:
    return ids + 1


def transpose_indices(
    indices: Int[Array, "rows cols"],
) -> Int[Array, "cols rows"]:
    return jnp.transpose(indices)
