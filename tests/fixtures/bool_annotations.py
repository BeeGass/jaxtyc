"""Functions using Bool jaxtyping annotations."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float


def make_mask(
    x: Float[Array, "batch seq"],
) -> Bool[Array, "batch seq"]:
    return x > 0.0


def apply_mask(
    x: Float[Array, "batch seq"],
    mask: Bool[Array, "batch seq"],
) -> Float[Array, "batch seq"]:
    return jnp.where(mask, x, 0.0)
