"""Fixture for testing inline suppression comments."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


# jaxtyc: ignore
def wrong_but_suppressed(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    """Wrong return shape but suppressed."""
    return jnp.sum(x, axis=-1)


# jaxtyc: ignore[rank-mismatch]
def wrong_specific_suppress(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    """Wrong return shape, specific rule suppressed."""
    return jnp.sum(x, axis=-1)


def wrong_not_suppressed(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    """Wrong return shape, NOT suppressed."""
    return jnp.sum(x, axis=-1)
