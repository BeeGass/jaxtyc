"""Multiple shape-annotated functions that call each other."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def encode(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq hidden"]:
    w = jnp.ones((7, 11))  # d_model -> hidden
    return jnp.matmul(x, w)


def decode(
    h: Float[Array, "batch seq hidden"],
) -> Float[Array, "batch seq d_model"]:
    w = jnp.ones((11, 7))  # hidden -> d_model
    return jnp.matmul(h, w)


def autoencoder(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    h = encode(x)
    return decode(h)
