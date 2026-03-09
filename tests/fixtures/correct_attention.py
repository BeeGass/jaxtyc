"""Correct attention implementation — should produce zero diagnostics."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def attention(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
    v: Float[Array, "batch heads seq head_dim"],
) -> Float[Array, "batch heads seq head_dim"]:
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)


import jax  # noqa: E402
