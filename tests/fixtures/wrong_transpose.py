"""Bug: swapped axes produce wrong output shape. Should produce a shape-mismatch."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def attention(
    q: Float[Array, "batch heads seq head_dim"],
    k: Float[Array, "batch heads seq head_dim"],
) -> Float[Array, "batch heads seq seq"]:
    # Bug: transposes q instead of k, then matmul produces (batch, heads, head_dim, seq)
    # instead of (batch, heads, seq, seq)
    return jnp.matmul(jnp.swapaxes(q, -1, -2), k)
