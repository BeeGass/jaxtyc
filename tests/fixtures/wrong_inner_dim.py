"""Bug: linear projection changes last dim but annotation says it stays the same."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def linear(
    x: Float[Array, "batch seq d_model"],
    w: Float[Array, "d_model d_out"],
) -> Float[Array, "batch seq d_model"]:
    # Bug: output is (batch, seq, d_out) but annotation says (batch, seq, d_model)
    return jnp.matmul(x, w)
