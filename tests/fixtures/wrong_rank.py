"""Bug: sum reduces a dimension, producing wrong rank."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def project(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    # Bug: sum over last axis collapses it — output is (batch, seq) not (batch, seq, d_model)
    return jnp.sum(x, axis=-1)
