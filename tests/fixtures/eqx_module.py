"""Equinox module fixture for testing equinox-aware tracing."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


class SimpleLinear(eqx.Module):
    weight: Float[Array, "d_in d_out"]
    bias: Float[Array, "d_out"]

    def __init__(self, d_in: int, d_out: int, *, key):
        self.weight = jax.random.normal(key, (d_in, d_out))
        self.bias = jnp.zeros(d_out)

    def __call__(
        self,
        x: Float[Array, "batch d_in"],
    ) -> Float[Array, "batch d_out"]:
        return x @ self.weight + self.bias


class BuggyLinear(eqx.Module):
    weight: Float[Array, "d_in d_out"]

    def __init__(self, d_in: int, d_out: int, *, key):
        self.weight = jax.random.normal(key, (d_in, d_out))

    def __call__(
        self,
        x: Float[Array, "batch d_in"],
    ) -> Float[Array, "batch d_in"]:  # Bug: should be d_out
        return x @ self.weight


import jax  # noqa: E402
