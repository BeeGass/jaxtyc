"""Flax NNX module fixture for testing NNX-aware tracing."""

from flax import nnx
from jaxtyping import Array
from jaxtyping import Float


class SimpleMLP(nnx.Module):
    def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "batch d_in"],
    ) -> Float[Array, "batch d_out"]:
        return self.linear(x)


class BuggyMLP(nnx.Module):
    def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "batch d_in"],
    ) -> Float[Array, "batch d_in"]:  # Bug: should be d_out
        return self.linear(x)
