"""NNX module with divisibility constraint for testing optional-param handling."""

from flax import nnx
from jaxtyping import Array
from jaxtyping import Float


class MultiHeadLayer(nnx.Module):
    def __init__(
        self,
        features: int,
        *,
        num_head: int = 1,
        rngs: nnx.Rngs,
    ):
        assert features % num_head == 0, (
            f"features ({features}) must be divisible by num_head ({num_head})"
        )
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.num_head = num_head

    def __call__(
        self,
        x: Float[Array, "batch features"],
    ) -> Float[Array, "batch features"]:
        return self.linear(x)
