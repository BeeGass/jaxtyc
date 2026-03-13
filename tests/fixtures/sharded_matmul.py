"""Standard matmul — no pipe syntax, should pass shape checking."""

from jaxtyping import Array
from jaxtyping import Float


def sharded_matmul(
    x: Float[Array, "batch seq d_model"],
    w: Float[Array, "d_model d_ff"],
) -> Float[Array, "batch seq d_ff"]:
    return x @ w
