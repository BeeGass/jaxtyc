"""Matmul with pipe syntax for sharding annotations."""

from jaxtyping import Array
from jaxtyping import Float


# Pipe syntax for sharding annotations
def sharded_matmul(
    x: Float[Array, "batch|dp seq|None d_model|mp"],
    w: Float[Array, "d_model|mp d_ff|None"],
) -> Float[Array, "batch|dp seq|None d_ff|None"]:
    return x @ w
