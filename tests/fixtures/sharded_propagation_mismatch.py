"""Sharded matmul where return annotation claims sharding JAX does not propagate.

JAX propagates batch|data through matmul, but d_ff is NOT sharded on 'model'
since neither input has d_ff sharded. The return annotation incorrectly claims
d_ff|model, which should trigger sharding-propagation-mismatch.
"""

from jax.sharding import AbstractMesh
from jaxtyping import Array
from jaxtyping import Float

mesh = AbstractMesh((2, 4), ("data", "model"))


def sharded_matmul_wrong_return(
    x: Float[Array, "batch|data seq|None d_model|None"],
    w: Float[Array, "d_model|None d_ff|None"],
) -> Float[Array, "batch|data seq|None d_ff|model"]:
    """Return claims d_ff is on 'model' but JAX won't propagate that."""
    return x @ w
