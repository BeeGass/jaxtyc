"""Correct sharded matmul with mesh definition and pipe annotations.

The mesh is defined via AbstractMesh (device-free) so AST inference picks it up.
The return annotation matches what JAX propagates: batch stays on 'data', d_ff is unsharded.
"""

from jax.sharding import AbstractMesh
from jaxtyping import Array
from jaxtyping import Float

mesh = AbstractMesh((2, 4), ("data", "model"))


def sharded_matmul(
    x: Float[Array, "batch|data seq|None d_model|None"],
    w: Float[Array, "d_model|None d_ff|None"],
) -> Float[Array, "batch|data seq|None d_ff|None"]:
    return x @ w
