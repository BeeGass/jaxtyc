"""Sharded function where the same dim name is on different mesh axes.

'batch' is sharded on 'data' in param x but on 'model' in param y.
This triggers sharding-dim-conflict.
"""

from jax.sharding import AbstractMesh
from jaxtyping import Array
from jaxtyping import Float

mesh = AbstractMesh((2, 4), ("data", "model"))


def conflicting_dims(
    x: Float[Array, "batch|data seq|None"],
    y: Float[Array, "batch|model seq|None"],
) -> Float[Array, "batch|data seq|None"]:
    return x + y
