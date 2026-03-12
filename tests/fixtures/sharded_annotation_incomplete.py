"""Sharded function with incomplete annotations — some dims piped, some bare.

In strict mode, if any dim has |axis then ALL dims must have |axis or |None.
Here 'seq' is bare while 'batch' has |data, triggering sharding-annotation-incomplete.
"""

from jax.sharding import AbstractMesh
from jaxtyping import Array
from jaxtyping import Float

mesh = AbstractMesh((2, 4), ("data", "model"))


def incomplete_annotation(
    x: Float[Array, "batch|data seq d_model|None"],
) -> Float[Array, "batch|data seq d_model|None"]:
    """seq is bare while batch and d_model have pipe annotations."""
    return x
