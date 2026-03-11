"""NNX module with sharding constraint for testing sharding detection."""

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from jaxtyping import Float

devices = np.array(jax.devices()).reshape(1)
mesh = Mesh(devices, axis_names=("data",))


class ShardedMLP(nnx.Module):
    def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "batch d_in"],
    ) -> Float[Array, "batch d_out"]:
        y = self.linear(x)
        # Sharding constraint with deliberate rank mismatch:
        # P('data') has 1 entry for a rank-2 array (batch, d_out)
        return jax.lax.with_sharding_constraint(y, NamedSharding(mesh, P("data")))
