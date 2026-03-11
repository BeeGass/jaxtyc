"""Fixture: function with a sharding rank mismatch for testing."""

import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from jaxtyping import Float

devices = np.array(jax.devices()).reshape(1, 1)
mesh = Mesh(devices, ("data", "model"))


def sharded_fn(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    # PartitionSpec has 2 entries but array has rank 3 — rank mismatch
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))
