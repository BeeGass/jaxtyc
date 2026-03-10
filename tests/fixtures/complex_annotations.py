"""Functions using Complex jaxtyping annotations."""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float


def to_complex(
    real: Float[Array, "batch freq"],
    imag: Float[Array, "batch freq"],
) -> Complex[Array, "batch freq"]:
    return real + 1j * imag


def magnitude(
    z: Complex[Array, "batch freq"],
) -> Float[Array, "batch freq"]:
    return jnp.abs(z)
