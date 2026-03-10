"""Functions using Shaped (dtype-agnostic) jaxtyping annotations."""

from jaxtyping import Array
from jaxtyping import Shaped


def identity(
    x: Shaped[Array, "batch features"],
) -> Shaped[Array, "batch features"]:
    return x


def reshape_flat(
    x: Shaped[Array, "batch features"],
) -> Shaped[Array, "batch features"]:
    return x
