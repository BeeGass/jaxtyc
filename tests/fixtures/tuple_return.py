"""Correct tuple return: function returns multiple arrays with matching shapes."""

from jaxtyping import Array
from jaxtyping import Float


def duplicate(
    x: Float[Array, "batch seq d_model"],
) -> tuple[Float[Array, "batch seq d_model"], Float[Array, "batch seq d_model"]]:
    return x, x
