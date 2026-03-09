"""Functions with ellipsis and any-shape annotations — should not error."""

from jaxtyping import Array
from jaxtyping import Float


def flexible_input(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return x * 2.0


def trailing_dims(x: Float[Array, "... d_model"]) -> Float[Array, "... d_model"]:
    return x + 1.0
