"""Tuple return mismatch: second output annotation has swapped dims."""

from jaxtyping import Array
from jaxtyping import Float


def bad_duplicate(
    x: Float[Array, "batch seq d_model"],
) -> tuple[Float[Array, "batch seq d_model"], Float[Array, "batch d_model seq"]]:
    # Both outputs have shape (batch, seq, d_model), but annotation
    # says the second should be (batch, d_model, seq)
    return x, x
