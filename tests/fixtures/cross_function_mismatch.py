"""Cross-function shape mismatch: encode's annotation claims hidden output but returns d_model."""

from jaxtyping import Array
from jaxtyping import Float


def encode(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq hidden"]:
    # Bug: annotation says output is "hidden" but we just return x unchanged,
    # so actual output shape matches d_model, not hidden.
    return x


def decode(
    h: Float[Array, "batch seq hidden"],
) -> Float[Array, "batch seq d_model"]:
    return h


def pipeline(
    x: Float[Array, "batch seq d_model"],
) -> Float[Array, "batch seq d_model"]:
    h = encode(x)
    return decode(h)
