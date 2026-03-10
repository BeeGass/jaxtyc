"""Functions using PRNGKeyArray jaxtyping annotations."""

import jax
from jaxtyping import PRNGKeyArray


def split_key(
    key: PRNGKeyArray,
) -> PRNGKeyArray:
    return jax.random.split(key, 1)[0]
