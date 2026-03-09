"""Functions that cannot be JAX-traced — should be gracefully skipped."""


def plain_python(x: int, y: int) -> int:
    return x + y


async def async_fn(x: float) -> float:
    return x * 2
