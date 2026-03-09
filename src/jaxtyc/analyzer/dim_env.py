"""Dimension environment: maps dimension names to unique prime sizes for symbolic tracing."""

from __future__ import annotations

from jaxtyc.types import ShapeSpec


def _prime_sieve(limit: int) -> list[int]:
    """Sieve of Eratosthenes. Returns all primes up to `limit`."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]


class DimEnv:
    """Maps dimension names to unique prime sizes for symbolic tracing.

    Each named dimension gets a distinct prime number as its size. This ensures:
    - No two dimensions can be confused (unique sizes)
    - Reverse mapping (size -> name) is unambiguous
    - Product-based ops (reshape, flatten) produce unique results
    """

    def __init__(self, initial_sieve_limit: int = 1000) -> None:
        self._name_to_size: dict[str, int] = {}
        self._size_to_name: dict[int, str] = {}
        self._sieve_limit = initial_sieve_limit
        self._primes: list[int] = _prime_sieve(initial_sieve_limit)
        self._next_idx: int = 0

    def _next_prime(self) -> int:
        """Get the next unused prime. Extends sieve if exhausted."""
        while self._next_idx >= len(self._primes):
            self._sieve_limit *= 2
            self._primes = _prime_sieve(self._sieve_limit)
        prime = self._primes[self._next_idx]
        self._next_idx += 1
        return prime

    def get_size(self, name: str) -> int:
        """Get the prime size for a dimension name. Assigns one if new."""
        if name not in self._name_to_size:
            size = self._next_prime()
            self._name_to_size[name] = size
            self._size_to_name[size] = name
        return self._name_to_size[name]

    def resolve_name(self, size: int) -> str | None:
        """Reverse-map a size back to its dimension name, if known."""
        return self._size_to_name.get(size)

    def shape_to_names(self, shape: tuple[int, ...]) -> tuple[str | None, ...]:
        """Map a full shape tuple back to dimension names."""
        return tuple(self.resolve_name(s) for s in shape)

    def make_shape(self, spec: ShapeSpec) -> tuple[int, ...]:
        """Build a concrete shape tuple from a ShapeSpec using prime sizes."""
        shape: list[int] = []
        for dim in spec.dims:
            match dim.kind:
                case "named":
                    assert dim.name is not None
                    shape.append(self.get_size(dim.name))
                case "fixed":
                    assert dim.size is not None
                    shape.append(dim.size)
                case "variadic":
                    assert dim.name is not None
                    shape.extend(
                        [
                            self.get_size(f"_var_{dim.name}_0"),
                            self.get_size(f"_var_{dim.name}_1"),
                        ]
                    )
                case "ellipsis":
                    shape.extend(
                        [
                            self.get_size("_ellipsis_0"),
                            self.get_size("_ellipsis_1"),
                        ]
                    )
                case "anonymous":
                    shape.append(self.get_size(f"_anon_{len(shape)}"))
        return tuple(shape)

    def reset(self) -> None:
        """Clear all mappings and start fresh."""
        self._name_to_size.clear()
        self._size_to_name.clear()
        self._next_idx = 0
