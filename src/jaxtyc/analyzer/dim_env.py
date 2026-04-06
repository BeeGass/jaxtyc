"""Dimension environment: maps dimension names to symbolic sizes for tracing."""

from __future__ import annotations

from typing import Any

from jax.export import SymbolicScope
from jax.export import symbolic_shape

from jaxtyc.types import ShapeSpec

DimSize = Any


class DimEnv:
    """Maps dimension names to symbolic sizes for tracing.

    Each named dimension gets a distinct symbolic dimension object via
    ``jax.export.symbolic_shape``. This ensures:
    - No two dimensions can be confused (distinct symbolic objects)
    - Reverse mapping (dim -> name) is trivial via ``str(dim)``
    - No integer overflow from large prime products
    - Error messages from JAX use symbolic names directly
    """

    _MIN_CONCRETE: int = 101

    def __init__(self) -> None:
        self._scope = SymbolicScope()
        self._dims: dict[str, DimSize] = {}
        self._anon_counter: int = 0
        self._anon_concrete_counter: int = 0
        self._concrete: dict[str, int] = {}
        self._next_concrete: int = self._MIN_CONCRETE

    def get_size(self, name: str) -> DimSize:
        """Get the symbolic size for a dimension name. Creates one if new.

        Args:
            name: Symbolic dimension name (e.g. "batch", "seq").

        Returns:
            Symbolic dimension object uniquely assigned to this name.
        """
        if name not in self._dims:
            (dim,) = symbolic_shape(name, scope=self._scope)
            self._dims[name] = dim
        return self._dims[name]

    def resolve_name(self, size: DimSize) -> str | None:
        """Reverse-map a size back to its dimension name.

        For symbolic dims, returns ``str(size)``. For concrete ints from
        ``get_concrete_size``, looks up the reverse mapping. Returns None
        for unknown plain ints.

        Args:
            size: Dimension size (symbolic or plain int).

        Returns:
            Dimension name string, or None for unknown sizes.
        """
        if isinstance(size, int):
            return self.resolve_concrete_name(size)
        return str(size)

    def shape_to_names(self, shape: tuple[DimSize, ...]) -> tuple[str | None, ...]:
        """Map a full shape tuple back to dimension names.

        Args:
            shape: Shape tuple from JAX tracing (may contain symbolic dims or ints).

        Returns:
            Tuple of dimension name strings (or None for plain ints).
        """
        return tuple(self.resolve_name(s) for s in shape)

    def make_shape(self, spec: ShapeSpec) -> tuple[DimSize, ...]:
        """Build a concrete/symbolic shape tuple from a ShapeSpec.

        Args:
            spec: Parsed shape specification to materialise.

        Returns:
            Shape tuple where named dims are symbolic _DimExpr objects,
            fixed dims are plain ints, and variadic/ellipsis expand to two dims.
        """
        shape: list[DimSize] = []
        for i, dim in enumerate(spec.dims):
            match dim.kind:
                case "named":
                    if dim.name is None:
                        msg = f"DimSpec(kind='named') requires name, got None at index {i}"
                        raise ValueError(msg)
                    shape.append(self.get_size(dim.name))
                case "fixed":
                    if dim.size is None:
                        msg = f"DimSpec(kind='fixed') requires size, got None at index {i}"
                        raise ValueError(msg)
                    shape.append(dim.size)
                case "variadic":
                    if dim.name is None:
                        msg = f"DimSpec(kind='variadic') requires name, got None at index {i}"
                        raise ValueError(msg)
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
                    self._anon_counter += 1
                    shape.append(self.get_size(f"_anon_{self._anon_counter}"))
        return tuple(shape)

    def get_concrete_size(self, name: str) -> int:
        """Get a concrete integer size for module construction.

        Used by NNX/equinox module tracing where constructors need int args.
        Each name gets a unique odd integer >= 101.

        Args:
            name: Dimension name.

        Returns:
            Unique concrete integer for this dimension name.
        """
        if name not in self._concrete:
            self._concrete[name] = self._next_concrete
            self._next_concrete += 2
        return self._concrete[name]

    def make_concrete_shape(self, spec: ShapeSpec) -> tuple[int, ...]:
        """Build a concrete int shape tuple from a ShapeSpec.

        Uses ``get_concrete_size`` instead of ``get_size``, producing
        plain integers suitable for NNX/equinox module construction
        where symbolic dims are not accepted.

        Args:
            spec: Parsed shape specification to materialise.

        Returns:
            Shape tuple of plain integers.
        """
        shape: list[int] = []
        for i, dim in enumerate(spec.dims):
            match dim.kind:
                case "named":
                    if dim.name is None:
                        msg = f"DimSpec(kind='named') requires name, got None at index {i}"
                        raise ValueError(msg)
                    shape.append(self.get_concrete_size(dim.name))
                case "fixed":
                    if dim.size is None:
                        msg = f"DimSpec(kind='fixed') requires size, got None at index {i}"
                        raise ValueError(msg)
                    shape.append(dim.size)
                case "variadic":
                    if dim.name is None:
                        msg = f"DimSpec(kind='variadic') requires name, got None at index {i}"
                        raise ValueError(msg)
                    shape.extend(
                        [
                            self.get_concrete_size(f"_var_{dim.name}_0"),
                            self.get_concrete_size(f"_var_{dim.name}_1"),
                        ]
                    )
                case "ellipsis":
                    shape.extend(
                        [
                            self.get_concrete_size("_ellipsis_0"),
                            self.get_concrete_size("_ellipsis_1"),
                        ]
                    )
                case "anonymous":
                    self._anon_concrete_counter += 1
                    shape.append(self.get_concrete_size(f"_anon_c_{self._anon_concrete_counter}"))
        return tuple(shape)

    def resolve_concrete_name(self, size: int) -> str | None:
        """Reverse-map a concrete size back to its dimension name.

        Args:
            size: Concrete integer size from ``get_concrete_size``.

        Returns:
            Dimension name if known, else None.
        """
        for name, s in self._concrete.items():
            if s == size:
                return name
        return None

    def name_size_mapping(self) -> dict[str, DimSize]:
        """Return a copy of all name-to-size mappings (symbolic + concrete).

        Returns:
            Dictionary mapping dimension names to their symbolic or concrete sizes.
        """
        result: dict[str, DimSize] = dict(self._dims)
        result.update(self._concrete)
        return result

    def reset(self) -> None:
        """Clear all name-to-size mappings and create a fresh scope."""
        self._dims.clear()
        self._scope = SymbolicScope()
        self._anon_counter = 0
        self._anon_concrete_counter = 0
        self._concrete.clear()
        self._next_concrete = self._MIN_CONCRETE
