"""Tests for jaxtyc.analyzer.source_map — jaxpr source_info extraction."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.source_map import extract_source_mapped_intermediates
from jaxtyc.types import DimSpec
from jaxtyc.types import ShapeSpec


def _make_spec(*names: str, dtype: str = "float32") -> ShapeSpec:
    return ShapeSpec(
        dims=tuple(DimSpec(kind="named", name=n) for n in names),
        dtype=dtype,
    )


class TestExtractSourceMappedIntermediates:
    def test_returns_intermediates_with_shapes(self):
        def two_step(x):
            y = x * 2.0
            return y + 1.0

        env = DimEnv()
        spec = _make_spec("batch", "dim")
        abstract = jax.ShapeDtypeStruct(env.make_shape(spec), jnp.float32)

        intermediates = extract_source_mapped_intermediates(two_step, {"x": abstract}, env)
        assert len(intermediates) > 0
        # Each intermediate should have shape info
        for inter in intermediates:
            assert len(inter.shape) > 0
            assert inter.dtype != ""
            assert inter.op_name != ""

    def test_named_shapes_resolved(self):
        def identity(x):
            return x + 0.0  # Forces a jaxpr equation

        env = DimEnv()
        spec = _make_spec("batch", "dim")
        abstract = jax.ShapeDtypeStruct(env.make_shape(spec), jnp.float32)

        intermediates = extract_source_mapped_intermediates(identity, {"x": abstract}, env)
        assert len(intermediates) > 0
        # Named shapes should be resolved from env
        for inter in intermediates:
            names = inter.named_shape
            assert all(n is not None for n in names)

    def test_source_lines_populated(self):
        def matmul_fn(a, b):
            return jnp.matmul(a, b)

        env = DimEnv()
        spec_a = _make_spec("batch", "m", "k")
        spec_b = _make_spec("batch", "k", "n")
        abstract_a = jax.ShapeDtypeStruct(env.make_shape(spec_a), jnp.float32)
        abstract_b = jax.ShapeDtypeStruct(env.make_shape(spec_b), jnp.float32)

        intermediates = extract_source_mapped_intermediates(
            matmul_fn, {"a": abstract_a, "b": abstract_b}, env
        )
        assert len(intermediates) > 0
        # At least some should have source line info
        has_source = any(i.source_line > 0 for i in intermediates)
        assert has_source

    def test_empty_function(self):
        def passthrough(x):
            return x

        env = DimEnv()
        spec = _make_spec("batch", "dim")
        abstract = jax.ShapeDtypeStruct(env.make_shape(spec), jnp.float32)

        intermediates = extract_source_mapped_intermediates(passthrough, {"x": abstract}, env)
        # Pure passthrough may produce no jaxpr equations
        assert isinstance(intermediates, list)

    def test_multi_output_ops(self):
        def multi_step(x):
            a = jnp.sin(x)
            b = jnp.cos(x)
            return a + b

        env = DimEnv()
        spec = _make_spec("batch", "dim")
        abstract = jax.ShapeDtypeStruct(env.make_shape(spec), jnp.float32)

        intermediates = extract_source_mapped_intermediates(multi_step, {"x": abstract}, env)
        # Should have intermediates for sin, cos, and add
        assert len(intermediates) >= 3

    def test_filter_by_file(self):
        def compute(x):
            return jnp.sum(x, axis=-1)

        env = DimEnv()
        spec = _make_spec("batch", "dim")
        abstract = jax.ShapeDtypeStruct(env.make_shape(spec), jnp.float32)

        intermediates = extract_source_mapped_intermediates(compute, {"x": abstract}, env)
        # No intermediate should reference JAX internal files
        for inter in intermediates:
            if inter.source_file:
                assert "site-packages/jax" not in inter.source_file
