"""Tests for jaxtyc.analyzer.tracer — jax.eval_shape / make_jaxpr wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.tracer import trace_function
from jaxtyc.types import DimSpec
from jaxtyc.types import ShapeSpec


def _make_spec(*names: str, dtype: str = "float32") -> ShapeSpec:
    return ShapeSpec(
        dims=tuple(DimSpec(kind="named", name=n) for n in names),
        dtype=dtype,
    )


class TestTraceFunction:
    def test_simple_matmul(self):
        def matmul(a, b):
            return jnp.matmul(a, b)

        params = {
            "a": _make_spec("batch", "m", "k"),
            "b": _make_spec("batch", "k", "n"),
        }
        env = DimEnv()
        result = trace_function(matmul, params, env)
        assert result.success
        assert result.output_shape is not None
        # Output should be (batch, m, n)
        names = env.shape_to_names(result.output_shape)
        assert names == ("batch", "m", "n")

    def test_element_wise(self):
        def add_relu(x, y):
            return jax.nn.relu(x + y)

        params = {
            "x": _make_spec("batch", "dim"),
            "y": _make_spec("batch", "dim"),
        }
        env = DimEnv()
        result = trace_function(add_relu, params, env)
        assert result.success
        names = env.shape_to_names(result.output_shape)
        assert names == ("batch", "dim")

    def test_transpose(self):
        def swap_last_two(x):
            return jnp.swapaxes(x, -1, -2)

        params = {"x": _make_spec("batch", "seq", "dim")}
        env = DimEnv()
        result = trace_function(swap_last_two, params, env)
        assert result.success
        names = env.shape_to_names(result.output_shape)
        assert names == ("batch", "dim", "seq")

    def test_shape_error_reported(self):
        def bad_matmul(a, b):
            # Incompatible shapes: (batch, m, k) @ (batch, m, k)
            return jnp.matmul(a, b)

        params = {
            "a": _make_spec("batch", "m", "k"),
            "b": _make_spec("batch", "m", "k"),
        }
        env = DimEnv()
        result = trace_function(bad_matmul, params, env)
        assert not result.success
        assert result.error is not None

    def test_output_dtype(self):
        def identity(x):
            return x

        params = {"x": _make_spec("batch", "dim", dtype="float32")}
        env = DimEnv()
        result = trace_function(identity, params, env)
        assert result.success
        assert result.output_dtype == "float32"

    def test_intermediates_populated(self):
        def two_step(x):
            y = x * 2.0
            return y + 1.0

        params = {"x": _make_spec("batch", "dim")}
        env = DimEnv()
        result = trace_function(two_step, params, env)
        assert result.success
        # make_jaxpr should produce intermediates
        assert len(result.intermediates) > 0

    def test_softmax(self):
        def apply_softmax(x):
            return jax.nn.softmax(x, axis=-1)

        params = {"x": _make_spec("batch", "seq", "vocab")}
        env = DimEnv()
        result = trace_function(apply_softmax, params, env)
        assert result.success
        names = env.shape_to_names(result.output_shape)
        assert names == ("batch", "seq", "vocab")

    def test_concat(self):
        def concat_last(a, b):
            return jnp.concatenate([a, b], axis=-1)

        params = {
            "a": _make_spec("batch", "dim_a"),
            "b": _make_spec("batch", "dim_b"),
        }
        env = DimEnv()
        result = trace_function(concat_last, params, env)
        assert result.success
        # Output dim should be dim_a + dim_b (sum of two primes, not in env)
        assert result.output_shape is not None
        assert len(result.output_shape) == 2
        batch_size = env.get_size("batch")
        assert result.output_shape[0] == batch_size

    def test_fixed_dim_in_spec(self):
        def identity(x):
            return x

        spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="fixed", size=4),
                DimSpec(kind="named", name="dim"),
            ),
            dtype="float32",
        )
        env = DimEnv()
        result = trace_function(identity, {"x": spec}, env)
        assert result.success
        assert result.output_shape[1] == 4
