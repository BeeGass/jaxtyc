"""Tests for DimEnv with symbolic dimensions via jax.export.symbolic_shape."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import DimSpec
from jaxtyc.types import ShapeSpec


class TestGetSize:
    def test_returns_symbolic_not_int(self) -> None:
        env = DimEnv()
        dim = env.get_size("batch")
        assert not isinstance(dim, int)

    def test_returns_named_symbolic(self) -> None:
        env = DimEnv()
        dim = env.get_size("batch")
        assert str(dim) == "batch"

    def test_idempotent_same_name(self) -> None:
        env = DimEnv()
        a = env.get_size("batch")
        b = env.get_size("batch")
        assert a is b

    def test_different_names_different_dims(self) -> None:
        env = DimEnv()
        a = env.get_size("batch")
        b = env.get_size("seq")
        assert a != b

    def test_no_reserved_parameter(self) -> None:
        env = DimEnv()
        dim = env.get_size("batch")
        assert str(dim) == "batch"


class TestResolveName:
    def test_symbolic_dim_resolves_to_name(self) -> None:
        env = DimEnv()
        dim = env.get_size("batch")
        assert env.resolve_name(dim) == "batch"

    def test_plain_int_resolves_to_none(self) -> None:
        env = DimEnv()
        assert env.resolve_name(42) is None

    def test_composite_expression_resolves(self) -> None:
        env = DimEnv()
        a = env.get_size("batch")
        b = env.get_size("seq")
        product = a * b
        result = env.resolve_name(product)
        assert result is not None
        assert "batch" in result
        assert "seq" in result


class TestShapeToNames:
    def test_all_symbolic(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        d = env.get_size("d_model")
        assert env.shape_to_names((b, s, d)) == ("batch", "seq", "d_model")

    def test_mixed_symbolic_and_fixed(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        result = env.shape_to_names((b, 128))
        assert result[0] == "batch"
        assert result[1] is None

    def test_empty_shape(self) -> None:
        env = DimEnv()
        assert env.shape_to_names(()) == ()


class TestMakeShape:
    def test_named_dim(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch"),),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 1
        assert str(shape[0]) == "batch"
        assert not isinstance(shape[0], int)

    def test_fixed_dim(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(DimSpec(kind="fixed", size=128),),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert shape == (128,)
        assert isinstance(shape[0], int)

    def test_variadic_dim_expands_to_two(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(DimSpec(kind="variadic", name="batch"),),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 2
        assert not isinstance(shape[0], int)
        assert not isinstance(shape[1], int)

    def test_ellipsis_dim_expands_to_two(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(DimSpec(kind="ellipsis"),),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 2

    def test_anonymous_dim(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(DimSpec(kind="anonymous"), DimSpec(kind="anonymous")),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 2
        assert not isinstance(shape[0], int)
        assert not isinstance(shape[1], int)
        assert shape[0] != shape[1]

    def test_mixed_dims(self) -> None:
        env = DimEnv()
        spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="fixed", size=64),
                DimSpec(kind="named", name="d_model"),
            ),
            dtype="float32",
        )
        shape = env.make_shape(spec)
        assert len(shape) == 3
        assert str(shape[0]) == "batch"
        assert shape[1] == 64
        assert str(shape[2]) == "d_model"


class TestNameSizeMapping:
    def test_returns_assigned_dims(self) -> None:
        env = DimEnv()
        env.get_size("batch")
        env.get_size("seq")
        mapping = env.name_size_mapping()
        assert "batch" in mapping
        assert "seq" in mapping
        assert str(mapping["batch"]) == "batch"


class TestReset:
    def test_reset_clears_state(self) -> None:
        env = DimEnv()
        env.get_size("batch")
        env.reset()
        mapping = env.name_size_mapping()
        assert len(mapping) == 0

    def test_reset_allows_fresh_dims(self) -> None:
        env = DimEnv()
        old = env.get_size("batch")
        env.reset()
        new = env.get_size("batch")
        assert str(new) == "batch"


class TestConsistency:
    def test_same_name_same_dim_across_shapes(self) -> None:
        env = DimEnv()
        spec_a = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch"), DimSpec(kind="named", name="seq")),
            dtype="float32",
        )
        spec_b = ShapeSpec(
            dims=(DimSpec(kind="named", name="seq"), DimSpec(kind="named", name="d_model")),
            dtype="float32",
        )
        shape_a = env.make_shape(spec_a)
        shape_b = env.make_shape(spec_b)
        assert shape_a[1] == shape_b[0]

    def test_anonymous_dims_unique_across_shapes(self) -> None:
        env = DimEnv()
        spec_a = ShapeSpec(
            dims=(DimSpec(kind="anonymous"), DimSpec(kind="named", name="dim")),
            dtype="float32",
        )
        spec_b = ShapeSpec(
            dims=(DimSpec(kind="anonymous"), DimSpec(kind="named", name="dim")),
            dtype="float32",
        )
        shape_a = env.make_shape(spec_a)
        shape_b = env.make_shape(spec_b)
        assert shape_a[0] != shape_b[0]
        assert shape_a[1] == shape_b[1]


class TestSymbolicEvalShape:
    def test_identity(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        d = env.get_size("d_model")
        result = jax.eval_shape(
            lambda x: x,
            jax.ShapeDtypeStruct((b, s, d), jnp.float32),
        )
        assert result.shape == (b, s, d)

    def test_matmul_reduces_contracting_dim(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        d = env.get_size("d_model")
        f = env.get_size("d_ff")
        result = jax.eval_shape(
            lambda x, w: x @ w,
            jax.ShapeDtypeStruct((b, s, d), jnp.float32),
            jax.ShapeDtypeStruct((d, f), jnp.float32),
        )
        assert len(result.shape) == 3
        assert str(result.shape[0]) == "batch"
        assert str(result.shape[1]) == "seq"
        assert str(result.shape[2]) == "d_ff"

    def test_elementwise_preserves_shape(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        result = jax.eval_shape(
            lambda x: jax.nn.relu(x),
            jax.ShapeDtypeStruct((b, s), jnp.float32),
        )
        assert result.shape == (b, s)

    def test_sum_reduces_dim(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        d = env.get_size("d_model")
        result = jax.eval_shape(
            lambda x: jnp.sum(x, axis=1),
            jax.ShapeDtypeStruct((b, s, d), jnp.float32),
        )
        assert len(result.shape) == 2
        assert str(result.shape[0]) == "batch"
        assert str(result.shape[1]) == "d_model"

    def test_shape_mismatch_detected(self) -> None:
        env = DimEnv()
        b = env.get_size("batch")
        s = env.get_size("seq")
        d = env.get_size("d_model")
        f = env.get_size("d_ff")
        try:
            jax.eval_shape(
                lambda x, w: x @ w,
                jax.ShapeDtypeStruct((b, s, d), jnp.float32),
                jax.ShapeDtypeStruct((f, d), jnp.float32),
            )
            raise AssertionError("Expected shape error")
        except Exception as e:
            msg = str(e)
            assert "d_model" in msg or "d_ff" in msg
