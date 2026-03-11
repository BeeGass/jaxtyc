"""Tests for sharding extraction from jaxpr in tracer.py."""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.tracer import trace_function
from jaxtyc.types import DimSpec
from jaxtyc.types import ShapeSpec


def _single_device_mesh() -> Mesh:
    """Create a single-device mesh for testing."""
    devices = np.array(jax.devices()).reshape(1, 1)
    return Mesh(devices, ("data", "model"))


class TestShardingExtraction:
    def test_extract_sharding_constraint(self) -> None:
        """with_sharding_constraint produces IntermediateShape with ShardingInfo."""
        mesh = _single_device_mesh()

        def fn(x: jax.Array) -> jax.Array:
            return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))

        env = DimEnv()
        params = {
            "x": ShapeSpec(
                dims=(DimSpec("named", "batch"), DimSpec("named", "seq")),
                dtype="float32",
            )
        }
        result = trace_function(fn, params, env)
        assert result.success

        sharded = [i for i in result.intermediates if i.sharding is not None]
        assert len(sharded) >= 1, "Expected at least one intermediate with sharding info"

        info = sharded[0].sharding
        assert info is not None
        assert info.source_primitive == "sharding_constraint"
        assert info.partition_spec == ("data", None)
        assert "data" in info.mesh_axis_names
        assert "model" in info.mesh_axis_names

    def test_no_sharding_returns_none(self) -> None:
        """Plain function intermediates have sharding=None."""

        def fn(x: jax.Array, y: jax.Array) -> jax.Array:
            return x + y

        env = DimEnv()
        params = {
            "x": ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32"),
            "y": ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32"),
        }
        result = trace_function(fn, params, env)
        assert result.success

        for inter in result.intermediates:
            assert inter.sharding is None, (
                f"Expected no sharding for plain add, got {inter.sharding}"
            )

    def test_sharding_info_partition_spec_values(self) -> None:
        """ShardingInfo carries correct PartitionSpec entries."""
        mesh = _single_device_mesh()

        def fn(x: jax.Array) -> jax.Array:
            return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None, None)))

        env = DimEnv()
        params = {
            "x": ShapeSpec(
                dims=(DimSpec("named", "b"), DimSpec("named", "s"), DimSpec("named", "d")),
                dtype="float32",
            )
        }
        result = trace_function(fn, params, env)
        assert result.success

        sharded = [i for i in result.intermediates if i.sharding is not None]
        assert len(sharded) >= 1
        assert sharded[0].sharding is not None
        assert sharded[0].sharding.partition_spec == ("data", None, None)

    def test_multiple_sharding_constraints(self) -> None:
        """Multiple sharding constraints produce multiple ShardingInfo entries."""
        mesh = _single_device_mesh()

        def fn(x: jax.Array) -> jax.Array:
            y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None)))
            z = y * 2
            return jax.lax.with_sharding_constraint(z, NamedSharding(mesh, P(None, "model")))

        env = DimEnv()
        params = {
            "x": ShapeSpec(
                dims=(DimSpec("named", "batch"), DimSpec("named", "seq")),
                dtype="float32",
            )
        }
        result = trace_function(fn, params, env)
        assert result.success

        sharded = [i for i in result.intermediates if i.sharding is not None]
        assert len(sharded) >= 2
        specs = [s.sharding.partition_spec for s in sharded if s.sharding is not None]
        assert ("data", None) in specs
        assert (None, "model") in specs
