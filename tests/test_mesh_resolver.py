"""Tests for jaxtyc.analyzer.mesh_resolver — AST-based mesh and axis_rules inference."""

from __future__ import annotations

import ast

from jaxtyc.analyzer.mesh_resolver import MeshInfo
from jaxtyc.analyzer.mesh_resolver import resolve_mesh


class TestResolveMakeMesh:
    def test_make_mesh_tuple_args(self) -> None:
        source = 'import jax\nmesh = jax.make_mesh((2, 4), ("data", "model"))'
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"data": 2, "model": 4}

    def test_make_mesh_three_axes(self) -> None:
        source = 'import jax\nmesh = jax.make_mesh((2, 4, 8), ("dp", "fsdp", "tp"))'
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"dp": 2, "fsdp": 4, "tp": 8}

    def test_make_mesh_bare_name(self) -> None:
        source = 'from jax import make_mesh\nmesh = make_mesh((2, 4), ("data", "model"))'
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"data": 2, "model": 4}


class TestResolveAbstractMesh:
    def test_abstract_mesh(self) -> None:
        source = (
            'from jax.sharding import AbstractMesh\nmesh = AbstractMesh((2, 4), ("data", "model"))'
        )
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"data": 2, "model": 4}

    def test_abstract_mesh_fully_qualified(self) -> None:
        source = 'import jax\nmesh = jax.sharding.AbstractMesh((4,), ("dp",))'
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"dp": 4}


class TestResolveAxisRules:
    def test_logical_axis_rules(self) -> None:
        source = 'from flax import nnx\nnnx.logical_axis_rules([("dp", "data"), ("mp", "model")])'
        result = resolve_mesh(ast.parse(source))
        assert result.axis_rules == {"dp": "data", "mp": "model"}

    def test_logical_axis_rules_with_none(self) -> None:
        """Rules with None physical axis (replicated) are omitted."""
        source = 'from flax import nnx\nnnx.logical_axis_rules([("dp", "data"), ("embed", None)])'
        result = resolve_mesh(ast.parse(source))
        assert result.axis_rules == {"dp": "data"}

    def test_logical_axis_rules_bare(self) -> None:
        source = 'from flax.nnx import logical_axis_rules\nlogical_axis_rules([("dp", "data")])'
        result = resolve_mesh(ast.parse(source))
        assert result.axis_rules == {"dp": "data"}


class TestResolveNoMesh:
    def test_no_mesh_returns_empty(self) -> None:
        result = resolve_mesh(ast.parse("x = 42"))
        assert result.mesh == {}
        assert result.axis_rules == {}

    def test_unrelated_function_call(self) -> None:
        result = resolve_mesh(ast.parse("foo.bar((1, 2), ('a', 'b'))"))
        assert result.mesh == {}

    def test_empty_module(self) -> None:
        result = resolve_mesh(ast.parse(""))
        assert result.mesh == {}
        assert result.axis_rules == {}


class TestMeshInfo:
    def test_frozen(self) -> None:
        info = MeshInfo(mesh={"data": 4}, axis_rules={"dp": "data"})
        assert info.mesh == {"data": 4}
        assert info.axis_rules == {"dp": "data"}

    def test_defaults(self) -> None:
        info = MeshInfo()
        assert info.mesh == {}
        assert info.axis_rules == {}


class TestCombinedMeshAndRules:
    def test_both_mesh_and_rules(self) -> None:
        """File with both make_mesh and logical_axis_rules."""
        source = (
            "import jax\nfrom flax import nnx\n"
            'mesh = jax.make_mesh((2, 4), ("data", "model"))\n'
            'nnx.logical_axis_rules([("dp", "data"), ("mp", "model")])\n'
        )
        result = resolve_mesh(ast.parse(source))
        assert result.mesh == {"data": 2, "model": 4}
        assert result.axis_rules == {"dp": "data", "mp": "model"}
