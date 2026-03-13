"""Tests for jaxtyc.analyzer.sharding_checker — sharding validation rules."""

from __future__ import annotations

from unittest.mock import MagicMock

from jaxtyc.analyzer.sharding_checker import check_annotation_sharding
from jaxtyc.analyzer.sharding_checker import check_sharding
from jaxtyc.analyzer.sharding_checker import check_sharding_propagation
from jaxtyc.types import DimSpec
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import ShardingInfo


def _make_func_spec(
    params: dict[str, ShapeSpec] | None = None,
    return_spec: ShapeSpec | None = None,
) -> FunctionShapeSpec:
    return FunctionShapeSpec(
        name="fn",
        file_path="f.py",
        lineno=1,
        col_offset=0,
        params=params or {},
        return_spec=return_spec,
    )


def _make_inter(
    shape: tuple[int, ...] = (4, 8, 16),
    named_shape: tuple[str | None, ...] = ("b", "s", "d"),
    source_line: int = 5,
    op_name: str = "constraint",
    sharding: ShardingInfo | None = None,
) -> IntermediateShape:
    return IntermediateShape(
        shape=shape,
        dtype="float32",
        source_file="f.py",
        source_line=source_line,
        source_col=0,
        named_shape=named_shape,
        op_name=op_name,
        sharding=sharding,
    )


class TestShardingRankMismatch:
    def test_sharding_rank_mismatch(self) -> None:
        """PartitionSpec with wrong number of entries is caught."""
        inter = _make_inter(
            sharding=ShardingInfo(
                partition_spec=("data", None),  # 2 entries for rank-3 array
                mesh_axis_names=("data", "model"),
                source_primitive="sharding_constraint",
                source_line=5,
            ),
        )
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert any(d.rule == "sharding-rank-mismatch" for d in diags)

    def test_sharding_rank_matches(self) -> None:
        """Correct rank produces no rank mismatch diagnostic."""
        inter = _make_inter(
            sharding=ShardingInfo(
                partition_spec=("data", None, None),
                mesh_axis_names=("data", "model"),
                source_primitive="sharding_constraint",
                source_line=5,
            ),
        )
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert not any(d.rule == "sharding-rank-mismatch" for d in diags)


class TestShardingAxisUnknown:
    def test_sharding_axis_unknown(self) -> None:
        """PartitionSpec referencing non-existent mesh axis is caught."""
        inter = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            sharding=ShardingInfo(
                partition_spec=("dp", None),  # 'dp' not in mesh
                mesh_axis_names=("data", "model"),
                source_primitive="sharding_constraint",
                source_line=5,
            ),
        )
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert any(d.rule == "sharding-axis-unknown" for d in diags)
        axis_diag = next(d for d in diags if d.rule == "sharding-axis-unknown")
        assert "dp" in axis_diag.message

    def test_sharding_axis_known(self) -> None:
        """Valid axis names produce no unknown-axis diagnostic."""
        inter = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            sharding=ShardingInfo(
                partition_spec=("data", None),
                mesh_axis_names=("data", "model"),
                source_primitive="sharding_constraint",
                source_line=5,
            ),
        )
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert not any(d.rule == "sharding-axis-unknown" for d in diags)


class TestShardingConflict:
    def test_sharding_conflict(self) -> None:
        """Conflicting PartitionSpecs on same shape at same line are caught."""
        base_kwargs = {
            "shape": (4, 8),
            "named_shape": ("b", "s"),
            "source_line": 5,
            "op_name": "constraint",
        }
        inter1 = _make_inter(
            **base_kwargs,
            sharding=ShardingInfo(("data", None), ("data", "model"), "sharding_constraint", 5),
        )
        inter2 = _make_inter(
            **base_kwargs,
            sharding=ShardingInfo((None, "model"), ("data", "model"), "sharding_constraint", 5),
        )
        diags = check_sharding([inter1, inter2], _make_func_spec(), "f.py")
        assert any(d.rule == "sharding-conflict" for d in diags)

    def test_no_conflict_different_lines(self) -> None:
        """Different sharding specs on different lines is not a conflict."""
        inter1 = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            source_line=5,
            sharding=ShardingInfo(("data", None), ("data", "model"), "sharding_constraint", 5),
        )
        inter2 = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            source_line=10,
            sharding=ShardingInfo((None, "model"), ("data", "model"), "sharding_constraint", 10),
        )
        diags = check_sharding([inter1, inter2], _make_func_spec(), "f.py")
        assert not any(d.rule == "sharding-conflict" for d in diags)


class TestShardingIOMismatch:
    def test_sharding_io_mismatch(self) -> None:
        """jit out_shardings contradicting inner constraint is caught (warning)."""
        jit_inter = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            source_line=1,
            op_name="jit",
            sharding=ShardingInfo(("data", None), ("data", "model"), "jit", 1),
        )
        constraint_inter = _make_inter(
            shape=(4, 8),
            named_shape=("b", "s"),
            source_line=5,
            op_name="sharding_constraint",
            sharding=ShardingInfo((None, "model"), ("data", "model"), "sharding_constraint", 5),
        )
        diags = check_sharding([jit_inter, constraint_inter], _make_func_spec(), "f.py")
        assert any(d.rule == "sharding-io-mismatch" for d in diags)
        io_diag = next(d for d in diags if d.rule == "sharding-io-mismatch")
        assert io_diag.severity == "warning"


class TestNoSharding:
    def test_no_sharding_no_diagnostics(self) -> None:
        """Intermediates without sharding produce no sharding diagnostics."""
        inter = _make_inter(sharding=None)
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert diags == []

    def test_valid_sharding_no_diagnostics(self) -> None:
        """Correct sharding produces no diagnostics."""
        inter = _make_inter(
            sharding=ShardingInfo(
                partition_spec=("data", None, None),
                mesh_axis_names=("data", "model"),
                source_primitive="sharding_constraint",
                source_line=5,
            ),
        )
        diags = check_sharding([inter], _make_func_spec(), "f.py")
        assert diags == []

    def test_empty_intermediates(self) -> None:
        """Empty intermediates list produces no diagnostics."""
        diags = check_sharding([], _make_func_spec(), "f.py")
        assert diags == []


class TestCheckShardingPropagation:
    def _make_func_spec_with_return(
        self, return_spec: ShapeSpec | None = None
    ) -> FunctionShapeSpec:
        return FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={},
            return_spec=return_spec,
        )

    def test_match_no_diagnostic(self) -> None:
        """No diagnostic when propagated sharding matches return annotation."""
        propagated = MagicMock()
        propagated.spec = ("data", None, "model")
        return_spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch", mesh_axis="data"),
                DimSpec(kind="named", name="seq", mesh_axis=None),
                DimSpec(kind="named", name="d_model", mesh_axis="model"),
            ),
            dtype="float32",
        )
        diags = check_sharding_propagation(
            propagated_sharding=propagated,
            return_spec=return_spec,
            func_spec=self._make_func_spec_with_return(return_spec),
            file_path="f.py",
        )
        assert len(diags) == 0

    def test_mismatch_emits_diagnostic(self) -> None:
        """Emit sharding-propagation-mismatch when propagated != annotated."""
        propagated = MagicMock()
        propagated.spec = ("data", None, None)  # d_model not sharded
        return_spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch", mesh_axis="data"),
                DimSpec(kind="named", name="seq", mesh_axis=None),
                DimSpec(kind="named", name="d_model", mesh_axis="model"),
            ),
            dtype="float32",
        )
        diags = check_sharding_propagation(
            propagated_sharding=propagated,
            return_spec=return_spec,
            func_spec=self._make_func_spec_with_return(return_spec),
            file_path="f.py",
        )
        assert len(diags) == 1
        assert diags[0].rule == "sharding-propagation-mismatch"

    def test_no_propagated_sharding(self) -> None:
        """No diagnostic when propagated_sharding is None."""
        return_spec = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch", mesh_axis="data"),),
            dtype="float32",
        )
        diags = check_sharding_propagation(
            propagated_sharding=None,
            return_spec=return_spec,
            func_spec=self._make_func_spec_with_return(return_spec),
            file_path="f.py",
        )
        assert len(diags) == 0

    def test_no_return_spec(self) -> None:
        """No diagnostic when return_spec is None."""
        propagated = MagicMock()
        propagated.spec = ("data",)
        diags = check_sharding_propagation(
            propagated_sharding=propagated,
            return_spec=None,
            func_spec=self._make_func_spec_with_return(None),
            file_path="f.py",
        )
        assert len(diags) == 0

    def test_return_spec_no_sharding(self) -> None:
        """No diagnostic when return_spec has no sharding annotations."""
        propagated = MagicMock()
        propagated.spec = ("data", None)
        return_spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="named", name="seq"),
            ),
            dtype="float32",
        )
        diags = check_sharding_propagation(
            propagated_sharding=propagated,
            return_spec=return_spec,
            func_spec=self._make_func_spec_with_return(return_spec),
            file_path="f.py",
        )
        assert len(diags) == 0


class TestAnnotationIncomplete:
    def test_strict_incomplete_annotation(self) -> None:
        """Piped shape with bare dims in strict mode emits diagnostic."""
        func_spec = FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch", mesh_axis="dp"),
                        DimSpec(kind="named", name="seq"),  # bare dim
                    ),
                    dtype="float32",
                ),
            },
            return_spec=None,
        )
        diags = check_annotation_sharding(func_spec, "f.py", strict=True)
        assert any(d.rule == "sharding-annotation-incomplete" for d in diags)

    def test_no_sharding_no_diagnostic(self) -> None:
        """No pipe annotations at all produces no diagnostic."""
        func_spec = FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch"),
                        DimSpec(kind="named", name="seq"),
                    ),
                    dtype="float32",
                ),
            },
            return_spec=None,
        )
        diags = check_annotation_sharding(func_spec, "f.py", strict=True)
        assert not any(d.rule == "sharding-annotation-incomplete" for d in diags)

    def test_non_strict_no_diagnostic(self) -> None:
        """Non-strict mode does not emit incomplete annotation diagnostic."""
        func_spec = FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(
                        DimSpec(kind="named", name="batch", mesh_axis="dp"),
                        DimSpec(kind="named", name="seq"),
                    ),
                    dtype="float32",
                ),
            },
            return_spec=None,
        )
        diags = check_annotation_sharding(func_spec, "f.py", strict=False)
        assert not any(d.rule == "sharding-annotation-incomplete" for d in diags)


class TestCheckMeshAxes:
    def test_undefined_axis_emits_diagnostic(self) -> None:
        from jaxtyc.analyzer.sharding_checker import check_mesh_axes

        func_spec = _make_func_spec(
            params={
                "x": ShapeSpec(
                    dims=(DimSpec("named", "batch", mesh_axis="dp"),),
                    dtype="float32",
                )
            },
        )
        result = check_mesh_axes(func_spec, "f.py", {"data": 4}, {})
        assert len(result) == 1
        assert result[0].rule == "sharding-mesh-undefined"
        assert "dp" in result[0].message

    def test_physical_axis_in_mesh_no_diagnostic(self) -> None:
        from jaxtyc.analyzer.sharding_checker import check_mesh_axes

        func_spec = _make_func_spec(
            params={
                "x": ShapeSpec(
                    dims=(DimSpec("named", "batch", mesh_axis="data"),),
                    dtype="float32",
                )
            },
        )
        result = check_mesh_axes(func_spec, "f.py", {"data": 4}, {})
        assert result == []

    def test_logical_axis_in_rules_no_diagnostic(self) -> None:
        from jaxtyc.analyzer.sharding_checker import check_mesh_axes

        func_spec = _make_func_spec(
            params={
                "x": ShapeSpec(
                    dims=(DimSpec("named", "batch", mesh_axis="dp"),),
                    dtype="float32",
                )
            },
        )
        result = check_mesh_axes(func_spec, "f.py", {"data": 4}, {"dp": "data"})
        assert result == []

    def test_none_mesh_axis_no_diagnostic(self) -> None:
        from jaxtyc.analyzer.sharding_checker import check_mesh_axes

        func_spec = _make_func_spec(
            params={
                "x": ShapeSpec(
                    dims=(DimSpec("named", "batch", mesh_axis=None),),
                    dtype="float32",
                )
            },
        )
        result = check_mesh_axes(func_spec, "f.py", {"data": 4}, {})
        assert result == []


class TestDimConflict:
    def test_dim_sharded_differently(self) -> None:
        """Same dim sharded on different axes across params emits warning."""
        func_spec = FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch", mesh_axis="data"),),
                    dtype="float32",
                ),
                "y": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch", mesh_axis="model"),),
                    dtype="float32",
                ),
            },
            return_spec=None,
        )
        diags = check_annotation_sharding(func_spec, "f.py")
        assert any(d.rule == "sharding-dim-conflict" for d in diags)

    def test_dim_consistent_no_conflict(self) -> None:
        """Same dim same axis across params produces no conflict."""
        func_spec = FunctionShapeSpec(
            name="fn",
            file_path="f.py",
            lineno=1,
            col_offset=0,
            params={
                "x": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch", mesh_axis="data"),),
                    dtype="float32",
                ),
                "y": ShapeSpec(
                    dims=(DimSpec(kind="named", name="batch", mesh_axis="data"),),
                    dtype="float32",
                ),
            },
            return_spec=None,
        )
        diags = check_annotation_sharding(func_spec, "f.py")
        assert not any(d.rule == "sharding-dim-conflict" for d in diags)
