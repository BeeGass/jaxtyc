"""Tests for jaxtyc.analyzer.sharding_checker — sharding validation rules."""

from __future__ import annotations

from jaxtyc.analyzer.sharding_checker import check_sharding
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShardingInfo


def _make_func_spec() -> FunctionShapeSpec:
    return FunctionShapeSpec(
        name="fn", file_path="f.py", lineno=1, col_offset=0, params={}, return_spec=None
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
