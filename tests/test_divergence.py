"""Tests for jaxtyc.analyzer.divergence — divergence point detection."""

from __future__ import annotations

from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.analyzer.divergence import find_divergence_points
from jaxtyc.types import DimSpec
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult


def _make_func_spec(
    return_spec: ShapeSpec | None = None,
    params: dict[str, ShapeSpec] | None = None,
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
    shape: tuple[int, ...],
    named_shape: tuple[str | None, ...],
    source_line: int,
    source_file: str = "f.py",
    op_name: str = "add",
) -> IntermediateShape:
    return IntermediateShape(
        shape=shape,
        dtype="float32",
        source_file=source_file,
        source_line=source_line,
        source_col=0,
        named_shape=named_shape,
        op_name=op_name,
    )


class TestFindDivergencePoints:
    def test_find_divergence_dim_mismatch(self) -> None:
        """Detects first intermediate where a dimension diverges from expected."""
        env = DimEnv()
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("named", "seq")), dtype="float32"
        )
        expected_shape = env.make_shape(return_spec)

        func_spec = _make_func_spec(return_spec=return_spec)
        wrong_dim = env.get_size("head_dim")
        trace = TraceResult(
            function_name="fn",
            output_shape=(expected_shape[0], wrong_dim),
            output_dtype="float32",
            intermediates=[
                _make_inter(expected_shape, ("batch", "seq"), source_line=3),
                _make_inter(
                    (expected_shape[0], wrong_dim),
                    ("batch", "head_dim"),
                    source_line=5,
                    op_name="dot",
                ),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert len(result) == 1
        assert result[0].source_line == 5
        assert "head_dim" in result[0].message or "dim 1" in result[0].message

    def test_find_divergence_rank_change(self) -> None:
        """Detects rank change as divergence."""
        env = DimEnv()
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("named", "seq")), dtype="float32"
        )
        expected_shape = env.make_shape(return_spec)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=(expected_shape[0],),
            output_dtype="float32",
            intermediates=[
                _make_inter((expected_shape[0],), ("batch",), source_line=3, op_name="squeeze"),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert len(result) == 1
        assert result[0].source_line == 3
        assert "rank" in result[0].message.lower()

    def test_no_divergence_returns_empty(self) -> None:
        """All intermediates match expected -> empty list."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        expected_shape = env.make_shape(return_spec)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=expected_shape,
            output_dtype="float32",
            intermediates=[
                _make_inter(expected_shape, ("batch",), source_line=3),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert result == []

    def test_empty_intermediates_returns_empty(self) -> None:
        """No intermediates -> empty list."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=env.make_shape(return_spec),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert result == []

    def test_divergence_picks_first_by_line(self) -> None:
        """When multiple intermediates diverge, picks the first by line number."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        expected = env.make_shape(return_spec)
        wrong = (env.get_size("other"),)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=wrong,
            output_dtype="float32",
            intermediates=[
                _make_inter(wrong, ("other",), source_line=10, op_name="op1"),
                _make_inter(wrong, ("other",), source_line=5, op_name="op2"),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert len(result) == 1
        assert result[0].source_line == 5  # first by line order

    def test_divergence_skips_empty_source(self) -> None:
        """Intermediates with no source info are skipped."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        expected = env.make_shape(return_spec)
        wrong = (env.get_size("other"),)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=wrong,
            output_dtype="float32",
            intermediates=[
                _make_inter(wrong, ("other",), source_line=0, source_file="", op_name="internal"),
                _make_inter(wrong, ("other",), source_line=8, op_name="op"),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert len(result) == 1
        assert result[0].source_line == 8

    def test_divergence_no_return_spec_returns_empty(self) -> None:
        """Function with no return annotation -> empty list."""
        env = DimEnv()
        func_spec = _make_func_spec(return_spec=None)
        trace = TraceResult(
            function_name="fn",
            output_shape=(4,),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert result == []

    def test_divergence_any_shape_returns_empty(self) -> None:
        """Function with is_any_shape return -> empty list."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(), dtype="float32", is_any_shape=True)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=(4,),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert result == []

    def test_divergence_function_name_in_result(self) -> None:
        """ErrorHintInfo carries the function name."""
        env = DimEnv()
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        expected = env.make_shape(return_spec)
        wrong = (env.get_size("other"),)
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=wrong,
            output_dtype="float32",
            intermediates=[
                _make_inter(wrong, ("other",), source_line=5),
            ],
            error=None,
        )
        result = find_divergence_points(func_spec, trace, env)
        assert len(result) == 1
        assert result[0].function_name == "fn"
        assert result[0].rule == "shape-mismatch"


class TestCrossScopeDivergence:
    """Regression tests for cross-scope symbolic and concrete size issues."""

    def test_cross_scope_symbolic_no_false_positive(self) -> None:
        """Intermediates from one DimEnv scope match return spec names correctly.

        Reproduces the bug where hover_env (different SymbolicScope) caused
        'expected batch, got batch' false positives.
        """
        # Tracing env produces intermediates with named_shape from its scope
        tracing_env = DimEnv()
        tracing_env.get_size("batch")
        tracing_env.get_size("seq")

        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("named", "seq")),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        # Intermediate has correct names from the tracing env
        trace = TraceResult(
            function_name="fn",
            output_shape=(101, 103),
            output_dtype="float32",
            intermediates=[
                _make_inter((101, 103), ("batch", "seq"), source_line=5),
            ],
            error=None,
        )

        # Pass a DIFFERENT env (like LSP's hover_env) — should NOT matter
        different_env = DimEnv()
        result = find_divergence_points(func_spec, trace, different_env)
        assert result == [], f"False positive divergence: {result}"

    def test_concrete_size_mismatch_no_false_positive(self) -> None:
        """Adjusted concrete sizes from sharded tracing don't cause false positives.

        Reproduces the bug where sharding-adjusted concrete 102 didn't match
        hover_env's fresh concrete 101 for the same 'batch' dim.
        """
        return_spec = ShapeSpec(
            dims=(
                DimSpec("named", "batch"),
                DimSpec("named", "seq"),
                DimSpec("named", "d_ff"),
            ),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        # Intermediate has adjusted concrete sizes (102, 103, 107) but correct names
        trace = TraceResult(
            function_name="fn",
            output_shape=(102, 103, 107),
            output_dtype="float32",
            intermediates=[
                _make_inter((102, 103, 107), ("batch", "seq", "d_ff"), source_line=5),
            ],
            error=None,
        )

        # No env needed — comparison is purely name-based
        result = find_divergence_points(func_spec, trace)
        assert result == [], f"False positive from concrete size mismatch: {result}"

    def test_real_mismatch_still_detected_with_concrete(self) -> None:
        """Actual mismatches are still caught when shapes have concrete values."""
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("named", "seq")),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        trace = TraceResult(
            function_name="fn",
            output_shape=(102, 107),
            output_dtype="float32",
            intermediates=[
                _make_inter((102, 107), ("batch", "d_ff"), source_line=5, op_name="dot"),
            ],
            error=None,
        )

        result = find_divergence_points(func_spec, trace)
        assert len(result) == 1
        assert result[0].source_line == 5
        assert "seq" in result[0].message
        assert "d_ff" in result[0].message

    def test_fixed_dim_mismatch_detected(self) -> None:
        """Fixed dims (e.g. 128) are compared by value, not name."""
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("fixed", size=128)),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        trace = TraceResult(
            function_name="fn",
            output_shape=(101, 64),
            output_dtype="float32",
            intermediates=[
                _make_inter((101, 64), ("batch", None), source_line=5, op_name="reshape"),
            ],
            error=None,
        )

        result = find_divergence_points(func_spec, trace)
        assert len(result) == 1
        assert "128" in result[0].message
        assert "64" in result[0].message

    def test_fixed_dim_match_no_divergence(self) -> None:
        """Fixed dims that match produce no divergence."""
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("fixed", size=128)),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        trace = TraceResult(
            function_name="fn",
            output_shape=(101, 128),
            output_dtype="float32",
            intermediates=[
                _make_inter((101, 128), ("batch", None), source_line=5),
            ],
            error=None,
        )

        result = find_divergence_points(func_spec, trace)
        assert result == []

    def test_env_parameter_is_optional(self) -> None:
        """find_divergence_points works without env parameter."""
        return_spec = ShapeSpec(dims=(DimSpec("named", "batch"),), dtype="float32")
        func_spec = _make_func_spec(return_spec=return_spec)
        trace = TraceResult(
            function_name="fn",
            output_shape=(101,),
            output_dtype="float32",
            intermediates=[
                _make_inter((101,), ("batch",), source_line=3),
            ],
            error=None,
        )
        # Call without env — should not raise
        result = find_divergence_points(func_spec, trace)
        assert result == []

    def test_broadcast_intermediate_not_false_positive(self) -> None:
        """Broadcast intermediates with literal 1 and None named_shape are skipped.

        JAX creates intermediates for broadcast/constant ops with shape (1, ...)
        where named_shape is None because the literal 1 isn't in the DimEnv.
        These should not trigger 'expected batch, got 1'.
        """
        return_spec = ShapeSpec(
            dims=(
                DimSpec("named", "batch"),
                DimSpec("named", "seq"),
                DimSpec("named", "d_model"),
            ),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        trace = TraceResult(
            function_name="fn",
            output_shape=(101, 103, 105),
            output_dtype="float32",
            intermediates=[
                # Broadcast intermediate: shape (1, 103, 105) with None for dim 0
                _make_inter(
                    (1, 103, 105),
                    (None, "seq", "d_model"),
                    source_line=5,
                    op_name="broadcast_in_dim",
                ),
                # Normal intermediate matching expected
                _make_inter(
                    (101, 103, 105),
                    ("batch", "seq", "d_model"),
                    source_line=6,
                ),
            ],
            error=None,
        )

        result = find_divergence_points(func_spec, trace)
        assert result == [], f"False positive from broadcast intermediate: {result}"

    def test_all_none_named_shape_skipped(self) -> None:
        """Intermediate with all None named_shape is treated as compatible."""
        return_spec = ShapeSpec(
            dims=(DimSpec("named", "batch"), DimSpec("named", "seq")),
            dtype="float32",
        )
        func_spec = _make_func_spec(return_spec=return_spec)

        trace = TraceResult(
            function_name="fn",
            output_shape=(1, 1),
            output_dtype="float32",
            intermediates=[
                _make_inter((1, 1), (None, None), source_line=5, op_name="ones"),
            ],
            error=None,
        )

        result = find_divergence_points(func_spec, trace)
        assert result == []
