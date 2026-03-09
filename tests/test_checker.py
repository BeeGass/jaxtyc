"""Tests for jaxtyc.analyzer.checker — shape comparison and diagnostics."""

from __future__ import annotations

from jaxtyc.analyzer.checker import check_function
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import DimSpec
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult


def _named(*names: str, dtype: str = "float32") -> ShapeSpec:
    return ShapeSpec(
        dims=tuple(DimSpec(kind="named", name=n) for n in names),
        dtype=dtype,
    )


class TestCheckFunction:
    def test_matching_shapes_no_diagnostics(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="attention",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={"q": _named("batch", "heads", "seq", "head_dim")},
            return_spec=_named("batch", "heads", "seq", "seq"),
        )
        # Build the expected output shape from the return spec
        expected_output = env.make_shape(func_spec.return_spec)
        trace = TraceResult(
            function_name="attention",
            output_shape=expected_output,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 0

    def test_shape_mismatch_detected(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="attention",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=_named("batch", "heads", "seq", "seq"),
        )
        # Actual output has head_dim instead of seq in last two dims (transposed bug)
        wrong_shape = env.make_shape(_named("batch", "heads", "head_dim", "head_dim"))
        trace = TraceResult(
            function_name="attention",
            output_shape=wrong_shape,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert diag.severity == "error"
        assert diag.rule == "shape-mismatch"
        assert "batch" in diag.message
        assert "seq" in diag.message or "head_dim" in diag.message

    def test_rank_mismatch_detected(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="project",
            file_path="model.py",
            lineno=10,
            col_offset=0,
            params={},
            return_spec=_named("batch", "seq", "d_model"),
        )
        # Actual output has wrong rank (2 dims instead of 3)
        wrong_shape = env.make_shape(_named("batch", "seq"))
        trace = TraceResult(
            function_name="project",
            output_shape=wrong_shape,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        assert diagnostics[0].rule == "rank-mismatch"

    def test_no_return_spec_skips_check(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="process",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=None,
        )
        trace = TraceResult(
            function_name="process",
            output_shape=(2, 3),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 0

    def test_trace_error_produces_diagnostic(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="broken",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=_named("batch", "dim"),
        )
        trace = TraceResult(
            function_name="broken",
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error="Shape error in matmul",
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].rule == "trace-error"

    def test_any_shape_skips_check(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="flexible",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=ShapeSpec(dims=(), dtype="float32", is_any_shape=True),
        )
        trace = TraceResult(
            function_name="flexible",
            output_shape=(2, 3, 5, 7),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 0

    def test_scalar_return_match(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="loss",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=ShapeSpec(dims=(), dtype="float32", is_scalar=True),
        )
        trace = TraceResult(
            function_name="loss",
            output_shape=(),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 0

    def test_scalar_return_mismatch(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="loss",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=ShapeSpec(dims=(), dtype="float32", is_scalar=True),
        )
        # Actual output is not scalar
        trace = TraceResult(
            function_name="loss",
            output_shape=(2, 3),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        assert diagnostics[0].rule == "rank-mismatch"

    def test_fixed_dim_mismatch(self):
        env = DimEnv()
        ret_spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="fixed", size=4),
            ),
            dtype="float32",
        )
        func_spec = FunctionShapeSpec(
            name="project",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=ret_spec,
        )
        # Actual output has batch_prime and 8 instead of 4
        batch_size = env.get_size("batch")
        trace = TraceResult(
            function_name="project",
            output_shape=(batch_size, 8),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        assert diagnostics[0].rule == "shape-mismatch"

    def test_message_includes_named_shapes(self):
        env = DimEnv()
        func_spec = FunctionShapeSpec(
            name="attention",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=_named("batch", "heads", "seq", "seq"),
        )
        wrong_shape = env.make_shape(_named("batch", "heads", "head_dim", "head_dim"))
        trace = TraceResult(
            function_name="attention",
            output_shape=wrong_shape,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 1
        msg = diagnostics[0].message
        # Should mention the expected and actual named shapes
        assert "Expected" in msg or "expected" in msg
        assert "Got" in msg or "got" in msg
