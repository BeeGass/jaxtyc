"""Tests for jaxtyc.analyzer.checker — shape comparison and diagnostics."""

from __future__ import annotations

from jaxtyc.analyzer.checker import check_call_site
from jaxtyc.analyzer.checker import check_function
from jaxtyc.analyzer.dim_env import DimEnv
from jaxtyc.types import CallSite
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
    def test_matching_shapes_no_diagnostics(self) -> None:
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

    def test_shape_mismatch_detected(self) -> None:
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

    def test_rank_mismatch_detected(self) -> None:
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

    def test_no_return_spec_skips_check(self) -> None:
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

    def test_trace_error_produces_diagnostic(self) -> None:
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

    def test_any_shape_skips_check(self) -> None:
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

    def test_scalar_return_match(self) -> None:
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

    def test_scalar_return_mismatch(self) -> None:
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

    def test_fixed_dim_mismatch(self) -> None:
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

    def test_fixed_dim_no_collision(self) -> None:
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
        # Actual output matches: batch symbolic dim and literal 4
        batch_size = env.get_size("batch")
        trace = TraceResult(
            function_name="project",
            output_shape=(batch_size, 4),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        diagnostics = check_function(func_spec, trace, env)
        assert len(diagnostics) == 0

    def test_message_includes_named_shapes(self) -> None:
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


class TestCheckCallSite:
    def test_matching_callee_output_no_diagnostics(self) -> None:
        env = DimEnv()
        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=_named("batch", "hidden"),
        )
        expected_output = env.make_shape(callee_spec.return_spec)
        callee_trace = TraceResult(
            function_name="encode",
            output_shape=expected_output,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        caller_spec = FunctionShapeSpec(
            name="pipeline",
            file_path="model.py",
            lineno=15,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=_named("batch", "dim"),
        )
        call = CallSite(
            caller_name="pipeline",
            callee_name="encode",
            file_path="model.py",
            lineno=16,
            col_offset=4,
            end_col_offset=10,
        )
        diagnostics = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        assert len(diagnostics) == 0

    def test_mismatching_callee_output_produces_diagnostic(self) -> None:
        env = DimEnv()
        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=_named("batch", "hidden"),
        )
        # Actual output doesn't match annotation: returns (batch, dim) not (batch, hidden)
        wrong_output = env.make_shape(_named("batch", "dim"))
        callee_trace = TraceResult(
            function_name="encode",
            output_shape=wrong_output,
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        caller_spec = FunctionShapeSpec(
            name="pipeline",
            file_path="model.py",
            lineno=15,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=_named("batch", "dim"),
        )
        call = CallSite(
            caller_name="pipeline",
            callee_name="encode",
            file_path="model.py",
            lineno=16,
            col_offset=4,
            end_col_offset=10,
        )
        diagnostics = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        assert len(diagnostics) == 1
        assert diagnostics[0].rule == "cross-function-mismatch"

    def test_any_shape_return_no_diagnostic(self) -> None:
        """Callee with is_any_shape return should produce no diagnostics."""
        env = DimEnv()
        any_spec = ShapeSpec(dims=(), dtype="float32", is_any_shape=True)
        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={"x": _named("batch", "dim")},
            return_spec=any_spec,
        )
        callee_trace = TraceResult(
            function_name="encode",
            output_shape=(env.get_size("batch"), env.get_size("hidden")),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        caller_spec = FunctionShapeSpec(
            name="pipeline",
            file_path="model.py",
            lineno=15,
            col_offset=0,
            params={},
            return_spec=None,
        )
        call = CallSite(
            caller_name="pipeline",
            callee_name="encode",
            file_path="model.py",
            lineno=16,
            col_offset=4,
            end_col_offset=10,
        )
        diagnostics = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        assert len(diagnostics) == 0

    def test_multi_output_callee_mismatch(self) -> None:
        """Cross-function check should detect mismatches in multi-output callees."""
        env = DimEnv()
        spec_a = _named("batch", "dim")
        spec_b = _named("batch", "hidden")
        callee_spec = FunctionShapeSpec(
            name="split",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            name_col_offset=4,
            params={"x": _named("batch", "dim")},
            return_spec=spec_a,
            return_specs=[spec_a, spec_b],
        )
        shape_a = env.make_shape(spec_a)
        wrong_shape_b = env.make_shape(_named("batch", "dim"))  # wrong: dim instead of hidden
        callee_trace = TraceResult(
            function_name="split",
            output_shape=None,
            output_dtype="float32",
            intermediates=[],
            error=None,
            output_shapes=[shape_a, wrong_shape_b],
        )
        caller_spec = FunctionShapeSpec(
            name="pipeline",
            file_path="model.py",
            lineno=15,
            col_offset=0,
            params={},
            return_spec=None,
        )
        call = CallSite(
            caller_name="pipeline",
            callee_name="split",
            file_path="model.py",
            lineno=16,
            col_offset=4,
            end_col_offset=9,
        )
        diagnostics = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        assert len(diagnostics) >= 1
        assert any(d.rule == "cross-function-mismatch" for d in diagnostics)

    def test_trace_error_skips_check(self) -> None:
        env = DimEnv()
        callee_spec = FunctionShapeSpec(
            name="encode",
            file_path="model.py",
            lineno=5,
            col_offset=0,
            params={},
            return_spec=_named("batch", "hidden"),
        )
        callee_trace = TraceResult(
            function_name="encode",
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error="trace failed",
        )
        caller_spec = FunctionShapeSpec(
            name="pipeline",
            file_path="model.py",
            lineno=15,
            col_offset=0,
            params={},
            return_spec=None,
        )
        call = CallSite(
            caller_name="pipeline",
            callee_name="encode",
            file_path="model.py",
            lineno=16,
            col_offset=4,
            end_col_offset=10,
        )
        diagnostics = check_call_site(call, caller_spec, callee_spec, callee_trace, env)
        assert len(diagnostics) == 0


def test_shape_mismatch_has_related_locations() -> None:
    """Shape mismatch diagnostics should include related_locations."""
    env = DimEnv()
    spec = FunctionShapeSpec(
        name="bad_fn",
        file_path="/test.py",
        lineno=5,
        col_offset=0,
        params={"x": _named("batch", "d_model")},
        return_spec=_named("batch", "d_model"),
        name_col_offset=4,
    )
    trace = TraceResult(
        function_name="bad_fn",
        output_shape=(env.get_size("batch"), env.get_size("d_out")),
        output_dtype="float32",
        intermediates=[],
        error=None,
    )
    diags = check_function(spec, trace, env)
    shape_diags = [d for d in diags if d.rule == "shape-mismatch"]
    assert len(shape_diags) >= 1
    assert shape_diags[0].data is not None
    assert len(shape_diags[0].data.related_locations) >= 1
    rl = shape_diags[0].data.related_locations[0]
    assert rl.file_path == "/test.py"
    assert rl.line == 5


def test_cross_function_mismatch_has_related_locations() -> None:
    """Cross-function mismatch should link to callee definition."""
    env = DimEnv()
    callee = FunctionShapeSpec(
        name="encode",
        file_path="/callee.py",
        lineno=5,
        col_offset=0,
        params={"x": _named("batch")},
        return_spec=_named("batch", "hidden"),
        name_col_offset=4,
    )
    trace = TraceResult(
        function_name="encode",
        output_shape=(env.get_size("batch"), env.get_size("d_model")),
        output_dtype="float32",
        intermediates=[],
        error=None,
    )
    caller = FunctionShapeSpec(
        name="main",
        file_path="/caller.py",
        lineno=10,
        col_offset=0,
        params={},
        return_spec=None,
        name_col_offset=4,
    )
    call = CallSite(
        caller_name="main",
        callee_name="encode",
        file_path="/caller.py",
        lineno=12,
        col_offset=4,
        end_col_offset=10,
    )
    diags = check_call_site(call, caller, callee, trace, env)
    assert len(diags) >= 1
    assert diags[0].data is not None
    assert len(diags[0].data.related_locations) >= 1
    rl = diags[0].data.related_locations[0]
    assert rl.file_path == "/callee.py"
    assert rl.line == 5
    assert "encode" in rl.message
