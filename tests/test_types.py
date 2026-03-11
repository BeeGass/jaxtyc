"""Tests for jaxtyc.types — core data types."""

from __future__ import annotations

from jaxtyc.types import Diagnostic
from jaxtyc.types import DimSpec
from jaxtyc.types import ErrorHintInfo
from jaxtyc.types import FileResult
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import ShardingInfo
from jaxtyc.types import TraceResult


class TestDimSpec:
    def test_named_dim(self) -> None:
        d = DimSpec(kind="named", name="batch")
        assert d.kind == "named"
        assert d.name == "batch"
        assert d.size is None

    def test_fixed_dim(self) -> None:
        d = DimSpec(kind="fixed", size=4)
        assert d.kind == "fixed"
        assert d.size == 4
        assert d.name is None

    def test_variadic_dim(self) -> None:
        d = DimSpec(kind="variadic", name="batch")
        assert d.kind == "variadic"
        assert d.name == "batch"

    def test_anonymous_dim(self) -> None:
        d = DimSpec(kind="anonymous")
        assert d.kind == "anonymous"

    def test_ellipsis_dim(self) -> None:
        d = DimSpec(kind="ellipsis")
        assert d.kind == "ellipsis"

    def test_frozen(self) -> None:
        import pytest

        d = DimSpec(kind="named", name="seq")
        with pytest.raises(AttributeError):
            d.name = "other"  # type: ignore[misc]


class TestShapeSpec:
    def test_basic_shape(self) -> None:
        dims = (
            DimSpec(kind="named", name="batch"),
            DimSpec(kind="named", name="seq"),
            DimSpec(kind="named", name="d_model"),
        )
        spec = ShapeSpec(dims=dims, dtype="float32")
        assert len(spec.dims) == 3
        assert spec.dtype == "float32"
        assert not spec.is_scalar
        assert not spec.is_any_shape

    def test_scalar(self) -> None:
        spec = ShapeSpec(dims=(), dtype="float32", is_scalar=True)
        assert spec.is_scalar
        assert len(spec.dims) == 0

    def test_any_shape(self) -> None:
        spec = ShapeSpec(dims=(), dtype="float32", is_any_shape=True)
        assert spec.is_any_shape

    def test_frozen(self) -> None:
        import pytest

        spec = ShapeSpec(dims=(), dtype="float32")
        with pytest.raises(AttributeError):
            spec.dtype = "int32"  # type: ignore[misc]


class TestFunctionShapeSpec:
    def test_basic(self) -> None:
        param = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch"),),
            dtype="float32",
        )
        ret = ShapeSpec(
            dims=(DimSpec(kind="named", name="batch"),),
            dtype="float32",
        )
        func = FunctionShapeSpec(
            name="forward",
            file_path="model.py",
            lineno=10,
            col_offset=0,
            params={"x": param},
            return_spec=ret,
        )
        assert func.name == "forward"
        assert "x" in func.params
        assert func.return_spec is not None
        assert not func.is_method
        assert func.class_name is None

    def test_method(self) -> None:
        func = FunctionShapeSpec(
            name="__call__",
            file_path="model.py",
            lineno=5,
            col_offset=4,
            params={},
            return_spec=None,
            is_method=True,
            class_name="MyModel",
        )
        assert func.is_method
        assert func.class_name == "MyModel"


class TestDiagnostic:
    def test_error(self) -> None:
        diag = Diagnostic(
            file="model.py",
            line=10,
            col=0,
            severity="error",
            message="Shape mismatch",
            rule="shape-mismatch",
        )
        assert diag.severity == "error"
        assert diag.rule == "shape-mismatch"

    def test_info(self) -> None:
        diag = Diagnostic(
            file="model.py",
            line=1,
            col=0,
            severity="info",
            message="Could not import",
            rule="import-error",
        )
        assert diag.severity == "info"


class TestTraceResult:
    def test_success(self) -> None:
        result = TraceResult(
            function_name="attention",
            output_shape=(2, 3, 5, 5),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        assert result.success
        assert result.output_shape == (2, 3, 5, 5)

    def test_failure(self) -> None:
        result = TraceResult(
            function_name="broken",
            output_shape=None,
            output_dtype=None,
            intermediates=[],
            error="Shape error in matmul",
        )
        assert not result.success
        assert result.error is not None


class TestIntermediateShape:
    def test_basic(self) -> None:
        inter = IntermediateShape(
            shape=(2, 3, 5),
            dtype="float32",
            source_file="model.py",
            source_line=15,
            source_col=4,
            named_shape=("batch", "seq", "d_model"),
            op_name="dot_general",
        )
        assert inter.shape == (2, 3, 5)
        assert inter.named_shape == ("batch", "seq", "d_model")
        assert inter.op_name == "dot_general"


class TestShardingInfo:
    def test_sharding_info_frozen(self) -> None:
        info = ShardingInfo(
            partition_spec=("data", None, None),
            mesh_axis_names=("data", "model"),
            source_primitive="sharding_constraint",
        )
        assert info.partition_spec == ("data", None, None)
        assert info.source_line == 0  # default

    def test_sharding_info_with_line(self) -> None:
        info = ShardingInfo(
            partition_spec=("data",),
            mesh_axis_names=("data",),
            source_primitive="jit",
            source_line=42,
        )
        assert info.source_line == 42

    def test_sharding_info_immutable(self) -> None:
        import pytest

        info = ShardingInfo(
            partition_spec=("data",),
            mesh_axis_names=("data",),
            source_primitive="jit",
        )
        with pytest.raises(AttributeError):
            info.source_primitive = "other"  # type: ignore[misc]


class TestErrorHintInfo:
    def test_error_hint_info_frozen(self) -> None:
        info = ErrorHintInfo(
            source_line=10,
            message="expected dim 1: seq, got head_dim",
            rule="shape-mismatch",
            function_name="forward",
        )
        assert info.source_line == 10
        assert info.expected_named is None  # default
        assert info.actual_named is None  # default

    def test_error_hint_info_with_shapes(self) -> None:
        info = ErrorHintInfo(
            source_line=5,
            message="rank mismatch",
            rule="rank-mismatch",
            function_name="encode",
            expected_named=("batch", "seq"),
            actual_named=("batch",),
        )
        assert info.expected_named == ("batch", "seq")
        assert info.actual_named == ("batch",)


class TestIntermediateShapeSharding:
    def test_intermediate_shape_with_sharding(self) -> None:
        inter = IntermediateShape(
            shape=(4, 8),
            dtype="float32",
            source_file="f.py",
            source_line=5,
            source_col=0,
            named_shape=("batch", "seq"),
            op_name="add",
            sharding=ShardingInfo(("data", None), ("data",), "sharding_constraint"),
        )
        assert inter.sharding is not None
        assert inter.sharding.partition_spec == ("data", None)

    def test_intermediate_shape_sharding_default_none(self) -> None:
        inter = IntermediateShape(
            shape=(4,),
            dtype="float32",
            source_file="f.py",
            source_line=1,
            source_col=0,
            named_shape=("batch",),
            op_name="add",
        )
        assert inter.sharding is None


class TestFileResult:
    def test_basic(self) -> None:
        result = FileResult(
            file_path="model.py",
            functions_checked=3,
            diagnostics=[],
            trace_results=[],
        )
        assert result.functions_checked == 3
        assert len(result.diagnostics) == 0

    def test_with_diagnostics(self) -> None:
        diag = Diagnostic(
            file="model.py",
            line=10,
            col=0,
            severity="error",
            message="mismatch",
            rule="shape-mismatch",
        )
        result = FileResult(
            file_path="model.py",
            functions_checked=1,
            diagnostics=[diag],
            trace_results=[],
        )
        assert len(result.diagnostics) == 1
