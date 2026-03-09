"""Tests for jaxtyc.types — core data types."""

from __future__ import annotations

from jaxtyc.types import Diagnostic
from jaxtyc.types import DimSpec
from jaxtyc.types import FileResult
from jaxtyc.types import FunctionShapeSpec
from jaxtyc.types import IntermediateShape
from jaxtyc.types import ShapeSpec
from jaxtyc.types import TraceResult


class TestDimSpec:
    def test_named_dim(self):
        d = DimSpec(kind="named", name="batch")
        assert d.kind == "named"
        assert d.name == "batch"
        assert d.size is None

    def test_fixed_dim(self):
        d = DimSpec(kind="fixed", size=4)
        assert d.kind == "fixed"
        assert d.size == 4
        assert d.name is None

    def test_variadic_dim(self):
        d = DimSpec(kind="variadic", name="batch")
        assert d.kind == "variadic"
        assert d.name == "batch"

    def test_anonymous_dim(self):
        d = DimSpec(kind="anonymous")
        assert d.kind == "anonymous"

    def test_ellipsis_dim(self):
        d = DimSpec(kind="ellipsis")
        assert d.kind == "ellipsis"

    def test_frozen(self):
        import pytest

        d = DimSpec(kind="named", name="seq")
        with pytest.raises(AttributeError):
            d.name = "other"  # type: ignore[misc]


class TestShapeSpec:
    def test_basic_shape(self):
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

    def test_scalar(self):
        spec = ShapeSpec(dims=(), dtype="float32", is_scalar=True)
        assert spec.is_scalar
        assert len(spec.dims) == 0

    def test_any_shape(self):
        spec = ShapeSpec(dims=(), dtype="float32", is_any_shape=True)
        assert spec.is_any_shape

    def test_frozen(self):
        import pytest

        spec = ShapeSpec(dims=(), dtype="float32")
        with pytest.raises(AttributeError):
            spec.dtype = "int32"  # type: ignore[misc]


class TestFunctionShapeSpec:
    def test_basic(self):
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

    def test_method(self):
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
    def test_error(self):
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

    def test_info(self):
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
    def test_success(self):
        result = TraceResult(
            function_name="attention",
            output_shape=(2, 3, 5, 5),
            output_dtype="float32",
            intermediates=[],
            error=None,
        )
        assert result.success
        assert result.output_shape == (2, 3, 5, 5)

    def test_failure(self):
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
    def test_basic(self):
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


class TestFileResult:
    def test_basic(self):
        result = FileResult(
            file_path="model.py",
            functions_checked=3,
            diagnostics=[],
            trace_results=[],
        )
        assert result.functions_checked == 3
        assert len(result.diagnostics) == 0

    def test_with_diagnostics(self):
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
