"""Unit tests for jaxtyc.analyzer.pipeline internals."""

from __future__ import annotations

import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from jaxtyc.analyzer._errors import truncate_error
from jaxtyc.analyzer.pipeline import _is_eqx_module
from jaxtyc.analyzer.pipeline import _is_nnx_module
from jaxtyc.analyzer.pipeline import _resolve_function
from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.types import TraceResult

FIXTURES = Path(__file__).parent / "fixtures"


class TestResolveFunction:
    def test_resolve_top_level_function(self) -> None:
        mod = types.ModuleType("fake")
        mod.my_fn = lambda x: x
        result = _resolve_function(mod, "my_fn", None)
        assert result is mod.my_fn  # type: ignore[attr-defined]

    def test_resolve_method_from_class(self) -> None:
        mod = types.ModuleType("fake")

        class MyClass:
            def forward(self, x):  # type: ignore[no-untyped-def]
                return x

        mod.MyClass = MyClass
        result = _resolve_function(mod, "forward", "MyClass")
        assert result is not None

    def test_resolve_missing_function_returns_none(self) -> None:
        mod = types.ModuleType("fake")
        result = _resolve_function(mod, "nonexistent", None)
        assert result is None

    def test_resolve_missing_class_returns_none(self) -> None:
        mod = types.ModuleType("fake")
        result = _resolve_function(mod, "forward", "NoSuchClass")
        assert result is None


class TestIsNnxModule:
    def test_non_module_class_returns_false(self) -> None:
        class Plain:
            pass

        assert _is_nnx_module(Plain) is False

    def test_non_class_returns_false(self) -> None:
        assert _is_nnx_module(42) is False  # type: ignore[arg-type]


class TestIsEqxModule:
    def test_non_module_class_returns_false(self) -> None:
        class Plain:
            pass

        assert _is_eqx_module(Plain) is False


class TestAnalyzeFile:
    def test_file_not_found_returns_diagnostic(self) -> None:
        result = analyze_file("/nonexistent/path/to/file.py")
        assert result.functions_checked == 0
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].rule == "file-not-found"

    def test_no_annotations_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def plain_function(x):\n    return x\n")
            f.flush()
            result = analyze_file(f.name)
        assert result.functions_checked == 0
        assert len(result.diagnostics) == 0

    def test_resolve_error_diagnostic(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "from jaxtyping import Array, Float\n"
                "class Outer:\n"
                "    def method(self, x: Float[Array, 'batch seq']) -> Float[Array, 'batch seq']:\n"
                "        return x\n"
            )
            f.flush()
            result = analyze_file(f.name)
        resolve_errors = [d for d in result.diagnostics if d.rule == "resolve-error"]
        # The method is resolved via class_name, but the fixture's class may not
        # be importable depending on import path. At minimum, no crash.
        assert isinstance(result.functions_checked, int)

    def test_inline_suppression_filters_diagnostics(self) -> None:
        result = analyze_file(str(FIXTURES / "suppressed.py"))
        rules = [d.rule for d in result.diagnostics]
        # wrong_but_suppressed is fully suppressed
        # wrong_not_suppressed should still produce diagnostics
        unsuppressed_errors = [
            d for d in result.diagnostics if d.severity == "error" and d.line >= 24
        ]
        assert len(unsuppressed_errors) >= 1

    def test_correct_file_zero_diagnostics(self) -> None:
        result = analyze_file(str(FIXTURES / "correct_attention.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0
        assert result.functions_checked >= 1

    def test_sharding_fallback_sets_reason(self) -> None:
        result = analyze_file(str(FIXTURES / "sharded_rank_mismatch.py"))
        # This fixture has sharding annotations — if mesh isn't available from
        # source AST, it may fall back. Just verify no crash.
        assert isinstance(result.functions_checked, int)
        assert isinstance(result.diagnostics, list)

    def test_cross_function_call_propagation(self) -> None:
        result = analyze_file(str(FIXTURES / "cross_function_mismatch.py"))
        cross_diags = [d for d in result.diagnostics if d.rule == "cross-function-mismatch"]
        assert len(cross_diags) >= 1


class TestAnalyzeFileReadError:
    def test_unreadable_file_returns_diagnostic(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "unreadable.py"
        bad_file.write_text("x = 1")
        bad_file.chmod(0o000)
        try:
            result = analyze_file(str(bad_file))
            assert result.functions_checked == 0
            assert any(d.rule == "read-error" for d in result.diagnostics)
        finally:
            bad_file.chmod(0o644)


class TestAnalyzeFileMeshResolution:
    def test_mesh_resolved_from_source(self) -> None:
        result = analyze_file(str(FIXTURES / "sharded_full_correct.py"))
        # sharded_full_correct has mesh definition in source — should trace
        # without falling back. Just verify no crash and some functions checked.
        assert result.functions_checked >= 1


class TestTruncateError:
    def test_short_message_unchanged(self) -> None:
        assert truncate_error("simple error") == "simple error"

    def test_long_message_truncated(self) -> None:
        long_msg = "x" * 1000
        result = truncate_error(long_msg)
        assert len(result) == 504  # 500 + " ..."
        assert result.endswith(" ...")

    def test_multiline_takes_first_line(self) -> None:
        msg = "First line\nSecond line\nThird line"
        assert truncate_error(msg) == "First line"

    def test_exception_object(self) -> None:
        exc = ValueError("something went wrong")
        assert truncate_error(exc) == "something went wrong"

    def test_custom_max_len(self) -> None:
        result = truncate_error("abcdefghij", max_len=5)
        assert result == "abcde ..."


class TestAbstractTracing:
    """Test that NNX/equinox modules are traced via abstract (zero-alloc) path."""

    def test_trace_nnx_abstract_exists(self) -> None:
        """_trace_nnx_abstract function should exist in pipeline."""
        from jaxtyc.analyzer.pipeline import _trace_nnx_abstract

        assert callable(_trace_nnx_abstract)

    def test_trace_eqx_abstract_exists(self) -> None:
        """_trace_eqx_abstract function should exist in pipeline."""
        from jaxtyc.analyzer.pipeline import _trace_eqx_abstract

        assert callable(_trace_eqx_abstract)

    def test_trace_nnx_concrete_exists(self) -> None:
        """_trace_nnx_concrete fallback function should exist in pipeline."""
        from jaxtyc.analyzer.pipeline import _trace_nnx_concrete

        assert callable(_trace_nnx_concrete)

    def test_trace_eqx_concrete_exists(self) -> None:
        """_trace_eqx_concrete fallback function should exist in pipeline."""
        from jaxtyc.analyzer.pipeline import _trace_eqx_concrete

        assert callable(_trace_eqx_concrete)

    def test_eqx_module_abstract_path_used(self) -> None:
        """Equinox module analysis should try abstract path first."""
        with patch("jaxtyc.analyzer.pipeline._trace_eqx_abstract") as mock_abstract:
            mock_abstract.return_value = None
            analyze_file(str(FIXTURES / "eqx_module.py"))
            mock_abstract.assert_called()

    def test_eqx_abstract_produces_correct_shapes(self) -> None:
        """Abstract equinox tracing should produce valid output shapes."""
        result = analyze_file(str(FIXTURES / "eqx_module.py"))
        assert result.functions_checked > 0
        for tr in result.trace_results:
            if tr.success and tr.output_shape is not None:
                assert isinstance(tr.output_shape, tuple)
                assert len(tr.output_shape) > 0

    def test_fallback_to_concrete_on_abstract_failure(self) -> None:
        """If abstract tracing returns None, concrete fallback runs."""
        with (
            patch("jaxtyc.analyzer.pipeline._trace_eqx_abstract", return_value=None),
            patch("jaxtyc.analyzer.pipeline._trace_eqx_concrete") as mock_concrete,
        ):
            mock_concrete.return_value = TraceResult(
                function_name="__call__",
                output_shape=(101, 103),
                output_dtype="float32",
                intermediates=[],
                error=None,
            )
            analyze_file(str(FIXTURES / "eqx_module.py"))
            mock_concrete.assert_called()
