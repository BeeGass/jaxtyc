"""End-to-end integration tests using test fixtures."""

from __future__ import annotations

from pathlib import Path

from jaxtyc.analyzer.pipeline import analyze_file

FIXTURES = Path(__file__).parent / "fixtures"


class TestAnalyzeFileEndToEnd:
    def test_correct_attention_no_errors(self) -> None:
        result = analyze_file(str(FIXTURES / "correct_attention.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_wrong_transpose_caught(self) -> None:
        result = analyze_file(str(FIXTURES / "wrong_transpose.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
        # Should be a shape mismatch on the return
        assert any(d.rule == "shape-mismatch" for d in errors)

    def test_wrong_rank_caught(self) -> None:
        result = analyze_file(str(FIXTURES / "wrong_rank.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
        assert any(d.rule == "rank-mismatch" for d in errors)

    def test_wrong_inner_dim_caught(self) -> None:
        result = analyze_file(str(FIXTURES / "wrong_inner_dim.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
        assert any(d.rule == "shape-mismatch" for d in errors)

    def test_ellipsis_patterns_no_errors(self) -> None:
        result = analyze_file(str(FIXTURES / "ellipsis_patterns.py"))
        # flexible_input uses "..." which is any_shape — should be skipped
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0

    def test_untraceable_graceful(self) -> None:
        result = analyze_file(str(FIXTURES / "untraceable.py"))
        # No jaxtyping annotations, so no functions checked
        assert result.functions_checked == 0
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0

    def test_nonexistent_file(self) -> None:
        result = analyze_file("/nonexistent/path.py")
        assert result.functions_checked == 0
        assert any(d.severity == "info" for d in result.diagnostics)

    def test_tuple_return_correct(self) -> None:
        result = analyze_file(str(FIXTURES / "tuple_return.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_tuple_return_mismatch(self) -> None:
        result = analyze_file(str(FIXTURES / "tuple_return_mismatch.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
        assert any(d.rule == "shape-mismatch" for d in errors)

    def test_cross_function_mismatch(self) -> None:
        result = analyze_file(str(FIXTURES / "cross_function_mismatch.py"))
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert any(d.rule == "cross-function-mismatch" for d in errors)
