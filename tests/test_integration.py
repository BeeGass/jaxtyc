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

    def test_pipeline_runs_sharding_checks(self) -> None:
        """analyze_file runs sharding checker and returns sharding diagnostics."""
        result = analyze_file(str(FIXTURES / "sharded_rank_mismatch.py"))
        assert result.functions_checked >= 1
        sharding_diags = [d for d in result.diagnostics if d.rule == "sharding-rank-mismatch"]
        assert len(sharding_diags) >= 1, (
            f"Expected sharding-rank-mismatch diagnostic, got rules: "
            f"{[d.rule for d in result.diagnostics]}"
        )


class TestShardedPipelineIntegration:
    """End-to-end tests for sharding diagnostics wired through the pipeline."""

    def test_correct_sharded_matmul_no_errors(self) -> None:
        """Correct sharding annotations produce no diagnostics at all."""
        result = analyze_file(str(FIXTURES / "sharded_full_correct.py"))
        assert result.functions_checked >= 1
        assert len(result.diagnostics) == 0, (
            f"Unexpected diagnostics: {[(d.rule, d.message) for d in result.diagnostics]}"
        )

    def test_propagation_mismatch_is_only_sharding_error(self) -> None:
        """Return annotation claiming unearned sharding triggers exactly one diagnostic."""
        result = analyze_file(str(FIXTURES / "sharded_propagation_mismatch.py"))
        assert result.functions_checked >= 1
        rules = [d.rule for d in result.diagnostics]
        assert "sharding-propagation-mismatch" in rules, (
            f"Expected sharding-propagation-mismatch, got: {rules}"
        )
        # No spurious trace-error or other sharding diagnostics
        assert "trace-error" not in rules, (
            f"Unexpected trace-error alongside propagation check: {rules}"
        )

    def test_annotation_incomplete_no_trace_error(self) -> None:
        """Incomplete annotation produces annotation-incomplete, not trace-error."""
        result = analyze_file(str(FIXTURES / "sharded_annotation_incomplete.py"))
        assert result.functions_checked >= 1
        rules = [d.rule for d in result.diagnostics]
        assert "sharding-annotation-incomplete" in rules, (
            f"Expected sharding-annotation-incomplete, got: {rules}"
        )
        assert "trace-error" not in rules, f"Unexpected trace-error: {rules}"

    def test_dim_conflict_no_trace_error(self) -> None:
        """Conflicting dim axes produce dim-conflict, not a spurious trace-error."""
        result = analyze_file(str(FIXTURES / "sharded_dim_conflict.py"))
        assert result.functions_checked >= 1
        rules = [d.rule for d in result.diagnostics]
        assert "sharding-dim-conflict" in rules, f"Expected sharding-dim-conflict, got: {rules}"
        # Annotation error causes fallback to unsharded tracing, so no trace-error
        assert "trace-error" not in rules, (
            f"Spurious trace-error from concrete size collision: {rules}"
        )

    def test_rank_mismatch_from_make_jaxpr(self) -> None:
        """PartitionSpec rank != array rank produces sharding-rank-mismatch."""
        result = analyze_file(str(FIXTURES / "sharded_rank_mismatch.py"))
        assert result.functions_checked >= 1
        rules = [d.rule for d in result.diagnostics]
        assert "sharding-rank-mismatch" in rules, f"Expected sharding-rank-mismatch, got: {rules}"


class TestShardedMatmul:
    def test_sharded_matmul_correct_no_errors(self) -> None:
        """Standard matmul fixture with mesh passes with zero diagnostics."""
        result = analyze_file(str(FIXTURES / "sharded_matmul.py"))
        assert result.functions_checked >= 1
        assert len(result.diagnostics) == 0, (
            f"Unexpected diagnostics: {[(d.rule, d.message) for d in result.diagnostics]}"
        )

    def test_sharded_matmul_piped_no_errors(self) -> None:
        """Piped matmul fixture parses pipe syntax and produces no errors."""
        result = analyze_file(str(FIXTURES / "sharded_matmul_piped.py"))
        assert result.functions_checked >= 1
        assert len(result.diagnostics) == 0, (
            f"Unexpected diagnostics: {[(d.rule, d.message) for d in result.diagnostics]}"
        )
