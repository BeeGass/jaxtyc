"""Self-analysis: run jaxtyc on its own test fixtures as a dogfooding test."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxtyc.analyzer.pipeline import analyze_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSelfAnalysis:
    """jaxtyc should correctly analyze its own test fixtures."""

    @pytest.mark.parametrize(
        "fixture",
        [
            "correct_attention.py",
            "ellipsis_patterns.py",
            "int_annotations.py",
            "bool_annotations.py",
            "complex_annotations.py",
            "shaped_annotations.py",
        ],
    )
    def test_correct_fixtures_produce_no_errors(self, fixture: str) -> None:
        """Fixtures with correct annotations should have zero error diagnostics."""
        result = analyze_file(str(FIXTURES_DIR / fixture))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert errors == [], f"Unexpected errors in {fixture}: {errors}"

    def test_multi_function_traces(self) -> None:
        """multi_function.py has hardcoded weight sizes that conflict with symbolic
        tracing, so it produces trace-errors — verify it at least runs without crash."""
        result = analyze_file(str(FIXTURES_DIR / "multi_function.py"))
        assert result.functions_checked > 0

    @pytest.mark.parametrize(
        "fixture",
        [
            "wrong_transpose.py",
            "wrong_rank.py",
            "wrong_inner_dim.py",
        ],
    )
    def test_buggy_fixtures_produce_errors(self, fixture: str) -> None:
        """Fixtures with intentional bugs should produce error diagnostics."""
        result = analyze_file(str(FIXTURES_DIR / fixture))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) > 0, f"Expected errors in {fixture} but found none"

    def test_key_annotations_no_crash(self) -> None:
        """PRNGKeyArray annotations should be handled without crashing."""
        result = analyze_file(str(FIXTURES_DIR / "key_annotations.py"))
        assert result is not None
