"""Tests for inline suppression comment parsing and filtering."""

from __future__ import annotations

from pathlib import Path

from jaxtyc.analyzer.suppressions import extract_suppressions
from jaxtyc.analyzer.suppressions import filter_inline_suppressions
from jaxtyc.types import Diagnostic
from jaxtyc.types import SuppressionComment

FIXTURES = Path(__file__).parent / "fixtures"


class TestExtractSuppressions:
    def test_no_suppressions(self) -> None:
        source = "x = 1\ny = 2\n"
        assert extract_suppressions(source) == []

    def test_ignore_all(self) -> None:
        source = "x = 1  # jaxtyc: ignore\n"
        result = extract_suppressions(source)
        assert len(result) == 1
        assert result[0].line == 1
        assert result[0].rules == frozenset()

    def test_ignore_specific_rule(self) -> None:
        source = "x = 1  # jaxtyc: ignore[shape-mismatch]\n"
        result = extract_suppressions(source)
        assert len(result) == 1
        assert result[0].rules == frozenset({"shape-mismatch"})

    def test_ignore_multiple_rules(self) -> None:
        source = "x = 1  # jaxtyc: ignore[shape-mismatch, rank-mismatch]\n"
        result = extract_suppressions(source)
        assert len(result) == 1
        assert result[0].rules == frozenset({"shape-mismatch", "rank-mismatch"})

    def test_extra_whitespace(self) -> None:
        source = "x = 1  #  jaxtyc:  ignore\n"
        result = extract_suppressions(source)
        assert len(result) == 1

    def test_multiple_lines(self) -> None:
        source = "a = 1  # jaxtyc: ignore\nb = 2\nc = 3  # jaxtyc: ignore[foo]\n"
        result = extract_suppressions(source)
        assert len(result) == 2
        assert result[0].line == 1
        assert result[1].line == 3


class TestFilterInlineSuppressions:
    def _make_diag(self, line: int, rule: str) -> Diagnostic:
        return Diagnostic(
            file="test.py", line=line, col=0, severity="error", message="test", rule=rule
        )

    def test_no_suppressions_passes_all(self) -> None:
        diags = [self._make_diag(1, "shape-mismatch")]
        assert filter_inline_suppressions(diags, []) == diags

    def test_ignore_all_suppresses(self) -> None:
        diags = [self._make_diag(5, "shape-mismatch")]
        supps = [SuppressionComment(line=5, rules=frozenset())]
        assert filter_inline_suppressions(diags, supps) == []

    def test_ignore_specific_rule_suppresses(self) -> None:
        diags = [self._make_diag(5, "shape-mismatch")]
        supps = [SuppressionComment(line=5, rules=frozenset({"shape-mismatch"}))]
        assert filter_inline_suppressions(diags, supps) == []

    def test_wrong_rule_not_suppressed(self) -> None:
        diags = [self._make_diag(5, "shape-mismatch")]
        supps = [SuppressionComment(line=5, rules=frozenset({"rank-mismatch"}))]
        assert filter_inline_suppressions(diags, supps) == diags

    def test_previous_line_suppression(self) -> None:
        diags = [self._make_diag(6, "shape-mismatch")]
        supps = [SuppressionComment(line=5, rules=frozenset())]
        assert filter_inline_suppressions(diags, supps) == []

    def test_mixed_suppressed_and_not(self) -> None:
        diags = [
            self._make_diag(5, "shape-mismatch"),
            self._make_diag(10, "rank-mismatch"),
        ]
        supps = [SuppressionComment(line=5, rules=frozenset())]
        result = filter_inline_suppressions(diags, supps)
        assert len(result) == 1
        assert result[0].line == 10


class TestSuppressionIntegration:
    def test_suppressed_fixture(self) -> None:
        from jaxtyc import analyze_file

        result = analyze_file(str(FIXTURES / "suppressed.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        # wrong_but_suppressed and wrong_specific_suppress should be suppressed
        # wrong_not_suppressed should still produce an error
        assert len(errors) == 1
        assert "wrong_not_suppressed" in errors[0].message
