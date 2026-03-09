"""Tests for jaxtyc public API exports."""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class TestPublicAPIImports:
    def test_analyze_file_importable_from_jaxtyc(self):
        """analyze_file should be importable from the top-level jaxtyc package."""
        from jaxtyc import analyze_file

        assert callable(analyze_file)

    def test_version_importable(self):
        """__version__ should be importable from jaxtyc."""
        from jaxtyc import __version__

        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_types_importable(self):
        """Core types should be importable from jaxtyc."""
        from jaxtyc import Diagnostic
        from jaxtyc import FileResult
        from jaxtyc import TraceResult

        assert Diagnostic is not None
        assert FileResult is not None
        assert TraceResult is not None


class TestPublicAPIFunctionality:
    def test_analyze_file_returns_file_result(self):
        """analyze_file should return a FileResult."""
        from jaxtyc import FileResult
        from jaxtyc import analyze_file

        result = analyze_file(str(FIXTURES / "correct_attention.py"))
        assert isinstance(result, FileResult)

    def test_analyze_file_correct_has_no_errors(self):
        """analyze_file on correct code should produce zero errors."""
        from jaxtyc import analyze_file

        result = analyze_file(str(FIXTURES / "correct_attention.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) == 0

    def test_analyze_file_wrong_has_errors(self):
        """analyze_file on buggy code should produce errors."""
        from jaxtyc import analyze_file

        result = analyze_file(str(FIXTURES / "wrong_transpose.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
