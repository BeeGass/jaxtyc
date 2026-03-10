"""Tests for NNX and equinox module tracing."""

from __future__ import annotations

from pathlib import Path

from jaxtyc.analyzer.pipeline import analyze_file

FIXTURES = Path(__file__).parent / "fixtures"


class TestNNXModuleTracing:
    def test_correct_nnx_module_no_errors(self) -> None:
        """SimpleMLP.__call__ should produce zero shape errors."""
        result = analyze_file(str(FIXTURES / "nnx_module.py"))
        # Should trace at least the SimpleMLP.__call__
        assert result.functions_checked >= 1
        errors = [d for d in result.diagnostics if d.severity == "error"]
        # SimpleMLP is correct, BuggyMLP has a mismatch
        # Find diagnostics for SimpleMLP specifically
        simple_errors = [d for d in errors if "SimpleMLP" in d.message or "SimpleMLP" in str(d)]
        assert len(simple_errors) == 0

    def test_buggy_nnx_module_shape_mismatch(self) -> None:
        """BuggyMLP.__call__ annotates d_in as return but linear produces d_out."""
        result = analyze_file(str(FIXTURES / "nnx_module.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        # BuggyMLP should have at least one shape error
        assert len(errors) >= 1


class TestEquinoxModuleTracing:
    def test_correct_eqx_module_no_errors(self) -> None:
        """SimpleLinear.__call__ should produce zero shape errors."""
        result = analyze_file(str(FIXTURES / "eqx_module.py"))
        assert result.functions_checked >= 1
        # Find errors specific to SimpleLinear
        errors = [d for d in result.diagnostics if d.severity == "error"]
        simple_errors = [
            d for d in errors if "SimpleLinear" in d.message or "SimpleLinear" in str(d)
        ]
        assert len(simple_errors) == 0

    def test_buggy_eqx_module_shape_mismatch(self) -> None:
        """BuggyLinear.__call__ annotates d_in as return but matmul produces d_out."""
        result = analyze_file(str(FIXTURES / "eqx_module.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
