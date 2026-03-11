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


class TestNNXIntermediateExtraction:
    def test_nnx_trace_has_intermediates(self) -> None:
        """SimpleMLP trace should produce intermediates from make_jaxpr."""
        result = analyze_file(str(FIXTURES / "nnx_module.py"))
        simple_traces = [
            t
            for t in result.trace_results
            if t.function_name == "__call__" and t.success and t.output_shape is not None
        ]
        assert len(simple_traces) >= 1
        has_intermediates = any(len(t.intermediates) > 0 for t in simple_traces)
        assert has_intermediates, (
            "NNX module traces should have intermediates from make_jaxpr, "
            f"got: {[(t.function_name, len(t.intermediates)) for t in simple_traces]}"
        )

    def test_nnx_intermediates_have_shapes(self) -> None:
        """NNX intermediates should have valid shape tuples."""
        result = analyze_file(str(FIXTURES / "nnx_module.py"))
        traces_with_inters = [
            t for t in result.trace_results if t.success and len(t.intermediates) > 0
        ]
        assert len(traces_with_inters) >= 1
        for trace in traces_with_inters:
            for inter in trace.intermediates:
                assert isinstance(inter.shape, tuple)
                assert len(inter.shape) > 0
                assert inter.dtype != ""
                assert inter.op_name != ""


class TestEquinoxIntermediateExtraction:
    def test_eqx_trace_has_intermediates(self) -> None:
        """SimpleLinear trace should produce intermediates from make_jaxpr."""
        result = analyze_file(str(FIXTURES / "eqx_module.py"))
        simple_traces = [
            t
            for t in result.trace_results
            if t.function_name == "__call__" and t.success and t.output_shape is not None
        ]
        assert len(simple_traces) >= 1
        has_intermediates = any(len(t.intermediates) > 0 for t in simple_traces)
        assert has_intermediates, (
            "Equinox module traces should have intermediates from make_jaxpr, "
            f"got: {[(t.function_name, len(t.intermediates)) for t in simple_traces]}"
        )

    def test_eqx_intermediates_have_shapes(self) -> None:
        """Equinox intermediates should have valid shape tuples."""
        result = analyze_file(str(FIXTURES / "eqx_module.py"))
        traces_with_inters = [
            t for t in result.trace_results if t.success and len(t.intermediates) > 0
        ]
        assert len(traces_with_inters) >= 1
        for trace in traces_with_inters:
            for inter in trace.intermediates:
                assert isinstance(inter.shape, tuple)
                assert len(inter.shape) > 0


class TestNNXShardingDetection:
    def test_nnx_sharded_intermediates_have_sharding_info(self) -> None:
        """ShardedMLP intermediates should include ShardingInfo."""
        result = analyze_file(str(FIXTURES / "nnx_sharded.py"))
        assert result.functions_checked >= 1
        traces_with_inters = [
            t for t in result.trace_results if t.success and len(t.intermediates) > 0
        ]
        assert len(traces_with_inters) >= 1
        sharded_inters = [
            inter
            for trace in traces_with_inters
            for inter in trace.intermediates
            if inter.sharding is not None
        ]
        assert len(sharded_inters) >= 1, "Expected at least one intermediate with sharding info"

    def test_nnx_sharded_produces_sharding_diagnostic(self) -> None:
        """ShardedMLP has P('data') on rank-2 array — should flag rank mismatch."""
        result = analyze_file(str(FIXTURES / "nnx_sharded.py"))
        sharding_diags = [d for d in result.diagnostics if d.rule == "sharding-rank-mismatch"]
        assert len(sharding_diags) >= 1, (
            f"Expected sharding-rank-mismatch, got rules: {[d.rule for d in result.diagnostics]}"
        )


class TestNNXMultiHeadConstruction:
    def test_multihead_with_divisibility_assertion(self) -> None:
        """NNX module with assert features % num_head == 0 should trace successfully.

        Optional int params with defaults (like num_head=1) should keep their
        defaults rather than getting overridden with primes.
        """
        result = analyze_file(str(FIXTURES / "nnx_multihead.py"))
        assert result.functions_checked >= 1
        trace_errors = [d for d in result.diagnostics if d.rule == "trace-error"]
        assert len(trace_errors) == 0, (
            f"Expected no trace errors for MultiHeadLayer, got: {[d.message for d in trace_errors]}"
        )


class TestNNXDivergenceDetection:
    def test_buggy_nnx_has_intermediates_for_divergence(self) -> None:
        """BuggyMLP trace should have intermediates enabling divergence detection."""
        result = analyze_file(str(FIXTURES / "nnx_module.py"))
        errors = [d for d in result.diagnostics if d.severity == "error"]
        assert len(errors) >= 1
        traces_with_inters = [
            t for t in result.trace_results if t.success and len(t.intermediates) > 0
        ]
        assert len(traces_with_inters) >= 1, (
            "Expected NNX traces to have intermediates for divergence detection"
        )
