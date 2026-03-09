"""Performance benchmark tests — enforce timing targets from the plan.

These are regression guards. If a change blows the performance budget,
the test fails. Targets are generous (5-10x headroom) to avoid flaky
failures on CI, but still catch egregious regressions.
"""

from __future__ import annotations

import time
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class TestAnnotationParserPerformance:
    def test_parse_single_file_under_5ms(self):
        """AST parse a single file should take < 5ms."""
        from jaxtyc.analyzer.annotations import extract_function_specs

        source = (FIXTURES / "correct_attention.py").read_text()
        # Warm up
        extract_function_specs(source, "bench.py")

        _, elapsed = _timed(extract_function_specs, source, "bench.py")
        assert elapsed < 0.005, f"AST parse took {elapsed * 1000:.1f}ms, expected < 5ms"

    def test_parse_all_fixtures_under_10ms(self):
        """Parsing all 6 fixture files should take < 10ms total."""
        from jaxtyc.analyzer.annotations import extract_function_specs

        fixtures = list(FIXTURES.glob("*.py"))
        assert len(fixtures) >= 6

        # Warm up
        for f in fixtures:
            extract_function_specs(f.read_text(), str(f))

        start = time.perf_counter()
        for f in fixtures:
            extract_function_specs(f.read_text(), str(f))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.010, (
            f"Parsing {len(fixtures)} files took {elapsed * 1000:.1f}ms, expected < 10ms"
        )


class TestDimEnvPerformance:
    def test_allocate_50_dims_under_1ms(self):
        """Allocating 50 dimension names should take < 1ms."""
        from jaxtyc.analyzer.dim_env import DimEnv

        # Warm up
        env = DimEnv()
        for i in range(50):
            env.get_size(f"dim_{i}")

        env = DimEnv()
        start = time.perf_counter()
        for i in range(50):
            env.get_size(f"dim_{i}")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, f"50 dim allocations took {elapsed * 1000:.2f}ms, expected < 1ms"


class TestTracerPerformance:
    def test_eval_shape_per_function_under_5ms(self):
        """trace_function (eval_shape) should take < 5ms per function."""
        import jax.numpy as jnp

        from jaxtyc.analyzer.dim_env import DimEnv
        from jaxtyc.analyzer.tracer import trace_function
        from jaxtyc.types import DimSpec
        from jaxtyc.types import ShapeSpec

        def attention(q, k):
            return jnp.matmul(q, jnp.swapaxes(k, -1, -2))

        params = {
            "q": ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch"),
                    DimSpec(kind="named", name="heads"),
                    DimSpec(kind="named", name="seq"),
                    DimSpec(kind="named", name="head_dim"),
                ),
                dtype="float32",
            ),
            "k": ShapeSpec(
                dims=(
                    DimSpec(kind="named", name="batch"),
                    DimSpec(kind="named", name="heads"),
                    DimSpec(kind="named", name="seq"),
                    DimSpec(kind="named", name="head_dim"),
                ),
                dtype="float32",
            ),
        }

        # Warm up (first call may JIT compile)
        env = DimEnv()
        trace_function(attention, params, env)

        # Benchmark
        env = DimEnv()
        _, elapsed = _timed(trace_function, attention, params, env)
        assert elapsed < 0.005, f"eval_shape took {elapsed * 1000:.2f}ms, expected < 5ms"

    def test_make_jaxpr_per_function_under_10ms(self):
        """extract_source_mapped_intermediates (make_jaxpr) should take < 10ms."""
        import jax.numpy as jnp

        from jaxtyc.analyzer.dim_env import DimEnv
        from jaxtyc.analyzer.source_map import extract_source_mapped_intermediates
        from jaxtyc.analyzer.tracer import _build_abstract_input
        from jaxtyc.types import DimSpec
        from jaxtyc.types import ShapeSpec

        def attention(q, k):
            return jnp.matmul(q, jnp.swapaxes(k, -1, -2))

        spec = ShapeSpec(
            dims=(
                DimSpec(kind="named", name="batch"),
                DimSpec(kind="named", name="heads"),
                DimSpec(kind="named", name="seq"),
                DimSpec(kind="named", name="head_dim"),
            ),
            dtype="float32",
        )

        env = DimEnv()
        abstract_inputs = {
            "q": _build_abstract_input(spec, env),
            "k": _build_abstract_input(spec, env),
        }

        # Warm up
        extract_source_mapped_intermediates(attention, abstract_inputs, env)

        # Benchmark
        env2 = DimEnv()
        abstract_inputs2 = {
            "q": _build_abstract_input(spec, env2),
            "k": _build_abstract_input(spec, env2),
        }
        _, elapsed = _timed(extract_source_mapped_intermediates, attention, abstract_inputs2, env2)
        assert elapsed < 0.010, f"make_jaxpr took {elapsed * 1000:.2f}ms, expected < 10ms"


class TestPipelinePerformance:
    def test_analyze_single_file_under_50ms(self):
        """Full analyze_file on a single-function file should take < 50ms."""
        from jaxtyc.analyzer.pipeline import analyze_file

        path = str(FIXTURES / "correct_attention.py")

        # Warm up (first call imports JAX + module)
        analyze_file(path)

        _, elapsed = _timed(analyze_file, path)
        assert elapsed < 0.050, f"analyze_file took {elapsed * 1000:.1f}ms, expected < 50ms"

    def test_analyze_all_fixtures_under_500ms(self):
        """Full analyze_file on all 6 fixtures should take < 500ms."""
        from jaxtyc.analyzer.pipeline import analyze_file

        fixtures = sorted(FIXTURES.glob("*.py"))
        assert len(fixtures) >= 6

        # Warm up
        for f in fixtures:
            analyze_file(str(f))

        start = time.perf_counter()
        for f in fixtures:
            analyze_file(str(f))
        elapsed = time.perf_counter() - start

        assert elapsed < 0.500, (
            f"Analyzing {len(fixtures)} files took {elapsed * 1000:.1f}ms, expected < 500ms"
        )
