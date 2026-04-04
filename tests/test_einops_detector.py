"""Tests for jaxtyc.analyzer.einops_detector -- AST-based einops call detection."""

from __future__ import annotations

from jaxtyc.analyzer.einops_detector import extract_einops_calls


class TestRearrangeDetection:
    def test_attribute_call(self) -> None:
        source = """
import einops
y = einops.rearrange(x, 'b c h w -> b (c h) w')
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 1
        assert calls[0].operation == "rearrange"
        assert calls[0].pattern == "b c h w -> b (c h) w"
        assert calls[0].output_names == ("b", "c*h", "w")
        assert calls[0].line == 3

    def test_bare_import_call(self) -> None:
        source = """
from einops import rearrange
y = rearrange(x, 'b c h w -> b c h w')
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 1
        assert calls[0].operation == "rearrange"

    def test_double_quoted_pattern(self) -> None:
        source = """
import einops
y = einops.rearrange(x, "b c -> c b")
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 1
        assert calls[0].output_names == ("c", "b")


class TestReduceDetection:
    def test_reduce_call(self) -> None:
        source = """
import einops
y = einops.reduce(x, 'b c h w -> b c', 'mean')
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 1
        assert calls[0].operation == "reduce"
        assert calls[0].output_names == ("b", "c")


class TestRepeatDetection:
    def test_repeat_call(self) -> None:
        source = """
import einops
y = einops.repeat(x, 'b c -> b c h', h=4)
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 1
        assert calls[0].operation == "repeat"
        assert calls[0].output_names == ("b", "c", "h")


class TestEdgeCases:
    def test_variable_pattern_skipped(self) -> None:
        source = """
import einops
pattern = 'b c -> c b'
y = einops.rearrange(x, pattern)
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 0

    def test_non_einops_call_ignored(self) -> None:
        source = """
def rearrange(x, pattern):
    return x
y = rearrange(x, 'b c -> c b')
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 0

    def test_multiple_calls(self) -> None:
        source = """
import einops
y = einops.rearrange(x, 'b c -> c b')
z = einops.reduce(y, 'c b -> c', 'sum')
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 2

    def test_syntax_error_returns_empty(self) -> None:
        calls = extract_einops_calls("def broken(")
        assert calls == []

    def test_no_einops_returns_empty(self) -> None:
        source = """
import jax.numpy as jnp
y = jnp.reshape(x, (2, 3))
"""
        calls = extract_einops_calls(source)
        assert len(calls) == 0
