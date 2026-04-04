"""Tests for jaxtyc.analyzer.einops_parser -- einops pattern string parsing."""

from __future__ import annotations

from jaxtyc.analyzer.einops_parser import parse_einops_output


class TestSimplePatterns:
    def test_identity_rearrange(self) -> None:
        result = parse_einops_output("b c h w -> b c h w")
        assert result is not None
        assert result.dim_names == ("b", "c", "h", "w")

    def test_transpose(self) -> None:
        result = parse_einops_output("b c h w -> b h w c")
        assert result is not None
        assert result.dim_names == ("b", "h", "w", "c")

    def test_single_dim(self) -> None:
        result = parse_einops_output("features -> features")
        assert result is not None
        assert result.dim_names == ("features",)


class TestMergedDims:
    def test_two_dims_merged(self) -> None:
        result = parse_einops_output("b c h w -> b (c h) w")
        assert result is not None
        assert result.dim_names == ("b", "c*h", "w")

    def test_three_dims_merged(self) -> None:
        result = parse_einops_output("b c h w -> b (c h w)")
        assert result is not None
        assert result.dim_names == ("b", "c*h*w")

    def test_multiple_groups(self) -> None:
        result = parse_einops_output("a b c d -> (a b) (c d)")
        assert result is not None
        assert result.dim_names == ("a*b", "c*d")


class TestReducePatterns:
    def test_reduce_trailing(self) -> None:
        result = parse_einops_output("b c h w -> b c")
        assert result is not None
        assert result.dim_names == ("b", "c")

    def test_reduce_to_scalar(self) -> None:
        result = parse_einops_output("b c -> ")
        assert result is not None
        assert result.dim_names == ()


class TestRepeatPatterns:
    def test_repeat_adds_dim(self) -> None:
        result = parse_einops_output("b c -> b c h")
        assert result is not None
        assert result.dim_names == ("b", "c", "h")


class TestAnonymousAndFixed:
    def test_underscore_anonymous(self) -> None:
        result = parse_einops_output("b _ h w -> b _ h w")
        assert result is not None
        assert result.dim_names == ("b", None, "h", "w")

    def test_numeric_fixed(self) -> None:
        result = parse_einops_output("b c -> b 1 c")
        assert result is not None
        assert result.dim_names == ("b", None, "c")


class TestMultiWordDimNames:
    def test_underscored_names(self) -> None:
        result = parse_einops_output("batch seq_len d_model -> batch seq_len d_model")
        assert result is not None
        assert result.dim_names == ("batch", "seq_len", "d_model")


class TestMalformedPatterns:
    def test_no_arrow(self) -> None:
        assert parse_einops_output("b c h w") is None

    def test_empty_string(self) -> None:
        assert parse_einops_output("") is None

    def test_unbalanced_parens(self) -> None:
        assert parse_einops_output("b c -> b (c") is None

    def test_nested_parens(self) -> None:
        assert parse_einops_output("b c -> ((b c))") is None
