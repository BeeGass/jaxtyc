"""Tests for jaxtyc.lsp.suggestions — shape fix generation."""

from __future__ import annotations

from jaxtyc.lsp.suggestions import suggest_fixes


class TestTransposeDetection:
    def test_simple_swap(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5, 5),
            actual=(2, 3, 7, 7),
            dim_names={2: "batch", 3: "heads", 5: "seq", 7: "head_dim"},
        )
        # Not a transpose (different multiset), should get reshape suggestions
        assert len(fixes) >= 1

    def test_actual_transpose(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5, 7),
            actual=(2, 3, 7, 5),
            dim_names={2: "batch", 3: "heads", 5: "seq", 7: "d_model"},
        )
        assert any(f.kind == "transpose" for f in fixes)
        jax_fix = next(f for f in fixes if "swapaxes" in f.code or "transpose" in f.code)
        assert jax_fix is not None

    def test_transpose_with_einops(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5, 7),
            actual=(2, 3, 7, 5),
            dim_names={2: "batch", 3: "heads", 5: "seq", 7: "d_model"},
            prefer_einops=True,
        )
        assert fixes[0].kind == "transpose"
        assert "einops" in fixes[0].code
        assert "rearrange" in fixes[0].code


class TestExpandDims:
    def test_missing_dimension(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5),
            actual=(2, 3),
            dim_names={2: "batch", 3: "seq", 5: "d_model"},
        )
        assert any(f.kind == "expand" for f in fixes)
        jax_fix = next(f for f in fixes if "expand_dims" in f.code)
        assert jax_fix is not None

    def test_expand_with_einops(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5),
            actual=(2, 3),
            dim_names={2: "batch", 3: "seq", 5: "d_model"},
            prefer_einops=True,
        )
        assert fixes[0].kind == "expand"
        assert "einops" in fixes[0].code


class TestSqueezeDims:
    def test_extra_dimension(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3),
            actual=(2, 3, 5),
            dim_names={2: "batch", 3: "seq", 5: "d_model"},
        )
        assert any(f.kind == "squeeze" for f in fixes)

    def test_squeeze_with_einops(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3),
            actual=(2, 3, 5),
            dim_names={2: "batch", 3: "seq", 5: "d_model"},
            prefer_einops=True,
        )
        assert fixes[0].kind == "squeeze"
        assert "einops" in fixes[0].code


class TestEinopsPreference:
    def test_default_jax_native_first(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5, 7),
            actual=(2, 3, 7, 5),
            dim_names={2: "batch", 3: "heads", 5: "seq", 7: "d_model"},
            prefer_einops=False,
        )
        # JAX-native should come first when prefer_einops is False
        assert "jnp" in fixes[0].code or "jax" in fixes[0].code.lower()

    def test_einops_first_when_preferred(self) -> None:
        fixes = suggest_fixes(
            expected=(2, 3, 5, 7),
            actual=(2, 3, 7, 5),
            dim_names={2: "batch", 3: "heads", 5: "seq", 7: "d_model"},
            prefer_einops=True,
        )
        assert "einops" in fixes[0].code
