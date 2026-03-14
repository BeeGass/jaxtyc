"""Unit tests for LSP handler helper functions (_completion, _signature_help, _code_actions)."""

from __future__ import annotations

from unittest.mock import patch

from jaxtyc.config import JaxtycConfig
from jaxtyc.config import ShardingConfig
from jaxtyc.lsp._completion import _get_mesh_axis_completions
from jaxtyc.lsp._completion import _get_partial_word
from jaxtyc.lsp._completion import _is_after_pipe
from jaxtyc.lsp._completion import _is_in_shape_string
from jaxtyc.lsp._signature_help import _find_call_context


class TestIsInShapeString:
    def test_inside_shape_string(self) -> None:
        line = 'def f(x: Float[Array, "batch seq"]):'
        assert _is_in_shape_string(line, 25)

    def test_outside_shape_string(self) -> None:
        line = 'def f(x: Float[Array, "batch seq"]):'
        assert not _is_in_shape_string(line, 5)

    def test_unclosed_string(self) -> None:
        line = 'x: Float[Array, "batch seq'
        assert _is_in_shape_string(line, 25)

    def test_before_opening_quote(self) -> None:
        line = 'x: Float[Array, "batch seq"]'
        assert not _is_in_shape_string(line, 16)

    def test_at_closing_quote(self) -> None:
        line = 'x: Float[Array, "batch seq"]'
        # Col at the closing quote itself is outside
        closing_pos = line.index('"', 17)
        assert not _is_in_shape_string(line, closing_pos + 1)

    def test_no_shape_string(self) -> None:
        line = "x = 42"
        assert not _is_in_shape_string(line, 3)

    def test_shaped_annotation(self) -> None:
        line = 'x: Shaped[Array, "batch dim"]'
        assert _is_in_shape_string(line, 22)

    def test_int_annotation(self) -> None:
        line = 'x: Int[Array, "batch seq"]'
        assert _is_in_shape_string(line, 18)


class TestIsAfterPipe:
    def test_after_pipe(self) -> None:
        line = '"batch|d'
        assert _is_after_pipe(line, 8)

    def test_not_after_pipe(self) -> None:
        line = '"batch seq'
        assert not _is_after_pipe(line, 10)

    def test_after_pipe_no_partial(self) -> None:
        line = '"batch|'
        assert _is_after_pipe(line, 7)

    def test_pipe_in_middle(self) -> None:
        line = '"batch|dp seq'
        assert not _is_after_pipe(line, 13)


class TestGetPartialWord:
    def test_partial_word(self) -> None:
        line = '"bat'
        assert _get_partial_word(line, 4) == "bat"

    def test_empty_prefix(self) -> None:
        line = '"batch '
        assert _get_partial_word(line, 7) == ""

    def test_full_word(self) -> None:
        line = '"batch"'
        assert _get_partial_word(line, 6) == "batch"

    def test_after_space(self) -> None:
        line = '"batch se'
        assert _get_partial_word(line, 9) == "se"


class TestGetMeshAxisCompletions:
    def test_empty_mesh(self) -> None:
        config = JaxtycConfig(sharding=ShardingConfig(mesh={}))
        with patch("jaxtyc.lsp._completion._state") as mock_state:
            mock_state.config = config
            assert _get_mesh_axis_completions("") == []

    def test_filters_by_prefix(self) -> None:
        config = JaxtycConfig(sharding=ShardingConfig(mesh={"data": 4, "model": 2, "dp": 8}))
        with patch("jaxtyc.lsp._completion._state") as mock_state:
            mock_state.config = config
            result = _get_mesh_axis_completions("d")
            assert result == ["data", "dp"]

    def test_no_prefix_returns_all(self) -> None:
        config = JaxtycConfig(sharding=ShardingConfig(mesh={"data": 4, "model": 2}))
        with patch("jaxtyc.lsp._completion._state") as mock_state:
            mock_state.config = config
            result = _get_mesh_axis_completions("")
            assert result == ["data", "model"]


class TestFindCallContext:
    def test_simple_call(self) -> None:
        line = "attention(q, k, v)"
        name, idx = _find_call_context(line, 14)
        assert name == "attention"
        assert idx == 1

    def test_nested_call(self) -> None:
        line = "outer(inner(x), y)"
        name, idx = _find_call_context(line, 17)
        assert name == "outer"
        assert idx == 1

    def test_no_call_context(self) -> None:
        line = "x = 42"
        name, idx = _find_call_context(line, 5)
        assert name is None

    def test_method_call(self) -> None:
        line = "self.forward(x, y)"
        name, idx = _find_call_context(line, 16)
        assert name == "forward"
        assert idx == 1

    def test_first_param(self) -> None:
        line = "fn(x"
        name, idx = _find_call_context(line, 4)
        assert name == "fn"
        assert idx == 0

    def test_after_third_comma(self) -> None:
        line = "f(a, b, c, d)"
        name, idx = _find_call_context(line, 12)
        assert name == "f"
        assert idx == 3

    def test_empty_parens(self) -> None:
        line = "fn()"
        name, idx = _find_call_context(line, 3)
        assert name == "fn"
        assert idx == 0
