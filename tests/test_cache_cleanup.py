"""Tests for JAX cache cleanup after analysis."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch


class TestJaxCacheCleanup:
    """Test that jax.clear_caches() is called after analysis."""

    def test_clear_caches_called_after_lsp_analysis(self) -> None:
        """_analyze_and_publish should call jax.clear_caches() after publishing."""
        from jaxtyc.lsp import _state
        from jaxtyc.lsp.server import _analyze_and_publish

        uri = "file:///tmp/test_cache_cleanup.py"
        source = "x = 1\n"

        _state.content_hash_cache.pop(uri, None)

        ls = MagicMock()
        ls.client_capabilities = None

        with (
            patch("jaxtyc.lsp.server.analyze_file") as mock_analyze,
            patch("jax.clear_caches") as mock_clear,
        ):
            mock_analyze.return_value = MagicMock(
                diagnostics=[],
                trace_results=[],
                functions_checked=0,
                file_path="/tmp/test_cache_cleanup.py",
            )
            _analyze_and_publish(ls, uri, source)
            mock_clear.assert_called()

        _state.content_hash_cache.pop(uri, None)
