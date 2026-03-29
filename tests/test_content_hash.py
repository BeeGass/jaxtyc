"""Tests for content-hash gating in the LSP server."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock
from unittest.mock import patch

from jaxtyc.lsp import _state


class TestContentHashCache:
    """Test that content_hash_cache exists and is used correctly."""

    def test_content_hash_cache_exists(self) -> None:
        """_state should have a content_hash_cache dict."""
        assert hasattr(_state, "content_hash_cache")
        assert isinstance(_state.content_hash_cache, dict)

    def test_hash_stored_after_analysis(self) -> None:
        """After _analyze_and_publish, the content hash should be cached."""
        from jaxtyc.lsp.server import _analyze_and_publish

        uri = "file:///tmp/test_hash.py"
        source = "x = 1\n"
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        ls = MagicMock()
        ls.client_capabilities = None

        _state.content_hash_cache.pop(uri, None)

        with patch("jaxtyc.lsp.server.analyze_file") as mock_analyze:
            mock_analyze.return_value = MagicMock(
                diagnostics=[],
                trace_results=[],
                functions_checked=0,
                file_path="/tmp/test_hash.py",
            )
            _analyze_and_publish(ls, uri, source)

        assert _state.content_hash_cache.get(uri) == expected_hash

    def test_unchanged_content_skips_analysis(self) -> None:
        """When content hash matches, analyze_file should NOT be called again."""
        from jaxtyc.lsp.server import _analyze_and_publish

        uri = "file:///tmp/test_skip.py"
        source = "x = 1\n"
        content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

        _state.content_hash_cache[uri] = content_hash

        ls = MagicMock()
        ls.client_capabilities = None

        with patch("jaxtyc.lsp.server.analyze_file") as mock_analyze:
            _analyze_and_publish(ls, uri, source)
            mock_analyze.assert_not_called()

        _state.content_hash_cache.pop(uri, None)

    def test_changed_content_triggers_analysis(self) -> None:
        """When content hash differs, analyze_file SHOULD be called."""
        from jaxtyc.lsp.server import _analyze_and_publish

        uri = "file:///tmp/test_changed.py"
        old_source = "x = 1\n"
        new_source = "x = 2\n"

        _state.content_hash_cache[uri] = hashlib.sha256(old_source.encode("utf-8")).hexdigest()

        ls = MagicMock()
        ls.client_capabilities = None

        with patch("jaxtyc.lsp.server.analyze_file") as mock_analyze:
            mock_analyze.return_value = MagicMock(
                diagnostics=[],
                trace_results=[],
                functions_checked=0,
                file_path="/tmp/test_changed.py",
            )
            _analyze_and_publish(ls, uri, new_source)
            mock_analyze.assert_called_once()

        _state.content_hash_cache.pop(uri, None)

    def test_did_close_clears_hash(self) -> None:
        """When a document is closed, its content hash should be removed."""
        uri = "file:///tmp/test_close_hash.py"
        _state.content_hash_cache[uri] = "somehash"
        _state.diagnostics_cache[uri] = []
        _state.analysis_cache[uri] = []
        _state.codelens_cache[uri] = []
        _state.error_hints_cache[uri] = []
        _state.source_cache[uri] = ""
        _state.trace_results_cache[uri] = {}

        from jaxtyc.lsp._diagnostics import did_close

        ls = MagicMock()
        did_close(ls, MagicMock(text_document=MagicMock(uri=uri)))

        assert uri not in _state.content_hash_cache
