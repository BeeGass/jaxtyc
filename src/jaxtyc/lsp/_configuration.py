"""Configuration change handlers for the jaxtyc LSP server."""

from __future__ import annotations

import logging

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from jaxtyc.config import load_config
from jaxtyc.lsp import _state
from jaxtyc.lsp._util import uri_to_path
from jaxtyc.lsp.server import server

logger: logging.Logger = logging.getLogger(__name__)


def _reload_config(ls: LanguageServer) -> None:
    """Reload config from workspace root."""
    root_uri = ls.workspace.root_uri
    if root_uri:
        root_path = uri_to_path(root_uri)
        _state.config = load_config(root_path)
        logger.info("Reloaded config: debounce_ms=%d", _state.config.debounce_ms)


@server.feature(types.WORKSPACE_DID_CHANGE_CONFIGURATION)
def did_change_configuration(
    ls: LanguageServer, params: types.DidChangeConfigurationParams
) -> None:
    """Reload config when client settings change."""
    _reload_config(ls)


@server.feature(types.WORKSPACE_DID_CHANGE_WATCHED_FILES)
def did_change_watched_files(ls: LanguageServer, params: types.DidChangeWatchedFilesParams) -> None:
    """React to pyproject.toml changes."""
    for change in params.changes:
        path = uri_to_path(change.uri)
        if path.endswith("pyproject.toml"):
            _reload_config(ls)
            break
