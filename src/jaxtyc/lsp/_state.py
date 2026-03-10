"""Shared mutable state for the jaxtyc LSP server."""

from __future__ import annotations

import threading
from typing import Any

from jaxtyc.config import JaxtycConfig
from jaxtyc.lsp.index import WorkspaceIndex
from jaxtyc.types import IntermediateShape

# Cached analysis results per URI
analysis_cache: dict[str, list[IntermediateShape]] = {}

# CodeLens data per URI: list of (0-indexed line, title text)
codelens_cache: dict[str, list[tuple[int, str]]] = {}

# Diagnostics cache per URI for pull model
diagnostics_cache: dict[str, list[Any]] = {}

# DimEnv cache per URI for hover enhancement
dim_env_cache: dict[str, object] = {}

# Debounce state
debounce_timers: dict[str, threading.Timer] = {}
debounce_lock: threading.Lock = threading.Lock()

# Server config
config: JaxtycConfig = JaxtycConfig()

# Workspace-level index
workspace_index: WorkspaceIndex = WorkspaceIndex()
