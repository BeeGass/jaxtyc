"""Configuration loading for jaxtyc from pyproject.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyc.types import Diagnostic


@dataclass(frozen=True)
class JaxtycConfig:
    """jaxtyc configuration options.

    Attributes:
        severity: Minimum severity threshold for reported diagnostics.
            One of "error", "warning", or "info".
        ignore_rules: List of rule identifiers to suppress (e.g.
            ["rank-mismatch", "trace-error"]).
        exclude: Glob patterns for files to skip during analysis.
        debounce_ms: LSP debounce delay in milliseconds before re-analyzing
            after a file change.
    """

    severity: str = "error"
    ignore_rules: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    debounce_ms: int = 500


_KNOWN_KEYS = frozenset(JaxtycConfig.__dataclass_fields__.keys())


def load_config(project_root: str | Path) -> JaxtycConfig:
    """Load jaxtyc config from ``[tool.jaxtyc]`` in pyproject.toml.

    Unrecognised keys are silently ignored. Returns defaults if the file
    or section is missing.

    Args:
        project_root: Directory containing ``pyproject.toml``.

    Returns:
        JaxtycConfig populated from the TOML section, or default values
        if the file/section is absent or unreadable.
    """
    pyproject_path = Path(project_root) / "pyproject.toml"
    if not pyproject_path.exists():
        return JaxtycConfig()

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return JaxtycConfig()

    jaxtyc_config = data.get("tool", {}).get("jaxtyc", {})
    if not jaxtyc_config:
        return JaxtycConfig()

    # Filter to known keys only
    filtered = {k: v for k, v in jaxtyc_config.items() if k in _KNOWN_KEYS}
    return JaxtycConfig(**filtered)


_SEVERITY_LEVELS = {"error": 3, "warning": 2, "info": 1}


def filter_diagnostics(diagnostics: list[Diagnostic], config: JaxtycConfig) -> list[Diagnostic]:
    """Filter diagnostics based on config severity threshold and ignore_rules.

    Args:
        diagnostics: Unfiltered list of diagnostics from analysis.
        config: JaxtycConfig controlling severity threshold and ignored rules.

    Returns:
        Filtered list containing only diagnostics at or above the configured
        severity level and not matching any ignored rule.
    """
    min_level = _SEVERITY_LEVELS.get(config.severity, 1)
    ignored = set(config.ignore_rules)
    return [
        d
        for d in diagnostics
        if _SEVERITY_LEVELS.get(d.severity, 1) >= min_level and d.rule not in ignored
    ]
