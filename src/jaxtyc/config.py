"""Configuration loading for jaxtyc from pyproject.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyc.types import Diagnostic
    from jaxtyc.types import Severity

_DEFAULT_SHARDING_RULES: list[str] = [
    "sharding-rank-mismatch",
    "sharding-axis-unknown",
    "sharding-conflict",
    "sharding-io-mismatch",
    "sharding-propagation-mismatch",
    "sharding-annotation-incomplete",
    "sharding-dim-conflict",
    "sharding-mesh-undefined",
]


@dataclass(frozen=True)
class HintsConfig:
    """Configuration for error inlay hints.

    Attributes:
        error_mode: How to display error info in inlay hints.
            "both" shows shape AND error, "replace" shows only error.
        error_location: Where to place error hints.
            "divergence" at first divergence point, "annotation" at the
            return annotation, "return" at the return statement, "both"
            at divergence and annotation.
        error_style: Separator style between shape and error text.
            "pipe" uses " | ", "icon" uses a warning triangle.
        dtype_style: How to display dtype in inlay hints.
            "numpy" uses abbreviations (f32, bf16, i32),
            "jax" uses full JAX names (float32, bfloat16),
            "jaxtyping" uses capitalized names (Float32, BFloat16).
    """

    error_mode: str = "both"
    error_location: str = "divergence"
    error_style: str = "pipe"
    dtype_style: str = "numpy"


@dataclass(frozen=True)
class ShardingConfig:
    """Configuration for sharding display and validation.

    Attributes:
        display: How to show sharding info in inlay hints.
            "all" shows dim|axis on every line with sharding info,
            "constrained_only" shows only for lines with explicit
            sharding constraints, "off" disables.
        rules: Allow-list of sharding diagnostic rules to enable.
    """

    display: str = "all"
    rules: list[str] = field(default_factory=lambda: list(_DEFAULT_SHARDING_RULES))
    mesh: dict[str, int] = field(default_factory=dict)
    axis_rules: dict[str, str] = field(default_factory=dict)
    strict_annotation: bool = True


@dataclass(frozen=True)
class NavigationConfig:
    """Configuration for LSP navigation features.

    Attributes:
        references_scope: Scope for finding references.
            "file" searches current file only, "workspace" searches all
            workspace files.
        include_external_calls: Whether to include calls to/from external
            (non-workspace) modules in navigation results.
    """

    references_scope: str = "workspace"
    include_external_calls: bool = True


_HINTS_KNOWN_KEYS: frozenset[str] = frozenset(HintsConfig.__dataclass_fields__.keys())
_SHARDING_KNOWN_KEYS: frozenset[str] = frozenset(ShardingConfig.__dataclass_fields__.keys())
_NAVIGATION_KNOWN_KEYS: frozenset[str] = frozenset(NavigationConfig.__dataclass_fields__.keys())


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
        hints: Nested config for error inlay hints.
        sharding: Nested config for sharding display and validation.
    """

    severity: Severity = "error"
    ignore_rules: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    debounce_ms: int = 500
    prefer_einops: bool = False
    hover_compact: bool = True
    hints: HintsConfig = field(default_factory=HintsConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)


_KNOWN_KEYS: frozenset[str] = frozenset(JaxtycConfig.__dataclass_fields__.keys()) - {
    "hints",
    "sharding",
    "navigation",
}


def _build_nested_config(
    raw: dict[str, object],
    cls: type,
    known_keys: frozenset[str],
) -> object:
    """Build a nested config dataclass, filtering unknown keys."""
    filtered = {k: v for k, v in raw.items() if k in known_keys}
    return cls(**filtered)


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

    # Pop nested subsections before filtering top-level keys
    hints_raw = jaxtyc_config.pop("hints", None)
    sharding_raw = jaxtyc_config.pop("sharding", None)
    navigation_raw = jaxtyc_config.pop("navigation", None)

    # Filter to known top-level keys only
    filtered: dict[str, object] = {k: v for k, v in jaxtyc_config.items() if k in _KNOWN_KEYS}

    # Build nested configs
    if isinstance(hints_raw, dict):
        filtered["hints"] = _build_nested_config(hints_raw, HintsConfig, _HINTS_KNOWN_KEYS)
    if isinstance(sharding_raw, dict):
        filtered["sharding"] = _build_nested_config(
            sharding_raw, ShardingConfig, _SHARDING_KNOWN_KEYS
        )
    if isinstance(navigation_raw, dict):
        filtered["navigation"] = _build_nested_config(
            navigation_raw, NavigationConfig, _NAVIGATION_KNOWN_KEYS
        )

    # Environment variable override for einops preference
    if os.environ.get("JAXTYC_PREFER_EINOPS", "").strip() in ("1", "true"):
        filtered["prefer_einops"] = True

    return JaxtycConfig(**filtered)


_SEVERITY_LEVELS: dict[str, int] = {"error": 3, "warning": 2, "info": 1}


def filter_diagnostics(diagnostics: list[Diagnostic], config: JaxtycConfig) -> list[Diagnostic]:
    """Filter diagnostics based on config severity threshold and ignore_rules.

    Sharding diagnostics (rules starting with "sharding-") are additionally
    filtered by the ``config.sharding.rules`` allow-list.

    Args:
        diagnostics: Unfiltered list of diagnostics from analysis.
        config: JaxtycConfig controlling severity threshold and ignored rules.

    Returns:
        Filtered list containing only diagnostics at or above the configured
        severity level and not matching any ignored rule.
    """
    min_level = _SEVERITY_LEVELS.get(config.severity, 1)
    ignored = set(config.ignore_rules)
    sharding_allowlist = set(config.sharding.rules)
    result: list[Diagnostic] = []
    for d in diagnostics:
        if _SEVERITY_LEVELS.get(d.severity, 1) < min_level:
            continue
        if d.rule in ignored:
            continue
        if d.rule.startswith("sharding-") and d.rule not in sharding_allowlist:
            continue
        result.append(d)
    return result
