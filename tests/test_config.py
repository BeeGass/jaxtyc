"""Tests for jaxtyc.config — configuration loading and filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

from jaxtyc.config import JaxtycConfig
from jaxtyc.config import filter_diagnostics
from jaxtyc.config import load_config
from jaxtyc.types import Diagnostic
from jaxtyc.types import Severity


class TestDefaultConfig:
    def test_defaults(self) -> None:
        """Default config should have sensible defaults."""
        config = JaxtycConfig()
        assert config.severity == "error"
        assert config.ignore_rules == []
        assert config.exclude == []
        assert config.debounce_ms == 500

    def test_frozen(self) -> None:
        """Config should be immutable."""
        import pytest

        config = JaxtycConfig()
        with pytest.raises(AttributeError):
            config.severity = "warning"


class TestLoadConfig:
    def test_load_from_pyproject_toml(self) -> None:
        """Should load config from [tool.jaxtyc] in pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                "[tool.jaxtyc]\n"
                'severity = "warning"\n'
                'ignore_rules = ["shape-mismatch"]\n'
                'exclude = ["tests/**"]\n'
                "debounce_ms = 1000\n"
            )
            config = load_config(tmpdir)
            assert config.severity == "warning"
            assert config.ignore_rules == ["shape-mismatch"]
            assert config.exclude == ["tests/**"]
            assert config.debounce_ms == 1000

    def test_load_missing_pyproject(self) -> None:
        """Should return defaults when pyproject.toml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(tmpdir)
            assert config.severity == "error"
            assert config.ignore_rules == []

    def test_load_no_tool_section(self) -> None:
        """Should return defaults when [tool.jaxtyc] section is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[project]\nname = "foo"\n')
            config = load_config(tmpdir)
            assert config.severity == "error"

    def test_load_partial_config(self) -> None:
        """Should merge partial config with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[tool.jaxtyc]\nseverity = "info"\n')
            config = load_config(tmpdir)
            assert config.severity == "info"
            assert config.ignore_rules == []
            assert config.debounce_ms == 500

    def test_load_ignores_unknown_keys(self) -> None:
        """Unknown keys in config should be ignored, not cause errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[tool.jaxtyc]\nunknown_key = true\nseverity = "warning"\n')
            config = load_config(tmpdir)
            assert config.severity == "warning"


class TestFilterDiagnostics:
    def _make_diag(self, severity: Severity = "error", rule: str = "shape-mismatch") -> Diagnostic:
        return Diagnostic(
            file="test.py", line=1, col=0, severity=severity, message="test", rule=rule
        )

    def test_default_severity_shows_only_errors(self) -> None:
        """Default config (severity='error') should only include errors."""
        config = JaxtycConfig()
        diags = [self._make_diag("error"), self._make_diag("warning"), self._make_diag("info")]
        result = filter_diagnostics(diags, config)
        assert len(result) == 1
        assert result[0].severity == "error"

    def test_severity_info_shows_all(self) -> None:
        """severity='info' should include all diagnostics."""
        config = JaxtycConfig(severity="info")
        diags = [self._make_diag("error"), self._make_diag("warning"), self._make_diag("info")]
        result = filter_diagnostics(diags, config)
        assert len(result) == 3

    def test_filter_by_severity_warning(self) -> None:
        """severity='warning' should include errors and warnings, exclude info."""
        config = JaxtycConfig(severity="warning")
        diags = [self._make_diag("error"), self._make_diag("warning"), self._make_diag("info")]
        result = filter_diagnostics(diags, config)
        assert len(result) == 2
        assert all(d.severity in ("error", "warning") for d in result)

    def test_filter_by_severity_error(self) -> None:
        """severity='error' should only include errors."""
        config = JaxtycConfig(severity="error")
        diags = [self._make_diag("error"), self._make_diag("warning"), self._make_diag("info")]
        result = filter_diagnostics(diags, config)
        assert len(result) == 1
        assert result[0].severity == "error"

    def test_filter_by_ignore_rules(self) -> None:
        """Diagnostics with ignored rules should be excluded."""
        config = JaxtycConfig(ignore_rules=["shape-mismatch"])
        diags = [self._make_diag(rule="shape-mismatch"), self._make_diag(rule="rank-mismatch")]
        result = filter_diagnostics(diags, config)
        assert len(result) == 1
        assert result[0].rule == "rank-mismatch"

    def test_filter_combined(self) -> None:
        """Both severity and ignore_rules should apply together."""
        config = JaxtycConfig(severity="warning", ignore_rules=["rank-mismatch"])
        diags = [
            self._make_diag("error", "shape-mismatch"),
            self._make_diag("warning", "rank-mismatch"),
            self._make_diag("info", "shape-mismatch"),
        ]
        result = filter_diagnostics(diags, config)
        assert len(result) == 1
        assert result[0].severity == "error"
        assert result[0].rule == "shape-mismatch"
