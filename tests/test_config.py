"""Tests for jaxtyc.config — configuration loading and filtering."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from jaxtyc.config import HintsConfig
from jaxtyc.config import JaxtycConfig
from jaxtyc.config import ShardingConfig
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


class TestLoadConfigLogging:
    def test_corrupt_toml_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt pyproject.toml should log a warning and return defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_bytes(b"[invalid toml content <<<")
            with caplog.at_level(logging.WARNING, logger="jaxtyc.config"):
                config = load_config(tmpdir)
            assert config == JaxtycConfig()
            assert any("Failed to parse" in r.message for r in caplog.records)
            assert any(r.levelno == logging.WARNING for r in caplog.records)


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


class TestHintsConfig:
    def test_hints_config_defaults(self) -> None:
        """HintsConfig has correct defaults."""
        cfg = HintsConfig()
        assert cfg.error_mode == "both"
        assert cfg.error_location == "divergence"
        assert cfg.error_style == "pipe"
        assert cfg.dtype_style == "numpy"

    def test_hints_config_frozen(self) -> None:
        """HintsConfig should be immutable."""
        import pytest

        cfg = HintsConfig()
        with pytest.raises(AttributeError):
            cfg.error_mode = "replace"  # type: ignore[misc]

    def test_dtype_style_configurable(self) -> None:
        """dtype_style can be set to jax or jaxtyping."""
        cfg_jax = HintsConfig(dtype_style="jax")
        assert cfg_jax.dtype_style == "jax"
        cfg_jt = HintsConfig(dtype_style="jaxtyping")
        assert cfg_jt.dtype_style == "jaxtyping"


class TestDtypeFormat:
    def test_abbreviate_dtype_numpy_style(self) -> None:
        """numpy style abbreviates common dtypes."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float32", "numpy") == "f32"
        assert format_dtype("float64", "numpy") == "f64"
        assert format_dtype("float16", "numpy") == "f16"
        assert format_dtype("bfloat16", "numpy") == "bf16"
        assert format_dtype("int32", "numpy") == "i32"
        assert format_dtype("int64", "numpy") == "i64"
        assert format_dtype("int8", "numpy") == "i8"
        assert format_dtype("uint8", "numpy") == "u8"
        assert format_dtype("bool", "numpy") == "bool"
        assert format_dtype("complex64", "numpy") == "c64"

    def test_format_dtype_jax_style(self) -> None:
        """jax style returns dtypes as-is."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float32", "jax") == "float32"
        assert format_dtype("bfloat16", "jax") == "bfloat16"

    def test_format_dtype_jaxtyping_style(self) -> None:
        """jaxtyping style capitalizes dtype names."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float32", "jaxtyping") == "Float32"
        assert format_dtype("bfloat16", "jaxtyping") == "BFloat16"
        assert format_dtype("int32", "jaxtyping") == "Int32"
        assert format_dtype("bool", "jaxtyping") == "Bool"

    def test_format_dtype_fp8_numpy(self) -> None:
        """FP8 variants get abbreviated in numpy style."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float8_e4m3fn", "numpy") == "f8e4m3fn"
        assert format_dtype("float8_e5m2", "numpy") == "f8e5m2"
        assert format_dtype("float8_e4m3fnuz", "numpy") == "f8e4m3fnuz"
        assert format_dtype("float8_e5m2fnuz", "numpy") == "f8e5m2fnuz"
        assert format_dtype("float8_e4m3b11fnuz", "numpy") == "f8e4m3b11fnuz"
        assert format_dtype("float8_e4m3", "numpy") == "f8e4m3"
        assert format_dtype("float8_e3m4", "numpy") == "f8e3m4"
        assert format_dtype("float8_e8m0fnu", "numpy") == "f8e8m0fnu"

    def test_format_dtype_fp4_fp6_numpy(self) -> None:
        """FP4 and FP6 variants get abbreviated in numpy style."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float4_e2m1fn", "numpy") == "f4e2m1fn"
        assert format_dtype("float6_e2m3fn", "numpy") == "f6e2m3fn"
        assert format_dtype("float6_e3m2fn", "numpy") == "f6e3m2fn"

    def test_format_dtype_sub_byte_int_numpy(self) -> None:
        """Sub-byte integer types get abbreviated in numpy style."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("int2", "numpy") == "i2"
        assert format_dtype("int4", "numpy") == "i4"
        assert format_dtype("uint2", "numpy") == "u2"
        assert format_dtype("uint4", "numpy") == "u4"

    def test_format_dtype_fp8_jaxtyping(self) -> None:
        """FP8 variants get capitalized in jaxtyping style."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("float8_e4m3fn", "jaxtyping") == "Float8E4M3FN"
        assert format_dtype("float8_e5m2", "jaxtyping") == "Float8E5M2"
        assert format_dtype("float8_e4m3fnuz", "jaxtyping") == "Float8E4M3FNUZ"

    def test_format_dtype_unknown_passes_through(self) -> None:
        """Unknown dtype strings pass through unchanged."""
        from jaxtyc.lsp._util import format_dtype

        assert format_dtype("weird_type", "numpy") == "weird_type"
        assert format_dtype("weird_type", "jax") == "weird_type"


class TestShardingConfig:
    def test_sharding_config_defaults(self) -> None:
        """ShardingConfig has correct defaults."""
        cfg = ShardingConfig()
        assert cfg.display == "all"
        assert len(cfg.rules) == 8

    def test_sharding_config_frozen(self) -> None:
        """ShardingConfig should be immutable."""
        import pytest

        cfg = ShardingConfig()
        with pytest.raises(AttributeError):
            cfg.display = "off"  # type: ignore[misc]


class TestShardingConfigMesh:
    def test_mesh_from_toml(self) -> None:
        """Load mesh and axis_rules from [tool.jaxtyc.sharding]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                "[tool.jaxtyc.sharding]\n"
                "mesh = { data = 4, model = 2 }\n"
                'axis_rules = { dp = "data", mp = "model" }\n'
                "strict_annotation = true\n"
            )
            config = load_config(tmpdir)
            assert config.sharding.mesh == {"data": 4, "model": 2}
            assert config.sharding.axis_rules == {"dp": "data", "mp": "model"}
            assert config.sharding.strict_annotation is True

    def test_mesh_defaults_empty(self) -> None:
        """Default ShardingConfig has empty mesh and axis_rules."""
        config = ShardingConfig()
        assert config.mesh == {}
        assert config.axis_rules == {}
        assert config.strict_annotation is True

    def test_strict_annotation_false(self) -> None:
        """strict_annotation can be set to false."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.jaxtyc.sharding]\nstrict_annotation = false\n")
            config = load_config(tmpdir)
            assert config.sharding.strict_annotation is False

    def test_mesh_only(self) -> None:
        """Load just mesh without axis_rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.jaxtyc.sharding]\nmesh = { dp = 8 }\n")
            config = load_config(tmpdir)
            assert config.sharding.mesh == {"dp": 8}
            assert config.sharding.axis_rules == {}


class TestNestedConfig:
    def test_jaxtyc_config_has_nested(self) -> None:
        """JaxtycConfig includes hints and sharding sub-configs."""
        cfg = JaxtycConfig()
        assert isinstance(cfg.hints, HintsConfig)
        assert isinstance(cfg.sharding, ShardingConfig)

    def test_load_hints_subsection(self) -> None:
        """load_config reads [tool.jaxtyc.hints] from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[tool.jaxtyc.hints]\nerror_mode = "replace"\n')
            cfg = load_config(tmpdir)
            assert cfg.hints.error_mode == "replace"
            assert cfg.hints.error_location == "divergence"  # default preserved

    def test_load_dtype_style_from_toml(self) -> None:
        """load_config reads dtype_style from [tool.jaxtyc.hints]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[tool.jaxtyc.hints]\ndtype_style = "jaxtyping"\n')
            cfg = load_config(tmpdir)
            assert cfg.hints.dtype_style == "jaxtyping"

    def test_load_sharding_subsection(self) -> None:
        """load_config reads [tool.jaxtyc.sharding] from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                '[tool.jaxtyc.sharding]\ndisplay = "off"\nrules = ["sharding-rank-mismatch"]\n'
            )
            cfg = load_config(tmpdir)
            assert cfg.sharding.display == "off"
            assert cfg.sharding.rules == ["sharding-rank-mismatch"]

    def test_nested_defaults_when_missing(self) -> None:
        """Missing subsections use defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.jaxtyc]\nseverity = 'warning'\n")
            cfg = load_config(tmpdir)
            assert cfg.hints.error_mode == "both"
            assert cfg.sharding.display == "all"

    def test_unknown_nested_keys_ignored(self) -> None:
        """Unknown keys inside subsections are silently ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text('[tool.jaxtyc.hints]\nbogus_key = true\nerror_mode = "replace"\n')
            cfg = load_config(tmpdir)
            assert cfg.hints.error_mode == "replace"

    def test_filter_diagnostics_sharding_allowlist(self) -> None:
        """Sharding rules not in config.sharding.rules are filtered out."""
        diags = [
            Diagnostic(
                file="f.py",
                line=1,
                col=0,
                severity="error",
                message="x",
                rule="sharding-rank-mismatch",
            ),
            Diagnostic(
                file="f.py",
                line=2,
                col=0,
                severity="error",
                message="y",
                rule="sharding-axis-unknown",
            ),
            Diagnostic(
                file="f.py",
                line=3,
                col=0,
                severity="error",
                message="z",
                rule="shape-mismatch",
            ),
        ]
        cfg = JaxtycConfig(sharding=ShardingConfig(rules=["sharding-rank-mismatch"]))
        result = filter_diagnostics(diags, cfg)
        rules = [d.rule for d in result]
        assert "sharding-rank-mismatch" in rules
        assert "sharding-axis-unknown" not in rules  # filtered by allowlist
        assert "shape-mismatch" in rules  # non-sharding rules unaffected


class TestNavigationConfig:
    def test_navigation_config_defaults(self) -> None:
        """NavigationConfig has correct defaults."""
        from jaxtyc.config import NavigationConfig

        nav = NavigationConfig()
        assert nav.references_scope == "workspace"
        assert nav.include_external_calls is True

    def test_navigation_config_frozen(self) -> None:
        """NavigationConfig should be immutable."""
        import pytest

        from jaxtyc.config import NavigationConfig

        nav = NavigationConfig()
        with pytest.raises(AttributeError):
            nav.references_scope = "workspace"  # type: ignore[misc]

    def test_jaxtyc_config_has_navigation(self) -> None:
        """JaxtycConfig includes navigation sub-config."""
        from jaxtyc.config import NavigationConfig

        cfg = JaxtycConfig()
        assert isinstance(cfg.navigation, NavigationConfig)

    def test_load_navigation_subsection(self) -> None:
        """load_config reads [tool.jaxtyc.navigation] from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                "[tool.jaxtyc.navigation]\n"
                'references_scope = "file"\n'
                "include_external_calls = false\n"
            )
            cfg = load_config(tmpdir)
            assert cfg.navigation.references_scope == "file"
            assert cfg.navigation.include_external_calls is False

    def test_navigation_defaults_when_missing(self) -> None:
        """Missing [tool.jaxtyc.navigation] uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[tool.jaxtyc]\nseverity = 'warning'\n")
            cfg = load_config(tmpdir)
            assert cfg.navigation.references_scope == "workspace"
            assert cfg.navigation.include_external_calls is True

    def test_navigation_unknown_keys_ignored(self) -> None:
        """Unknown keys inside navigation subsection are silently ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                '[tool.jaxtyc.navigation]\nbogus_key = true\nreferences_scope = "workspace"\n'
            )
            cfg = load_config(tmpdir)
            assert cfg.navigation.references_scope == "workspace"
