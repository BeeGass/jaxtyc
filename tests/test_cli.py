"""Tests for jaxtyc.cli — command-line interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _run_jaxtyc(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run jaxtyc CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "jaxtyc.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )


class TestCLIVersion:
    def test_version(self) -> None:
        result = _run_jaxtyc("version")
        assert result.returncode == 0
        assert "jaxtyc" in result.stdout

    def test_version_matches_package(self) -> None:
        from importlib.metadata import version

        result = _run_jaxtyc("version")
        assert version("jaxtyc") in result.stdout


class TestCLICheck:
    def test_correct_file_exits_zero(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "correct_attention.py"))
        assert result.returncode == 0
        assert "passed" in result.stdout.lower() or "0 error" in result.stdout.lower()

    def test_wrong_transpose_exits_nonzero(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        assert "shape-mismatch" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_wrong_rank_exits_nonzero(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_rank.py"))
        assert result.returncode != 0

    def test_wrong_inner_dim_exits_nonzero(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_inner_dim.py"))
        assert result.returncode != 0

    def test_untraceable_exits_zero(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "untraceable.py"))
        assert result.returncode == 0

    def test_nonexistent_file(self) -> None:
        result = _run_jaxtyc("check", "/nonexistent/path.py")
        # Should still exit 0 (info, not error) or handle gracefully
        assert result.returncode == 0

    def test_directory_check(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES))
        # Should find files and check them
        assert result.returncode != 0  # fixtures contain buggy files

    def test_format_concise(self) -> None:
        result = _run_jaxtyc("check", "--format", "concise", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        # Concise format: one line per error
        error_lines = [
            line for line in result.stdout.strip().split("\n") if "error" in line.lower()
        ]
        assert len(error_lines) >= 1

    def test_format_json(self) -> None:
        result = _run_jaxtyc("check", "--format", "json", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "diagnostics" in data
        assert len(data["diagnostics"]) >= 1

    def test_format_json_fields(self) -> None:
        result = _run_jaxtyc("check", "--format", "json", str(FIXTURES / "wrong_transpose.py"))
        data = json.loads(result.stdout)
        assert "functions_checked" in data
        assert "elapsed_seconds" in data
        assert isinstance(data["elapsed_seconds"], float)
        diag = data["diagnostics"][0]
        assert "file" in diag
        assert "line" in diag
        assert "col" in diag
        assert "severity" in diag
        assert "message" in diag
        assert "rule" in diag

    def test_format_json_correct_file(self) -> None:
        result = _run_jaxtyc("check", "--format", "json", str(FIXTURES / "correct_attention.py"))
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["diagnostics"] == []
        assert data["functions_checked"] >= 1

    def test_format_github(self) -> None:
        result = _run_jaxtyc("check", "--format", "github", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        assert "::error" in result.stdout

    def test_format_github_fields(self) -> None:
        result = _run_jaxtyc("check", "--format", "github", str(FIXTURES / "wrong_transpose.py"))
        lines = [line for line in result.stdout.strip().split("\n") if line.startswith("::error")]
        assert len(lines) >= 1
        assert "file=" in lines[0]
        assert "line=" in lines[0]

    def test_multiple_files(self) -> None:
        result = _run_jaxtyc(
            "check",
            str(FIXTURES / "correct_attention.py"),
            str(FIXTURES / "wrong_rank.py"),
        )
        assert result.returncode != 0  # wrong_rank has errors

    def test_multiple_correct_files(self) -> None:
        result = _run_jaxtyc(
            "check",
            str(FIXTURES / "correct_attention.py"),
            str(FIXTURES / "ellipsis_patterns.py"),
        )
        assert result.returncode == 0

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_jaxtyc("check", tmpdir)
            # No python files => no errors, exit 0
            assert result.returncode == 0

    def test_no_args_shows_help(self) -> None:
        result = _run_jaxtyc()
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "jaxtyc" in result.stdout.lower()


class TestCLICheckWithConfig:
    """Integration tests: CLI check subcommand respects [tool.jaxtyc] config."""

    def test_severity_warning_shows_warnings(self) -> None:
        """With severity=warning, warnings are not filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy a buggy fixture into the temp dir
            shutil.copy(FIXTURES / "wrong_transpose.py", Path(tmpdir) / "wrong_transpose.py")
            # Write a pyproject.toml with warning severity
            (Path(tmpdir) / "pyproject.toml").write_text('[tool.jaxtyc]\nseverity = "warning"\n')
            result = _run_jaxtyc("check", str(Path(tmpdir) / "wrong_transpose.py"), cwd=tmpdir)
            assert result.returncode != 0

    def test_ignore_rules_suppresses_rule(self) -> None:
        """ignore_rules in config should suppress matching diagnostics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(FIXTURES / "wrong_transpose.py", Path(tmpdir) / "wrong_transpose.py")
            # Suppress shape-mismatch via config
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[tool.jaxtyc]\nignore_rules = ["shape-mismatch"]\n'
            )
            result = _run_jaxtyc("check", str(Path(tmpdir) / "wrong_transpose.py"), cwd=tmpdir)
            # shape-mismatch is the only error, so suppressing it => exit 0
            assert result.returncode == 0

    def test_ignore_rules_partial(self) -> None:
        """Suppressing one rule should not suppress others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(FIXTURES / "wrong_rank.py", Path(tmpdir) / "wrong_rank.py")
            # Suppress shape-mismatch but NOT rank-mismatch
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[tool.jaxtyc]\nignore_rules = ["shape-mismatch"]\n'
            )
            result = _run_jaxtyc("check", str(Path(tmpdir) / "wrong_rank.py"), cwd=tmpdir)
            # rank-mismatch should still be reported
            assert result.returncode != 0

    def test_exclude_pattern(self) -> None:
        """exclude glob patterns should skip matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(FIXTURES / "wrong_transpose.py", Path(tmpdir) / "wrong_transpose.py")
            shutil.copy(FIXTURES / "correct_attention.py", Path(tmpdir) / "correct_attention.py")
            (Path(tmpdir) / "pyproject.toml").write_text('[tool.jaxtyc]\nexclude = ["*wrong*"]\n')
            result = _run_jaxtyc("check", tmpdir, cwd=tmpdir)
            # wrong_transpose is excluded, only correct_attention is checked => exit 0
            assert result.returncode == 0

    def test_missing_config_uses_defaults(self) -> None:
        """Without pyproject.toml, defaults should apply."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(FIXTURES / "wrong_transpose.py", Path(tmpdir) / "wrong_transpose.py")
            result = _run_jaxtyc("check", str(Path(tmpdir) / "wrong_transpose.py"), cwd=tmpdir)
            assert result.returncode != 0


class TestCLITrace:
    def test_trace_correct_function(self) -> None:
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::attention")
        assert result.returncode == 0
        assert "attention" in result.stdout
        assert "matches" in result.stdout.lower()

    def test_trace_wrong_function(self) -> None:
        result = _run_jaxtyc("trace", str(FIXTURES / "wrong_transpose.py") + "::attention")
        # Should show MISMATCH
        assert "mismatch" in result.stdout.lower()

    def test_trace_nonexistent_function(self) -> None:
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::nonexistent")
        assert result.returncode != 0

    def test_trace_bad_syntax(self) -> None:
        result = _run_jaxtyc("trace", "no_double_colon")
        assert result.returncode != 0

    def test_trace_nonexistent_file(self) -> None:
        result = _run_jaxtyc("trace", "/nonexistent/file.py::fn")
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_trace_shows_intermediates(self) -> None:
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::attention")
        assert result.returncode == 0
        # Should show intermediate shapes
        assert "Output:" in result.stdout

    def test_trace_shows_param_shapes(self) -> None:
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::attention")
        assert result.returncode == 0
        # Should list parameter annotations
        assert "q:" in result.stdout or "batch" in result.stdout


class TestCLIWatch:
    def test_watch_detects_file_change(self) -> None:
        """Watch mode should detect a file change and re-analyze."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy a correct fixture into the temp dir
            src = FIXTURES / "correct_attention.py"
            target = Path(tmpdir) / "test_file.py"
            shutil.copy(src, target)

            # Start watch in background
            proc = subprocess.Popen(
                [sys.executable, "-m", "jaxtyc.cli.main", "watch", tmpdir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                # Give watchfiles time to start monitoring
                time.sleep(1)

                # Touch the file to trigger a change
                target.write_text(target.read_text() + "\n")

                # Wait for output (watch should re-analyze)
                time.sleep(3)
            finally:
                proc.terminate()
                stdout, stderr = proc.communicate(timeout=5)

            # Should have produced some analysis output
            assert "watching" in stdout.lower() or "checked" in stdout.lower()

    def test_watch_subcommand_exists(self) -> None:
        """Watch subcommand should be recognized."""
        result = _run_jaxtyc("watch", "--help")
        assert result.returncode == 0
        assert "watch" in result.stdout.lower()


class TestCLIMain:
    """Test main() directly for coverage (not via subprocess)."""

    def test_main_version(self) -> None:
        from jaxtyc.cli.main import main

        rc = main(["version"])
        assert rc == 0

    def test_main_check_correct(self) -> None:
        from jaxtyc.cli.main import main

        rc = main(["check", str(FIXTURES / "correct_attention.py")])
        assert rc == 0

    def test_main_check_buggy(self) -> None:
        from jaxtyc.cli.main import main

        rc = main(["check", str(FIXTURES / "wrong_transpose.py")])
        assert rc == 1

    def test_main_no_args(self) -> None:
        from jaxtyc.cli.main import main

        rc = main([])
        assert rc == 0

    def test_main_trace_bad_syntax(self) -> None:
        from jaxtyc.cli.main import main

        rc = main(["trace", "no_colon"])
        assert rc == 1


class TestCLIHelpers:
    """Unit tests for CLI helper functions."""

    def test_collect_python_files_single_file(self) -> None:
        from jaxtyc.cli.main import _collect_python_files

        result = _collect_python_files([str(FIXTURES / "correct_attention.py")])
        assert len(result) == 1
        assert result[0].endswith("correct_attention.py")

    def test_collect_python_files_directory(self) -> None:
        from jaxtyc.cli.main import _collect_python_files

        result = _collect_python_files([str(FIXTURES)])
        assert len(result) >= 5
        assert all(f.endswith(".py") for f in result)

    def test_collect_python_files_nonexistent(self) -> None:
        from jaxtyc.cli.main import _collect_python_files

        result = _collect_python_files(["/nonexistent/file.py"])
        assert result == ["/nonexistent/file.py"]

    def test_collect_python_files_mixed(self) -> None:
        from jaxtyc.cli.main import _collect_python_files

        result = _collect_python_files(
            [
                str(FIXTURES / "correct_attention.py"),
                str(FIXTURES),
            ]
        )
        # Should include the single file + all files from directory
        assert len(result) >= 6

    def test_apply_exclude(self) -> None:
        from jaxtyc.cli.main import _apply_exclude

        files = ["/a/foo.py", "/a/bar.py", "/a/test_foo.py"]
        result = _apply_exclude(files, ["*test_*"])
        assert result == ["/a/foo.py", "/a/bar.py"]

    def test_apply_exclude_no_patterns(self) -> None:
        from jaxtyc.cli.main import _apply_exclude

        files = ["/a/foo.py", "/a/bar.py"]
        result = _apply_exclude(files, [])
        assert result == files

    def test_apply_exclude_multiple_patterns(self) -> None:
        from jaxtyc.cli.main import _apply_exclude

        files = ["/a/foo.py", "/a/bar.py", "/a/baz.py"]
        result = _apply_exclude(files, ["*foo*", "*baz*"])
        assert result == ["/a/bar.py"]


class TestCLICheckEndToEnd:
    """Full CLI end-to-end tests with all fixtures."""

    def test_tuple_return_correct(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "tuple_return.py"))
        assert result.returncode == 0

    def test_tuple_return_mismatch(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "tuple_return_mismatch.py"))
        assert result.returncode != 0

    def test_cross_function_mismatch(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "cross_function_mismatch.py"))
        assert result.returncode != 0

    def test_ellipsis_patterns(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "ellipsis_patterns.py"))
        assert result.returncode == 0

    def test_suppressed_file(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "suppressed.py"))
        # The fixture has one unsuppressed function (wrong_not_suppressed) that errors
        assert result.returncode != 0
        # But suppressed functions should not appear in output
        assert "wrong_not_suppressed" in result.stdout
        assert "wrong_sum_suppressed" not in result.stdout

    def test_multi_function(self) -> None:
        result = _run_jaxtyc("check", str(FIXTURES / "multi_function.py"))
        # multi_function may have trace errors from prime-sized matmul mismatches
        # but should not crash
        assert result.returncode in (0, 1)

    def test_json_output_all_fixtures(self) -> None:
        """JSON format should produce valid JSON for any input."""
        result = _run_jaxtyc("check", "--format", "json", str(FIXTURES))
        data = json.loads(result.stdout)
        assert isinstance(data["diagnostics"], list)
        assert isinstance(data["functions_checked"], int)

    def test_concise_output_no_crash(self) -> None:
        result = _run_jaxtyc("check", "--format", "concise", str(FIXTURES))
        assert "error" in result.stdout.lower() or "passed" in result.stdout.lower()

    def test_github_output_no_crash(self) -> None:
        result = _run_jaxtyc("check", "--format", "github", str(FIXTURES))
        # Should not crash, may or may not have ::error lines
        assert result.returncode != 0 or result.stdout == ""


class TestCLICheckWithEnvOverrides:
    """Test environment variable overrides in CLI."""

    def test_prefer_einops_env_var(self) -> None:
        """JAXTYC_PREFER_EINOPS env var should be respected."""
        env = dict(os.environ, JAXTYC_PREFER_EINOPS="1")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jaxtyc.cli.main",
                "check",
                str(FIXTURES / "correct_attention.py"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 0
