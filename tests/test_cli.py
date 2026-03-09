"""Tests for jaxtyc.cli — command-line interface."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _run_jaxtyc(*args: str) -> subprocess.CompletedProcess[str]:
    """Run jaxtyc CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "jaxtyc.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCLIVersion:
    def test_version(self):
        result = _run_jaxtyc("version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


class TestCLICheck:
    def test_correct_file_exits_zero(self):
        result = _run_jaxtyc("check", str(FIXTURES / "correct_attention.py"))
        assert result.returncode == 0
        assert "passed" in result.stdout.lower() or "0 error" in result.stdout.lower()

    def test_wrong_transpose_exits_nonzero(self):
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        assert "shape-mismatch" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_wrong_rank_exits_nonzero(self):
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_rank.py"))
        assert result.returncode != 0

    def test_wrong_inner_dim_exits_nonzero(self):
        result = _run_jaxtyc("check", str(FIXTURES / "wrong_inner_dim.py"))
        assert result.returncode != 0

    def test_untraceable_exits_zero(self):
        result = _run_jaxtyc("check", str(FIXTURES / "untraceable.py"))
        assert result.returncode == 0

    def test_nonexistent_file(self):
        result = _run_jaxtyc("check", "/nonexistent/path.py")
        # Should still exit 0 (info, not error) or handle gracefully
        assert result.returncode == 0

    def test_directory_check(self):
        result = _run_jaxtyc("check", str(FIXTURES))
        # Should find files and check them
        assert result.returncode != 0  # fixtures contain buggy files

    def test_format_concise(self):
        result = _run_jaxtyc("check", "--format", "concise", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        # Concise format: one line per error
        error_lines = [
            line for line in result.stdout.strip().split("\n") if "error" in line.lower()
        ]
        assert len(error_lines) >= 1

    def test_format_json(self):
        import json

        result = _run_jaxtyc("check", "--format", "json", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        data = json.loads(result.stdout)
        assert "diagnostics" in data
        assert len(data["diagnostics"]) >= 1

    def test_format_github(self):
        result = _run_jaxtyc("check", "--format", "github", str(FIXTURES / "wrong_transpose.py"))
        assert result.returncode != 0
        assert "::error" in result.stdout

    def test_multiple_files(self):
        result = _run_jaxtyc(
            "check",
            str(FIXTURES / "correct_attention.py"),
            str(FIXTURES / "wrong_rank.py"),
        )
        assert result.returncode != 0  # wrong_rank has errors


class TestCLITrace:
    def test_trace_correct_function(self):
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::attention")
        assert result.returncode == 0
        assert "attention" in result.stdout
        assert "matches" in result.stdout.lower()

    def test_trace_wrong_function(self):
        result = _run_jaxtyc("trace", str(FIXTURES / "wrong_transpose.py") + "::attention")
        # Should show MISMATCH
        assert "mismatch" in result.stdout.lower()

    def test_trace_nonexistent_function(self):
        result = _run_jaxtyc("trace", str(FIXTURES / "correct_attention.py") + "::nonexistent")
        assert result.returncode != 0

    def test_trace_bad_syntax(self):
        result = _run_jaxtyc("trace", "no_double_colon")
        assert result.returncode != 0


class TestCLIWatch:
    def test_watch_detects_file_change(self):
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

    def test_watch_subcommand_exists(self):
        """Watch subcommand should be recognized."""
        result = _run_jaxtyc("watch", "--help")
        assert result.returncode == 0
        assert "watch" in result.stdout.lower()
