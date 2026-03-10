"""Tests for jaxtyc.cli.formatters — output formatting."""

from __future__ import annotations

import json

from jaxtyc.cli.formatters import format_concise
from jaxtyc.cli.formatters import format_full
from jaxtyc.cli.formatters import format_github
from jaxtyc.cli.formatters import format_json
from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult


def _make_result(
    errors: int = 0, warnings: int = 0, infos: int = 0, checked: int = 1
) -> FileResult:
    diags: list[Diagnostic] = []
    for i in range(errors):
        diags.append(
            Diagnostic(
                file="model.py",
                line=i + 1,
                col=0,
                severity="error",
                message=f"Shape mismatch #{i}",
                rule="shape-mismatch",
            )
        )
    for i in range(warnings):
        diags.append(
            Diagnostic(
                file="model.py",
                line=10 + i,
                col=0,
                severity="warning",
                message=f"Warning #{i}",
                rule="some-warning",
            )
        )
    for i in range(infos):
        diags.append(
            Diagnostic(
                file="model.py",
                line=20 + i,
                col=0,
                severity="info",
                message=f"Info #{i}",
                rule="some-info",
            )
        )
    return FileResult(
        file_path="model.py",
        functions_checked=checked,
        diagnostics=diags,
        trace_results=[],
    )


class TestFormatFull:
    def test_no_errors(self) -> None:
        result = _make_result(errors=0)
        output = format_full([result], 0.5)
        assert "All checks passed" in output
        assert "0.50s" in output

    def test_with_errors(self) -> None:
        result = _make_result(errors=2)
        output = format_full([result], 0.1)
        assert "Found 2 error(s)" in output
        assert "shape-mismatch" in output

    def test_multiline_message(self) -> None:
        result = _make_result(errors=1)
        output = format_full([result], 0.0)
        assert "model.py:1:0" in output


class TestFormatConcise:
    def test_one_line_per_error(self) -> None:
        result = _make_result(errors=3)
        output = format_concise([result], 0.1)
        error_lines = [line for line in output.split("\n") if "error" in line.lower()]
        assert len(error_lines) >= 3

    def test_no_errors(self) -> None:
        result = _make_result(errors=0)
        output = format_concise([result], 0.1)
        assert "All checks passed" in output


class TestFormatJson:
    def test_valid_json(self) -> None:
        result = _make_result(errors=1)
        output = format_json([result], 0.1)
        data = json.loads(output)
        assert "diagnostics" in data
        assert "functions_checked" in data
        assert "elapsed_seconds" in data

    def test_diagnostics_count(self) -> None:
        result = _make_result(errors=2, warnings=1)
        output = format_json([result], 0.1)
        data = json.loads(output)
        assert len(data["diagnostics"]) == 3

    def test_diagnostic_fields(self) -> None:
        result = _make_result(errors=1)
        output = format_json([result], 0.1)
        data = json.loads(output)
        diag = data["diagnostics"][0]
        assert diag["file"] == "model.py"
        assert diag["severity"] == "error"
        assert diag["rule"] == "shape-mismatch"


class TestFormatGithub:
    def test_annotation_format(self) -> None:
        result = _make_result(errors=1)
        output = format_github([result], 0.1)
        assert "::error" in output
        assert "file=model.py" in output
        assert "line=1" in output

    def test_no_errors_empty(self) -> None:
        result = _make_result(errors=0)
        output = format_github([result], 0.1)
        assert "::error" not in output
