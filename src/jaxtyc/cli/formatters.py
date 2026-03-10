"""Output formatters for CLI diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable

from jaxtyc.types import FileResult


def format_full(results: list[FileResult], elapsed: float) -> str:
    """Human-readable full format with context."""
    lines: list[str] = []
    total_errors = 0
    total_checked = 0

    for result in results:
        total_checked += result.functions_checked
        errors = [d for d in result.diagnostics if d.severity == "error"]
        total_errors += len(errors)

        for diag in result.diagnostics:
            if diag.severity == "error":
                lines.append(f"{diag.file}:{diag.line}:{diag.col}: error[{diag.rule}]")
                for msg_line in diag.message.split("\n"):
                    lines.append(f"  {msg_line}")
                lines.append("")

    if total_errors == 0:
        lines.append(f"All checks passed: {total_checked} function(s) checked ({elapsed:.2f}s)")
    else:
        lines.append(
            f"Found {total_errors} error(s) in {total_checked} function(s) checked ({elapsed:.2f}s)"
        )

    return "\n".join(lines)


def format_concise(results: list[FileResult], elapsed: float) -> str:
    """One line per error."""
    lines: list[str] = []
    total_errors = 0
    total_checked = 0

    for result in results:
        total_checked += result.functions_checked
        for diag in result.diagnostics:
            if diag.severity == "error":
                total_errors += 1
                first_line = diag.message.split("\n")[0]
                lines.append(f"{diag.file}:{diag.line}:{diag.col}: error[{diag.rule}] {first_line}")

    if total_errors == 0:
        lines.append(f"All checks passed ({total_checked} checked, {elapsed:.2f}s)")
    else:
        lines.append(f"{total_errors} error(s) ({total_checked} checked, {elapsed:.2f}s)")

    return "\n".join(lines)


def format_json(results: list[FileResult], elapsed: float) -> str:
    """Machine-readable JSON format."""
    diagnostics = []
    total_checked = 0

    for result in results:
        total_checked += result.functions_checked
        for diag in result.diagnostics:
            diagnostics.append(
                {
                    "file": diag.file,
                    "line": diag.line,
                    "col": diag.col,
                    "severity": diag.severity,
                    "message": diag.message,
                    "rule": diag.rule,
                }
            )

    return json.dumps(
        {
            "diagnostics": diagnostics,
            "functions_checked": total_checked,
            "elapsed_seconds": round(elapsed, 3),
        },
        indent=2,
    )


def format_github(results: list[FileResult], elapsed: float) -> str:
    """GitHub Actions annotation format."""
    lines: list[str] = []

    for result in results:
        for diag in result.diagnostics:
            if diag.severity == "error":
                first_line = diag.message.split("\n")[0]
                lines.append(
                    f"::error file={diag.file},line={diag.line},col={diag.col}::{first_line}"
                )

    return "\n".join(lines)


FORMATTERS: dict[str, Callable[[list[FileResult], float], str]] = {
    "full": format_full,
    "concise": format_concise,
    "json": format_json,
    "github": format_github,
}
