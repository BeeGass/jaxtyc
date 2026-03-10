"""Inline suppression comment parsing and filtering."""

from __future__ import annotations

import re

from jaxtyc.types import Diagnostic
from jaxtyc.types import SuppressionComment

_SUPPRESSION_RE: re.Pattern[str] = re.compile(r"#\s*jaxtyc:\s*ignore(?:\[([^\]]*)\])?")


def extract_suppressions(source: str) -> list[SuppressionComment]:
    """Extract inline suppression comments from source code.

    Supported syntax:
        - ``# jaxtyc: ignore`` -- suppress all rules on this line
        - ``# jaxtyc: ignore[shape-mismatch]`` -- suppress specific rule
        - ``# jaxtyc: ignore[rule1, rule2]`` -- suppress multiple rules

    Args:
        source: Python source code as a string.

    Returns:
        List of SuppressionComment objects, one per suppression comment found.
    """
    suppressions: list[SuppressionComment] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        match = _SUPPRESSION_RE.search(line)
        if match is None:
            continue
        rules_str = match.group(1)
        if rules_str is None:
            rules: frozenset[str] = frozenset()
        else:
            rules = frozenset(r.strip() for r in rules_str.split(",") if r.strip())
        suppressions.append(SuppressionComment(line=lineno, rules=rules))
    return suppressions


def filter_inline_suppressions(
    diagnostics: list[Diagnostic],
    suppressions: list[SuppressionComment],
) -> list[Diagnostic]:
    """Filter diagnostics that are suppressed by inline comments.

    A diagnostic is suppressed if:
        - A suppression comment exists on the same line or the line before
        - The suppression covers all rules (empty rules set) or includes the
          diagnostic's specific rule

    Args:
        diagnostics: Unfiltered diagnostics from analysis.
        suppressions: Parsed suppression comments from the source.

    Returns:
        Filtered list with suppressed diagnostics removed.
    """
    if not suppressions:
        return diagnostics

    suppression_map: dict[int, frozenset[str]] = {}
    for s in suppressions:
        suppression_map[s.line] = s.rules

    result: list[Diagnostic] = []
    for d in diagnostics:
        suppressed = False
        # Check same line and line before (for multi-line signatures)
        for check_line in (d.line, d.line - 1):
            if check_line in suppression_map:
                rules = suppression_map[check_line]
                if not rules or d.rule in rules:
                    suppressed = True
                    break
        if not suppressed:
            result.append(d)
    return result
