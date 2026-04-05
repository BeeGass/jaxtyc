"""jaxtyc - Static array shape checking for JAX powered by eval_shape."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyc.analyzer.pipeline import analyze_file as analyze_file

from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult
from jaxtyc.types import NamedShape
from jaxtyc.types import Severity
from jaxtyc.types import TraceResult

__version__: str = _pkg_version("jaxtyc")

__all__ = [
    "Diagnostic",
    "FileResult",
    "NamedShape",
    "Severity",
    "TraceResult",
    "analyze_file",
]


def __getattr__(name: str) -> object:
    """Lazy import for analyze_file to avoid eager JAX initialization.

    This allows the CLI's _enforce_cpu_backend() to set JAX_PLATFORMS=cpu
    before JAX is first imported.
    """
    if name == "analyze_file":
        from jaxtyc.analyzer.pipeline import analyze_file

        return analyze_file
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
