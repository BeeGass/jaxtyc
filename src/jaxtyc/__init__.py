"""jaxtyc - Static array shape checking for JAX powered by eval_shape."""

from importlib.metadata import version as _pkg_version

from jaxtyc.analyzer.pipeline import analyze_file
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
