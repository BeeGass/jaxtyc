"""jaxtyc - Static array shape checking for JAX powered by eval_shape."""

from jaxtyc.analyzer.pipeline import analyze_file
from jaxtyc.types import Diagnostic
from jaxtyc.types import FileResult
from jaxtyc.types import TraceResult

__version__ = "0.1.0"

__all__ = [
    "Diagnostic",
    "FileResult",
    "TraceResult",
    "analyze_file",
]
