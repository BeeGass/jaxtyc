"""Safely import user modules for JAX tracing."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def import_module_from_path(file_path: str) -> ModuleType:
    """Import a Python module from a file path.

    Adds the file's parent directory to sys.path so relative imports work,
    then loads the module via importlib.

    Raises ImportError if the module cannot be loaded.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise ImportError(f"File not found: {file_path}")
    if not path.suffix == ".py":
        raise ImportError(f"Not a Python file: {file_path}")

    # Add parent directory to sys.path for imports
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = path.stem
    # Use a unique name to avoid collisions with already-imported modules
    unique_name = f"_jaxtyc_user_.{path.parent.name}.{module_name}"

    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module
