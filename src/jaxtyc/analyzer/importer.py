"""Safely import user modules for JAX tracing."""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# Track which venvs we've already activated to avoid duplicates
_activated_venvs: set[str] = set()


def _find_project_root(start: Path) -> Path | None:
    """Walk up from *start* to find the project root (directory with pyproject.toml)."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _find_venv(start: Path) -> Path | None:
    """Discover a virtual environment, matching ty's resolution order.

    1. ``VIRTUAL_ENV`` env var (active venv)
    2. ``.venv`` or ``venv`` in the project root (directory with pyproject.toml)
    3. ``.venv`` or ``venv`` walking up from *start*
    """
    # 1. Active venv via env var
    env_var = os.environ.get("VIRTUAL_ENV")
    if env_var:
        venv = Path(env_var)
        if (venv / "pyvenv.cfg").exists():
            return venv

    # 2. Project root (where pyproject.toml lives)
    project_root = _find_project_root(start)
    if project_root is not None:
        for name in (".venv", "venv"):
            candidate = project_root / name
            if (candidate / "pyvenv.cfg").exists():
                return candidate

    # 3. Walk up from start directory
    for parent in (start, *start.parents):
        for name in (".venv", "venv"):
            candidate = parent / name
            if (candidate / "pyvenv.cfg").exists():
                return candidate
    return None


def _activate_venv(venv: Path) -> None:
    """Add a venv's site-packages to sys.path if not already present."""
    key = str(venv.resolve())
    if key in _activated_venvs:
        return

    # Find site-packages: lib/python3.*/site-packages
    pattern = str(venv / "lib" / "python3.*" / "site-packages")
    matches = glob.glob(pattern)
    if not matches:
        return

    site_packages = matches[0]
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

    _activated_venvs.add(key)


def import_module_from_path(file_path: str) -> ModuleType:
    """Import a Python module from a file path.

    Discovers the project's virtual environment by walking up from the file
    and adds its ``site-packages`` to ``sys.path``. Then adds the file's
    parent directory so relative imports work, and loads the module via
    ``importlib``. Each import uses a unique module name to avoid collisions.

    Args:
        file_path: Absolute or relative path to the ``.py`` file to import.

    Returns:
        The imported module object, ready for attribute access.

    Raises:
        ImportError: If the file does not exist, is not a ``.py`` file, or
            cannot be loaded by importlib.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise ImportError(f"File not found: {file_path}")
    if not path.suffix == ".py":
        raise ImportError(f"Not a Python file: {file_path}")

    # Venv activation is permanent (shared state for session)
    venv = _find_venv(path.parent)
    if venv is not None:
        _activate_venv(venv)

    # Save sys.path before adding temporary entries
    saved_path = sys.path[:]

    # Add project root and src/ directory to sys.path for package imports
    project_root = _find_project_root(path.parent)
    if project_root is not None:
        root_str = str(project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        src_dir = project_root / "src"
        if src_dir.is_dir() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    # Add parent directory to sys.path for imports
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = path.stem
    # Use a unique name to avoid collisions with already-imported modules
    unique_name = f"_jaxtyc_user_.{path.parent.name}.{module_name}"

    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        sys.path[:] = saved_path
        raise ImportError(f"Cannot create module spec for: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        import jax

        with jax.default_device(jax.devices("cpu")[0]):
            spec.loader.exec_module(module)
    except Exception:
        sys.path[:] = saved_path
        sys.modules.pop(unique_name, None)
        raise

    sys.path[:] = saved_path
    # Remove from sys.modules to prevent memory leaks. The caller holds
    # a reference to the module object and does not need it registered.
    sys.modules.pop(unique_name, None)
    return module
