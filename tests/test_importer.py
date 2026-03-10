"""Tests for jaxtyc.analyzer.importer — module import from file path."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from jaxtyc.analyzer.importer import import_module_from_path

FIXTURES = Path(__file__).parent / "fixtures"


class TestImportModuleFromPath:
    def test_valid_import(self) -> None:
        module = import_module_from_path(str(FIXTURES / "correct_attention.py"))
        assert hasattr(module, "attention")

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(ImportError, match="File not found"):
            import_module_from_path("/nonexistent/path/module.py")

    def test_non_py_file_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not python")
            f.flush()
            with pytest.raises(ImportError, match="Not a Python file"):
                import_module_from_path(f.name)
            Path(f.name).unlink()

    def test_adds_parent_to_sys_path(self) -> None:
        parent = str(FIXTURES.resolve())
        # Remove if already present to test the addition
        if parent in sys.path:
            sys.path.remove(parent)
        import_module_from_path(str(FIXTURES / "correct_attention.py"))
        assert parent in sys.path

    def test_unique_module_name(self) -> None:
        module = import_module_from_path(str(FIXTURES / "correct_attention.py"))
        # Should be in sys.modules with a unique name
        found = any(
            name.endswith("correct_attention") and name.startswith("_jaxtyc_user_")
            for name in sys.modules
        )
        assert found

    def test_multiple_imports_no_collision(self) -> None:
        m1 = import_module_from_path(str(FIXTURES / "correct_attention.py"))
        m2 = import_module_from_path(str(FIXTURES / "wrong_transpose.py"))
        assert hasattr(m1, "attention")
        assert hasattr(m2, "attention")
