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

    def test_sys_path_not_polluted(self) -> None:
        original_path = sys.path[:]
        import_module_from_path(str(FIXTURES / "correct_attention.py"))
        # Only venv entries (if any) should persist; parent dir should not
        new_entries = set(sys.path) - set(original_path)
        for entry in new_entries:
            assert "site-packages" in entry  # Only venv additions allowed

    def test_module_removed_from_sys_modules(self) -> None:
        """After import, the user module should NOT remain in sys.modules."""
        module = import_module_from_path(str(FIXTURES / "correct_attention.py"))
        assert hasattr(module, "attention")
        found = any(
            name.startswith("_jaxtyc_user_") and name.endswith("correct_attention")
            for name in sys.modules
        )
        assert not found, "User module should be removed from sys.modules after import"

    def test_repeated_imports_no_sys_modules_leak(self) -> None:
        """Importing the same file multiple times should not leak modules."""
        before_count = sum(1 for name in sys.modules if name.startswith("_jaxtyc_user_"))
        for _ in range(5):
            import_module_from_path(str(FIXTURES / "correct_attention.py"))
        after_count = sum(1 for name in sys.modules if name.startswith("_jaxtyc_user_"))
        assert after_count == before_count, (
            f"Leaked {after_count - before_count} modules into sys.modules"
        )

    def test_multiple_imports_no_collision(self) -> None:
        m1 = import_module_from_path(str(FIXTURES / "correct_attention.py"))
        m2 = import_module_from_path(str(FIXTURES / "wrong_transpose.py"))
        assert hasattr(m1, "attention")
        assert hasattr(m2, "attention")
