"""Tests for CPU backend enforcement in jaxtyc."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch


class TestEnforceCpuBackend:
    """Test that _enforce_cpu_backend() correctly sets JAX_PLATFORMS."""

    def test_sets_jax_platforms_when_unset(self) -> None:
        """When JAX_PLATFORMS is not set and JAXTYC_BACKEND is not set,
        _enforce_cpu_backend() should set JAX_PLATFORMS to 'cpu'."""
        from jaxtyc.cli.main import _enforce_cpu_backend

        env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "JAXTYC_BACKEND")}
        with patch.dict(os.environ, env, clear=True):
            _enforce_cpu_backend()
            assert os.environ["JAX_PLATFORMS"] == "cpu"

    def test_preserves_existing_jax_platforms(self) -> None:
        """When JAX_PLATFORMS is already set by the user, do not override it."""
        from jaxtyc.cli.main import _enforce_cpu_backend

        env = {**os.environ, "JAX_PLATFORMS": "gpu"}
        with patch.dict(os.environ, env, clear=True):
            _enforce_cpu_backend()
            assert os.environ["JAX_PLATFORMS"] == "gpu"

    def test_jaxtyc_backend_gpu_skips_override(self) -> None:
        """When JAXTYC_BACKEND=gpu, do not set JAX_PLATFORMS."""
        from jaxtyc.cli.main import _enforce_cpu_backend

        env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
        env["JAXTYC_BACKEND"] = "gpu"
        with patch.dict(os.environ, env, clear=True):
            _enforce_cpu_backend()
            assert "JAX_PLATFORMS" not in os.environ

    def test_jaxtyc_backend_cpu_sets_platforms(self) -> None:
        """When JAXTYC_BACKEND=cpu (default), set JAX_PLATFORMS=cpu."""
        from jaxtyc.cli.main import _enforce_cpu_backend

        env = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "JAXTYC_BACKEND")}
        env["JAXTYC_BACKEND"] = "cpu"
        with patch.dict(os.environ, env, clear=True):
            _enforce_cpu_backend()
            assert os.environ["JAX_PLATFORMS"] == "cpu"


class TestCpuBackendIntegration:
    """Integration test: verify the CLI subprocess only uses CPU."""

    def test_cli_check_uses_cpu_backend(self) -> None:
        """Running jaxtyc CLI should only see CPU devices."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from jaxtyc.cli.main import main; "
                    "import jax; "
                    "assert all('cpu' in str(d).lower() for d in jax.devices()), "
                    "f'Expected CPU only, got {jax.devices()}'"
                ),
            ],
            capture_output=True,
            text=True,
            env={
                k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "JAXTYC_BACKEND")
            },
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
