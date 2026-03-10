[![PyPI](https://img.shields.io/pypi/v/jaxtyc)](https://pypi.org/project/jaxtyc/) [![Python](https://img.shields.io/pypi/pyversions/jaxtyc)](https://pypi.org/project/jaxtyc/) [![License](https://img.shields.io/github/license/BeeGass/jaxtyc)](https://github.com/BeeGass/jaxtyc/blob/main/LICENSE)

# jaxtyc

**Static array shape checking for JAX.** jaxtyc reads your [jaxtyping](https://github.com/patrick-kidger/jaxtyping) annotations and verifies array shapes at analysis time using `jax.eval_shape`. Zero runtime cost, zero FLOPs -- shapes are checked symbolically before any computation runs. Each named dimension is assigned a unique prime number, making shape mismatches between dimensions unambiguous: if the output prime doesn't match the expected prime, the dimension name is wrong.

## Features

- **Zero runtime cost** -- uses `jax.eval_shape` only; no arrays allocated, no computation executed
- **Prime-based symbolic shapes** -- each dimension name maps to a unique prime (>= 101), so `d_in != d_out` is guaranteed (no accidental size collisions)
- **10 diagnostic rules** -- shape/rank mismatch, cross-function shape propagation, parameter consistency, tuple return checking, trace errors, and more
- **Inline suppressions** -- `# jaxtyc: ignore` and `# jaxtyc: ignore[rule-name]` to suppress diagnostics per-line
- **LSP server** -- diagnostics, hover, CodeLens, go-to-definition, references, rename, code actions, completion, semantic tokens, inlay hints, signature help, linked editing, folding, call hierarchy, and config hot-reload
- **LSP multiplexer** -- `jaxtyc mux` runs ty/pyright alongside jaxtyc behind a single stdio pipe, merging results transparently
- **CLI with 4 output formats** -- `full` (human-readable), `concise` (one line per error), `json` (machine-readable), `github` (inline PR annotations)
- **Flax NNX + Equinox support** -- traces bound methods on module instances via `nnx.eval_shape` / `jax.eval_shape`
- **CI-ready** -- `jaxtyc check --format github` emits `::error` annotations that GitHub Actions renders inline on PRs
- **Configurable via `pyproject.toml`** -- severity threshold, rule ignoring, file exclusion, einops preferences under `[tool.jaxtyc]`
- **Venv auto-discovery** -- follows ty's resolution order (`VIRTUAL_ENV` -> `.venv` at project root -> walk-up) so dependencies are found automatically

---

## Installation

=== "uv"

    ```bash
    uv add jaxtyc
    ```

=== "pip"

    ```bash
    pip install jaxtyc
    ```

**Prerequisites:** Python >=3.11 (!=3.12), JAX >=0.4.20, jaxtyping >=0.2.28.

**Extras:**

| Extra | Installs | Use case |
|-------|----------|----------|
| `jaxtyc[watch]` | `watchfiles` | `jaxtyc watch` -- re-check on file save |
| `jaxtyc[flax]` | `flax >=0.10` | Flax NNX module tracing |
| `jaxtyc[equinox]` | `equinox >=0.11` | Equinox module tracing |
| `jaxtyc[einops]` | `einops >=0.8` | einops-style fix suggestions in code actions |
| `jaxtyc[all]` | All of the above | Everything |

**Verify:**

```bash
jaxtyc version
```

---

## Quick Start

Write a function with jaxtyping annotations:

```python
# model.py
import jax.numpy as jnp
from jaxtyping import Array, Float

def linear(
    x: Float[Array, "batch seq d_in"],
    w: Float[Array, "d_in d_out"],
) -> Float[Array, "batch seq d_out"]:
    return jnp.matmul(x, w.T)  # Bug: .T swaps dims, produces (batch, seq, d_in)
```

Run the checker:

```bash
$ jaxtyc check model.py
model.py:8:0: error[shape-mismatch]
  Shape mismatch in return of `linear`
    Expected: (batch, seq, d_out)
    Got:      (batch, seq, d_in)

Found 1 error(s) in 1 function(s) checked (0.03s)
```

`w.T` transposes `(d_in, d_out)` into `(d_out, d_in)`, so `matmul(x, w.T)` contracts over `d_out` and produces `d_in` in the last axis -- not the annotated `d_out`. jaxtyc catches this because prime(d_in) != prime(d_out).

!!! tip "Fix"
    Use `jnp.matmul(x, w)` (no transpose) when `w` is already shaped `(d_in, d_out)`, or annotate `w` as `Float[Array, "d_out d_in"]` if you intend to transpose.
