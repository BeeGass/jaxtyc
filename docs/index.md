[![PyPI](https://img.shields.io/pypi/v/jaxtyc)](https://pypi.org/project/jaxtyc/) [![Python](https://img.shields.io/pypi/pyversions/jaxtyc)](https://pypi.org/project/jaxtyc/) [![License](https://img.shields.io/github/license/BeeGass/jaxtyc)](https://github.com/BeeGass/jaxtyc/blob/main/LICENSE)

# jaxtyc

**Static array shape checking for JAX.** jaxtyc reads your [jaxtyping](https://github.com/patrick-kidger/jaxtyping) annotations and verifies array shapes at analysis time using `jax.eval_shape`. Zero runtime cost, zero FLOPs -- shapes are checked symbolically before any computation runs. Each named dimension is assigned a unique prime number, making shape mismatches between dimensions unambiguous: if the output prime doesn't match the expected prime, the dimension name is wrong.

## Features

- **Zero runtime cost** -- uses `jax.eval_shape` only; no arrays allocated, no computation executed
- **Prime-based symbolic shapes** -- each dimension name maps to a unique prime, so `d_in != d_out` is guaranteed (no accidental size collisions)
- **LSP server** -- inline diagnostics, hover for intermediate shapes, CodeLens showing resolved shapes above function definitions
- **CLI with 4 output formats** -- `full` (human-readable), `concise` (one line per error), `json` (machine-readable), `github` (inline PR annotations)
- **CI-ready** -- `jaxtyc check --format github` emits `::error` annotations that GitHub Actions renders inline on PRs
- **Configurable via `pyproject.toml`** -- severity threshold, rule ignoring, file exclusion patterns under `[tool.jaxtyc]`

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
| `jaxtyc[flax]` | `flax >=0.10` | Flax NNX module support |
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
