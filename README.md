[![PyPI](https://img.shields.io/pypi/v/jaxtyc)](https://pypi.org/project/jaxtyc/) [![Python](https://img.shields.io/pypi/pyversions/jaxtyc)](https://pypi.org/project/jaxtyc/) [![CI](https://github.com/BeeGass/jaxtyc/actions/workflows/ci.yml/badge.svg)](https://github.com/BeeGass/jaxtyc/actions/workflows/ci.yml) [![License](https://img.shields.io/github/license/BeeGass/jaxtyc)](https://github.com/BeeGass/jaxtyc/blob/main/LICENSE)

# jaxtyc

Static array shape checking for JAX powered by `jax.eval_shape`.

Reads [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations and verifies shapes at analysis time -- no runtime cost, no FLOPs. Unlike jaxtyping's runtime beartype checks, jaxtyc catches shape bugs before your code ever runs on a GPU.

<p align="center">
  <img src="https://raw.githubusercontent.com/BeeGass/jaxtyc/main/docs/assets/vscode-inlay-hints.png" alt="VS Code inlay hints showing sharding annotations and shape overlays" width="600">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/BeeGass/jaxtyc/main/docs/assets/cli-diagnostics.png" alt="CLI diagnostics showing shape mismatches in Claude Code" width="600">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/BeeGass/jaxtyc/main/docs/assets/demo.gif" alt="jaxtyc CLI demo showing shape mismatch detection" width="800">
</p>

## How It Works

jaxtyc traces your annotated functions with `jax.eval_shape`, which propagates shapes through JAX operations without allocating any arrays. Each named dimension (`batch`, `d_model`, etc.) is assigned a unique prime number (>= 101), so distinct dimension names always produce distinct sizes -- making shape mismatches unambiguous and impossible to mask by accident.

## Features

- **Zero runtime cost** -- `jax.eval_shape` only; no arrays allocated, no computation executed
- **Prime-based symbolic shapes** -- each dimension name maps to a unique prime (>= 101), so `d_in != d_out` is guaranteed
- **18 diagnostic rules** -- shape/rank mismatch, cross-function propagation, parameter consistency, sharding validation, tuple return checking, trace errors
- **Sharding-in-types** -- annotate mesh axes inline with `|` syntax (`"batch|dp seq|None d_model|mp"`), validated against mesh topology with 8 dedicated sharding rules
- **Einops integration** -- detects `einops.rearrange`/`reduce`/`repeat` calls, parses pattern strings for shape checking, and provides einops-aware fix suggestions
- **Inline suppressions** -- `# jaxtyc: ignore` and `# jaxtyc: ignore[rule-name]`
- **LSP server** -- diagnostics, hover, CodeLens, go-to-definition, references, rename, code actions, completion, semantic tokens, inlay hints, signature help, linked editing, folding, call hierarchy
- **LSP multiplexer** -- `jaxtyc mux` runs ty/pyright + jaxtyc behind a single stdio pipe
- **CLI with 4 output formats** -- `full`, `concise`, `json`, `github` (inline PR annotations)
- **Flax NNX + Equinox support** -- traces bound methods on module instances
- **Configurable via `pyproject.toml`** -- severity threshold, rule ignoring, file exclusion, einops preferences

## Installation

```bash
pip install jaxtyc
# or
uv add jaxtyc
```

**Extras:**

| Extra | Installs | Use case |
|-------|----------|----------|
| `jaxtyc[watch]` | `watchfiles` | `jaxtyc watch` -- re-check on file save |
| `jaxtyc[flax]` | `flax >=0.10` | Flax NNX module tracing |
| `jaxtyc[equinox]` | `equinox >=0.11` | Equinox module tracing |
| `jaxtyc[einops]` | `einops >=0.8` | einops-style fix suggestions + inlay hints with pattern dim names |
| `jaxtyc[all]` | All of the above | Everything |

## Quick Start

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

```bash
$ jaxtyc check model.py
model.py:8:0: error[shape-mismatch]
  Shape mismatch in return of `linear`
    Expected: (batch, seq, d_out)
    Got:      (batch, seq, d_in)

Found 1 error(s) in 1 function(s) checked (0.03s)
```

Fix: replace `w.T` with `w` and annotate `w` as `"d_out d_in"`, or use `jnp.matmul(x, w)` with `w: Float[Array, "d_in d_out"]`.

## Editor Integration

### VS Code

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=beegass.jaxtyc) or search "jaxtyc" in the Extensions view.

Or build from source:

```bash
cd editors/vscode && npm install && npm run bundle
npx @vscode/vsce package --allow-missing-repository
code --install-extension jaxtyc-*.vsix
```

Or use the justfile: `just vscode-update`

The extension auto-discovers your Python environment (`.venv`, `VIRTUAL_ENV`, or `jaxtyc` on PATH) and starts the LSP server automatically. Supports multi-root workspaces with per-folder LSP clients. Includes jaxtyping snippets, a trace visualization webview, and a status bar quick pick menu.

### Other Editors

jaxtyc works in any editor that supports LSP (Neovim, Helix, etc.). See the [editor setup docs](https://beegass.github.io/jaxtyc/editors/editors/) for configuration.

## CLI

```
jaxtyc check <paths>...          # Shape-check files or directories
jaxtyc trace <file.py::func>     # Trace intermediate shapes through a function
jaxtyc watch <paths>...          # Watch and re-check on change
jaxtyc lsp                       # Start the LSP server (stdio)
jaxtyc mux                       # Start the LSP multiplexer (ty/pyright + jaxtyc)
jaxtyc version                   # Print version
```

## CI Integration

Use `--format github` to get inline annotations on pull request diffs:

```yaml
name: Shape Check
on: [push, pull_request]

jobs:
  jaxtyc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: astral-sh/setup-uv@v7
      - run: uv sync
      - run: uv run jaxtyc check src/ --format github
```

Each shape error appears as an annotation on the exact file and line in the PR. See the [CI docs](https://beegass.github.io/jaxtyc/guide/ci/) for JSON output, configuration, and full pipeline examples.

## Documentation

Full docs at [beegass.github.io/jaxtyc](https://beegass.github.io/jaxtyc/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
