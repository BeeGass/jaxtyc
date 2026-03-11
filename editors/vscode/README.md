# jaxtyc for VS Code

Static array shape checking for JAX. Catches shape mismatches in [jaxtyping](https://github.com/patrick-kidger/jaxtyping)-annotated functions at analysis time using `jax.eval_shape`.

## Features

- **Shape diagnostics** -- underlines shape mismatches, rank errors, and trace failures inline
- **Hover** -- shows resolved shapes on hover over annotated parameters and variables
- **CodeLens** -- displays shape summaries above annotated functions
- **Code actions** -- quick fixes with corrected shape suggestions (JAX-native and einops)
- **Navigation** -- go-to-definition, find references, and rename for dimension names
- **Call hierarchy** -- trace shape flow across function calls
- **Inlay hints** -- inline shape annotations next to variables
- **Semantic highlighting** -- colored dimension names in shape strings
- **Signature help** -- shape signatures when calling annotated functions
- **Completion** -- autocomplete dimension names inside shape annotations
- **Trace visualization** -- run `jaxtyc trace` on a function and view results in a webview panel with shape flow table
- **Multi-root workspaces** -- per-folder LSP clients, each discovering its own Python environment
- **Snippets** -- jaxtyping annotation snippets (`jfloat`, `jint`, `jbool`, `jshaped`, `jimport`, `jignore`)
- **Problem matcher** -- `jaxtyc check` output parsed into the Problems panel

All features come from the jaxtyc LSP server. The extension spawns it and connects over stdio.

## Requirements

jaxtyc must be installed in a Python environment accessible to VS Code:

```bash
uv tool install jaxtyc
# or
pip install jaxtyc
# or in a project venv
uv add jaxtyc
```

The extension auto-detects jaxtyc in this order:

1. `VIRTUAL_ENV` environment variable (activated venv with jaxtyc importable)
2. `.venv/bin/python3` in any workspace folder or immediate subfolder with `pyproject.toml` (handles worktree layouts)
3. `jaxtyc` executable on PATH (installed via `uv tool install` or `pipx`)
4. VS Code Python extension's `python.defaultInterpreterPath`
5. `python3` on PATH

Each candidate is validated before use -- the first one where jaxtyc is actually importable wins.

In multi-root workspaces, each folder discovers its own Python environment independently.

Override with the `jaxtyc.pythonPath` setting if needed.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `jaxtyc.mode` | `lsp` | `lsp` for shape checking only, `mux` to multiplex with ty/pyright |
| `jaxtyc.pythonPath` | (auto) | Path to Python interpreter with jaxtyc installed |
| `jaxtyc.args` | `[]` | Extra CLI arguments passed to the server |
| `jaxtyc.hints.errorMode` | `both` | `both` shows shape and error, `replace` shows only error |
| `jaxtyc.hints.errorLocation` | `divergence` | Where to place error hints: `divergence`, `annotation`, `return`, `both` |
| `jaxtyc.hints.errorStyle` | `pipe` | Separator style: `pipe` uses ` \| `, `icon` uses warning triangle |
| `jaxtyc.hints.dtypeStyle` | `numpy` | Dtype display: `numpy` (f32, bf16), `jax` (float32, bfloat16), `jaxtyping` (Float32, BFloat16) |
| `jaxtyc.sharding.display` | `append` | Sharding display: `append`, `constrained_only`, `off` |
| `jaxtyc.sharding.rules` | all enabled | Allow-list of sharding diagnostic rules |

### Mux mode

Setting `jaxtyc.mode` to `mux` starts the jaxtyc multiplexer, which runs both a type checker (ty or pyright) and jaxtyc behind a single LSP connection. Responses are merged so you get type errors and shape errors in one pass.

When using mux mode, disable Pylance or other Python language servers to avoid duplicate diagnostics.

## Commands

| Command | Description |
|---------|-------------|
| `jaxtyc: Show Menu` | Open the status bar quick pick menu |
| `jaxtyc: Restart Server` | Kill and respawn all LSP servers |
| `jaxtyc: Check Current File` | Run `jaxtyc check` on the active file and show output |
| `jaxtyc: Trace Function` | Trace a function and show shape flow in a webview panel |

### Status bar

The status bar shows the current mode and folder health:

- `jaxtyc [lsp]` -- single folder, running
- `jaxtyc [mux] (3 folders)` -- multi-root, all running
- `jaxtyc [lsp] (2/3)` -- multi-root, some folders have errors
- `jaxtyc (not found)` -- no folders have jaxtyc available

Click the status bar item to open the quick pick menu. Hover to see the server version.

### Trace visualization

Run `jaxtyc: Trace Function` (or select it from the quick pick menu) to trace a function:

1. Open a Python file with jaxtyping-annotated functions
2. Run the command and enter the function name
3. A webview panel opens beside your editor showing:
   - The function signature with shape annotations
   - A table of intermediate JAX operations with shapes and dtypes
   - The output shape with match/mismatch status

## Snippets

Type a prefix in a Python file and press Tab:

| Prefix | Expands to |
|--------|-----------|
| `jfloat` | `Float[Array, "batch seq dim"]` |
| `jint` | `Int[Array, "batch seq"]` |
| `jbool` | `Bool[Array, "batch seq"]` |
| `jshaped` | `Shaped[Array, "*dims"]` |
| `jimport` | `from jaxtyping import Array, Float, Int` |
| `jignore` | `# jaxtyc: ignore[rule-name]` |

## Multi-root Workspaces

In multi-root workspaces, the extension starts a separate LSP client for each workspace folder. Each folder:

- Discovers its own Python environment independently
- Gets scoped diagnostics (only files under that folder)
- Can be restarted individually via `jaxtyc: Restart Server` (restarts all)

The status bar reflects aggregate health across all folders.
