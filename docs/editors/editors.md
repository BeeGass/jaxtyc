# Editor Integration

jaxtyc ships with a built-in LSP server that provides real-time shape diagnostics in any editor that speaks the Language Server Protocol.

## LSP Capabilities

Start the server with:

```bash
jaxtyc lsp
```

The server communicates over stdio and provides:

- **Diagnostics** -- shape errors (`shape-mismatch`, `rank-mismatch`, `trace-error`, `cross-function-mismatch`, `param-inconsistency`, `return-count-mismatch`) appear as editor diagnostics on open, save, and change. Both push and pull diagnostic models are supported. Changes are debounced (default 500ms).
- **Hover** -- hover over any line in a traced function to see intermediate JAX operation shapes, or hover over a dimension name in a shape string to see its symbolic prime size, usage count, and all reference locations.
- **CodeLens** -- a virtual annotation above each jaxtyping-annotated function showing traced input/output shapes.
- **Go to Definition / Find References** -- navigate to dimension name definitions and find all references across the workspace, including cross-file.
- **Rename** -- rename a dimension name across all annotations in the workspace.
- **Code Actions** -- quick-fix suggestions for shape mismatches: transpose, expand_dims, squeeze, reshape. Supports JAX-native and einops notation (set `prefer_einops = true` in `[tool.jaxtyc]` or `JAXTYC_PREFER_EINOPS=1`).
- **Completion** -- autocomplete dimension names inside jaxtyping shape strings, drawn from all dims in the workspace.
- **Signature Help** -- shape signatures displayed when calling jaxtyping-annotated functions.
- **Semantic Tokens** -- dimension names in shape strings are highlighted, with definition vs. reference distinction.
- **Inlay Hints** -- inline resolved shapes at the end of lines with intermediate operations.
- **Linked Editing Range** -- simultaneously edit matching dimension names within the same function.
- **Folding Ranges** -- collapsible ranges for functions with many shape-annotated parameters.
- **Call Hierarchy** -- incoming/outgoing call graphs for shape-annotated functions, with shape details.
- **Document / Workspace Symbols** -- shape-annotated functions appear as symbols with shape summaries.
- **Configuration Hot-Reload** -- watches `pyproject.toml` for `[tool.jaxtyc]` changes and reloads without restart.

---

## Editor Setup

=== "VS Code"

    Install the jaxtyc extension from the `.vsix` file in `editors/vscode/`:

    ```bash
    cd editors/vscode && npm install && npm run bundle
    npx @vscode/vsce package --allow-missing-repository
    code --install-extension jaxtyc-*.vsix
    ```

    Or use the justfile from the project root: `just vscode-update`

    **Multi-root workspaces:** The extension starts a separate LSP client per workspace folder, each discovering its own Python environment independently.

    The extension auto-detects your Python environment in this order:

    1. `VIRTUAL_ENV` environment variable (activated venv with jaxtyc importable)
    2. `.venv/bin/python3` in any workspace folder or immediate subfolder (for worktree layouts)
    3. `jaxtyc` executable on PATH (installed via `uv tool install jaxtyc`)
    4. VS Code Python extension's `python.defaultInterpreterPath`
    5. `python3` on PATH

    Each candidate is validated before use -- the first one where jaxtyc is actually importable wins. Override with the `jaxtyc.pythonPath` setting if needed.

    **Configuration:**

    | Setting | Default | Description |
    |---------|---------|-------------|
    | `jaxtyc.mode` | `lsp` | `lsp` for shape checking only, `mux` to multiplex with ty/pyright |
    | `jaxtyc.pythonPath` | (auto) | Path to Python interpreter with jaxtyc installed |
    | `jaxtyc.args` | `[]` | Extra CLI arguments passed to the server |
    | `jaxtyc.hints.errorMode` | `both` | `both` shows shape and error, `replace` shows only error |
    | `jaxtyc.hints.errorLocation` | `divergence` | Where to place error hints |
    | `jaxtyc.hints.errorStyle` | `pipe` | Separator style between shape and error text |
    | `jaxtyc.sharding.display` | `append` | Sharding display mode |
    | `jaxtyc.sharding.rules` | all enabled | Allow-list of sharding diagnostic rules |

    **Commands:**

    | Command | Description |
    |---------|-------------|
    | `jaxtyc: Show Menu` | Open the status bar quick pick menu |
    | `jaxtyc: Restart Server` | Kill and respawn all LSP servers |
    | `jaxtyc: Check Current File` | Run `jaxtyc check` on the active file and show output |
    | `jaxtyc: Trace Function` | Trace a function and show shape flow in a webview panel |

    **Snippets:** Type a prefix in a Python file and press Tab:

    | Prefix | Expands to |
    |--------|-----------|
    | `jfloat` | `Float[Array, "batch seq dim"]` |
    | `jint` | `Int[Array, "batch seq"]` |
    | `jbool` | `Bool[Array, "batch seq"]` |
    | `jshaped` | `Shaped[Array, "*dims"]` |
    | `jimport` | `from jaxtyping import Array, Float, Int` |
    | `jignore` | `# jaxtyc: ignore[rule-name]` |

    **Status bar:** Shows mode and folder health. Click for quick pick menu, hover for server version.

    !!! tip "Mux mode"
        Setting `jaxtyc.mode` to `mux` starts the LSP multiplexer, which runs both a type checker (ty or pyright) and jaxtyc behind a single LSP connection. When using mux mode, disable Pylance to avoid duplicate diagnostics.

    !!! tip "Trace visualization"
        Run `jaxtyc: Trace Function` to trace a jaxtyping-annotated function. A webview panel opens showing the function signature, intermediate JAX operations with shapes/dtypes, and output match status.

=== "Neovim"

    Add a custom server config via `nvim-lspconfig`:

    ```lua
    local configs = require('lspconfig.configs')
    configs.jaxtyc = {
      default_config = {
        cmd = { 'jaxtyc', 'lsp' },
        filetypes = { 'python' },
        root_dir = require('lspconfig.util').root_pattern('pyproject.toml'),
      },
    }
    require('lspconfig').jaxtyc.setup({})
    ```

    This runs alongside pyright/pylsp. Diagnostics from both servers merge in the diagnostics list. Use `:LspInfo` to confirm jaxtyc is attached.

=== "Helix"

    Add to `~/.config/helix/languages.toml`:

    ```toml
    [[language]]
    name = "python"
    language-servers = ["pylsp", "jaxtyc"]

    [language-server.jaxtyc]
    command = "jaxtyc"
    args = ["lsp"]
    ```

    Restart Helix. Shape diagnostics will appear inline alongside pylsp diagnostics.

=== "Claude Code"

    jaxtyc can run as a Claude Code LSP plugin, surfacing shape diagnostics directly in the agent's context. Because Claude Code only starts **one LSP server per file extension**, jaxtyc ships a built-in **LSP multiplexer** that runs both a Python type checker (ty/pyright) and jaxtyc behind a single stdio pipe. Results from both servers are merged transparently.

    !!! info "Prerequisites"
        - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
        - [uv](https://docs.astral.sh/uv/) installed
        - jaxtyc installed (`uv pip install jaxtyc` or cloned locally)
        - A Python type checker: [ty](https://docs.astral.sh/ty/) (recommended) or [pyright](https://github.com/microsoft/pyright)
        - `ENABLE_LSP_TOOL` set to `"1"` in `~/.claude/settings.json`

    **Step 1: Enable the LSP tool**

    Add to `~/.claude/settings.json`:

    ```json
    {
      "env": {
        "ENABLE_LSP_TOOL": "1"
      }
    }
    ```

    **Step 2: Create the multiplexer launcher**

    Create `~/.local/bin/python-lsp-mux`:

    ```python
    #!/usr/bin/env python3
    """Shim: delegates to jaxtyc's built-in LSP multiplexer."""
    import asyncio
    from jaxtyc.lsp.mux import run_mux

    if __name__ == "__main__":
        asyncio.run(run_mux())
    ```

    Make it executable:

    ```bash
    chmod +x ~/.local/bin/python-lsp-mux
    ```

    The multiplexer auto-discovers the best available type checker in order: `pyright-langserver` (ty shim), `ty`, `pyright`. No configuration needed.

    !!! tip "If jaxtyc is not on PATH"
        If jaxtyc is installed in a project-local venv, the shim needs to find it. Either install jaxtyc globally (`uv tool install jaxtyc`) or point the shim at the project:

        ```python
        #!/usr/bin/env python3
        import asyncio, subprocess, sys
        subprocess.check_call(
            [sys.executable, "-c",
             "from jaxtyc.lsp.mux import run_mux; import asyncio; asyncio.run(run_mux())"],
            env={**__import__("os").environ,
                 "VIRTUAL_ENV": "/path/to/your/jaxtyc/.venv"})
        ```

    **Step 3: Create the plugin**

    Set up a plugin directory with a manifest:

    ```
    your-plugins/
      lsp-servers/
        python-lsp/
          .claude-plugin/
            plugin.json
    ```

    `plugin.json`:

    ```json
    {
      "name": "python-lsp",
      "description": "Python LSP: type checking + array shape checking via multiplexer",
      "version": "1.0.0",
      "author": {
        "name": "Your Name"
      },
      "lspServers": {
        "python-mux": {
          "command": "python-lsp-mux",
          "extensionToLanguage": {
            ".py": "python",
            ".pyi": "python"
          }
        }
      }
    }
    ```

    Install the plugin:

    ```bash
    claude plugin install python-lsp@your-marketplace
    ```

    Or place the plugin directory under `~/.claude/plugins/cache/your-marketplace/python-lsp/1.0.0/` manually.

    **Step 4: Restart Claude Code**

    After restart, the multiplexer activates when Claude Code opens a `.py` file. Three processes run behind the single stdio pipe:

    ```
    python-lsp-mux          # multiplexer (merges results)
      ├── ty server          # type checking (or pyright --stdio)
      └── jaxtyc lsp         # shape checking
    ```

    **How merging works:**

    | Response type | Strategy | Example |
    |---|---|---|
    | Diagnostics | Concatenate from both | ty: `unresolved-import`, jaxtyc: `shape-mismatch` |
    | Hover | Combine markdown | ty: type signature, jaxtyc: dimension info |
    | CodeLens, codeAction | Concatenate arrays | Both servers' results shown |
    | References | Concatenate arrays | Cross-file dim refs + Python refs |
    | Completion | Merge item lists | Dim name completions + Python completions |
    | Definition, rename | First non-null | Python symbols via ty, dim names via jaxtyc |

    !!! warning "Notes"
        - The jaxtyc venv must include all dependencies imported by analyzed files (jax, jaxtyping, flax, etc.).
        - A 3-second timeout ensures the client always gets a response even if one server is slow.

    ---

    ### What it looks like in practice

    Below is a real Claude Code conversation showing the multiplexer in action. The user opens a file with a shape bug, and both ty and jaxtyc diagnostics appear automatically. Then the LSP's hover, references, and call hierarchy features are exercised.

    **Diagnostics appear on file open:**

    ```
    > LSP documentSymbol tests/fixtures/cross_function_mismatch.py

    Document symbols:
    encode (Function) - Line 7
    decode (Function) - Line 15
    pipeline (Function) - Line 21

    Diagnostics:
      ✘ [Line 7:1] Shape mismatch in return of `encode`
        Expected: (batch, seq, hidden)
        Got:      (batch, seq, d_model) [shape-mismatch] (jaxtyc)
      ✘ [Line 15:1] Shape mismatch in return of `decode`
        Expected: (batch, seq, d_model)
        Got:      (batch, seq, hidden) [shape-mismatch] (jaxtyc)
      ✘ [Line 24:9] Cross-function shape mismatch: `encode` called
        from `pipeline`
        Annotated return: (batch, seq, hidden)
        Actual return:    (batch, seq, d_model)
        [cross-function-mismatch] (jaxtyc)
      ✘ [Line 25:12] Cross-function shape mismatch: `decode` called
        from `pipeline`
        Annotated return: (batch, seq, d_model)
        Actual return:    (batch, seq, hidden)
        [cross-function-mismatch] (jaxtyc)
    ```

    jaxtyc detected that `encode` claims to return `(batch, seq, hidden)` but actually returns `(batch, seq, d_model)`, and that `pipeline` is calling it with that wrong annotation. Four diagnostics total, all with structured shape information.

    **Hover on a Python symbol (served by ty):**

    ```
    > LSP hover tests/fixtures/wrong_transpose.py:14:20

    def matmul(
        a: Array | ndarray[...] | ...,
        b: Array | ndarray[...] | ...,
        *,
        precision: None | str | Precision | ... = ...,
        preferred_element_type: ... = ...,
    ) -> Array
    ```

    ty provides the full type signature for `jnp.matmul`.

    **Hover on a dimension name (served by jaxtyc):**

    ```
    > LSP hover tests/fixtures/wrong_transpose.py:9:45

    **`head_dim`** -- dimension name

    Symbolic size: `7` (prime)

    **Used 2 time(s) in this file:**

    - `attention` / `q` (line 9)
    - `attention` / `k` (line 10)
    ```

    ty returns null for this position (it's inside a string literal), so jaxtyc's dimension name information is used instead. Shows the prime-based symbolic size and all usages in the file.

    **Cross-file dimension references:**

    ```
    > LSP findReferences cross_function_mismatch.py:8:30

    Found 10 references across 2 files:

    tests/fixtures/wrong_transpose.py:
      Line 9:34
      Line 10:34
      Line 11:32
      Line 11:36

    tests/fixtures/cross_function_mismatch.py:
      Line 8:28
      Line 9:26
      Line 16:28
      Line 17:26
      Line 22:28
      Line 23:26
    ```

    The dimension name `seq` is tracked across all files in the workspace.

    **Call hierarchy with shape info:**

    ```
    > LSP prepareCallHierarchy cross_function_mismatch.py:7:5

    Call hierarchy item: encode (Function)
      tests/fixtures/cross_function_mismatch.py:7
      [x: (batch, seq, d_model) -> (batch, seq, hidden)]

    > LSP incomingCalls cross_function_mismatch.py:7:5

    Found 1 incoming call:
      pipeline (Function) - Line 21 [calls at: 24:9]
    ```

    Shape annotations are included in the call hierarchy detail, making it easy to trace shape flow through function boundaries.

---

## Troubleshooting

!!! note "Common issues"
    - **No diagnostics appear**: Ensure the file contains jaxtyping annotations (`Float[Array, "..."]`). Files without them are silently skipped.
    - **VS Code: "jaxtyc not found"**: The extension couldn't find a Python environment with jaxtyc installed. Either install it globally (`uv tool install jaxtyc`), add it to your project's venv (`uv add --dev jaxtyc`), or set `jaxtyc.pythonPath` explicitly.
    - **VS Code: diagnostics missing in multi-root workspace**: The extension checks `.venv` in each workspace folder and one level of subdirectories. If your venv is deeper, set `jaxtyc.pythonPath` per workspace.
    - **Import errors in diagnostics**: The jaxtyc process must have access to all imports used by the analyzed file. Install dependencies in the same venv or use `uv run --project`.
    - **Stale diagnostics after refactor**: Save the file to trigger a fresh analysis. Debounced change analysis uses a temp file and may lag behind large refactors.
    - **Claude Code: only one server's diagnostics show**: Ensure you are using the multiplexer (`python-lsp-mux`), not separate plugins for ty and jaxtyc. Claude Code only starts one LSP server per file extension.
    - **Claude Code: multiplexer not starting**: Run `ps aux | grep python-lsp-mux` to check. The multiplexer starts lazily on first `.py` file access. If it's not running, verify the shim is executable and on PATH.
    - **Claude Code: "No Python type checker found"**: The multiplexer needs ty or pyright. Install with `uv tool install ty` or `npm install -g pyright`.
