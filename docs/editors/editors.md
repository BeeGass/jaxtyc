# Editor Integration

jaxtyc ships with a built-in LSP server that provides real-time shape diagnostics in any editor that speaks the Language Server Protocol.

## LSP Capabilities

Start the server with:

```bash
jaxtyc lsp
```

The server communicates over stdio and provides:

- **Diagnostics on open/save/change** -- shape errors appear as editor diagnostics with rule codes (`shape-mismatch`, `rank-mismatch`, `trace-error`). Changes are debounced (default 500ms) to avoid thrashing during rapid edits.
- **Hover for intermediate shapes** -- hover over any line in a traced function to see the shapes produced by each JAX operation at that line, with named dimensions resolved back from primes.
- **CodeLens with shape annotations** -- a virtual annotation appears above each jaxtyping-annotated function showing its traced input/output shapes.

---

## Editor Setup

=== "VS Code"

    Use any generic LSP client extension (e.g., [glslang](https://marketplace.visualstudio.com/items?itemName=AntHillPlan.vscode-glslang) or [vscode-lsp-sample](https://github.com/nicolo-ribaudo/tc39-proposal-lsp)). Add to `.vscode/settings.json`:

    ```json
    {
      "generic-lsp.servers": [
        {
          "name": "jaxtyc",
          "command": "jaxtyc",
          "args": ["lsp"],
          "languages": ["python"]
        }
      ]
    }
    ```

    Alternatively, if jaxtyc is installed in a project-local venv managed by uv:

    ```json
    {
      "generic-lsp.servers": [
        {
          "name": "jaxtyc",
          "command": "uv",
          "args": ["run", "jaxtyc", "lsp"],
          "languages": ["python"]
        }
      ]
    }
    ```

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

    jaxtyc can run as a Claude Code LSP plugin, surfacing shape diagnostics directly in the agent's context.

    !!! info "Prerequisites"
        - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
        - [uv](https://docs.astral.sh/uv/) installed
        - jaxtyc installed or cloned locally
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

    **Step 2: Create a launcher script**

    Create `~/.local/bin/jaxtyc-langserver`:

    ```sh
    #!/bin/sh
    exec uv run --project /path/to/jaxtyc jaxtyc lsp
    ```

    Replace `/path/to/jaxtyc` with the absolute path to your jaxtyc checkout. If jaxtyc is installed in your project's venv, simplify to:

    ```sh
    #!/bin/sh
    exec uv run jaxtyc lsp
    ```

    Make it executable:

    ```bash
    chmod +x ~/.local/bin/jaxtyc-langserver
    ```

    **Step 3: Create the plugin**

    Set up a plugin directory with a manifest:

    ```
    your-plugins/
      lsp-servers/
        jaxtyc-lsp/
          .claude-plugin/
            plugin.json
    ```

    `plugin.json`:

    ```json
    {
      "name": "jaxtyc-lsp",
      "description": "JAX array shape checker for jaxtyping-annotated Python code",
      "version": "0.1.0",
      "author": {
        "name": "Your Name"
      },
      "lspServers": {
        "jaxtyc": {
          "command": "jaxtyc-langserver",
          "extensionToLanguage": {
            ".py": "python"
          }
        }
      }
    }
    ```

    If using a [Claude Code plugin marketplace](https://docs.anthropic.com/en/docs/claude-code/plugins):

    ```bash
    claude plugin install jaxtyc-lsp@your-marketplace
    ```

    Otherwise, place the plugin directory under `~/.claude/plugins/cache/your-marketplace/jaxtyc-lsp/0.1.0/` manually.

    **Step 4: Restart Claude Code**

    After restart, the LSP activates when Claude Code opens or saves a `.py` file. Diagnostics (`rank-mismatch`, `shape-mismatch`, `trace-error`) appear automatically. Hover over traced lines to see intermediate shapes with named dimensions.

    !!! warning "Limitations"
        - If two LSP plugins claim `.py` files, Claude Code may route to only one per session.
        - The jaxtyc venv must include all dependencies imported by analyzed files (jax, jaxtyping, flax, etc.).

---

## Troubleshooting

!!! note "Common issues"
    - **No diagnostics appear**: Ensure the file contains jaxtyping annotations (`Float[Array, "..."]`). Files without them are silently skipped.
    - **Import errors in diagnostics**: The jaxtyc process must have access to all imports used by the analyzed file. Install dependencies in the same venv or use `uv run --project`.
    - **Stale diagnostics after refactor**: Save the file to trigger a fresh analysis. Debounced change analysis uses a temp file and may lag behind large refactors.
