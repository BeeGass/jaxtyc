# CLI

jaxtyc ships a single `jaxtyc` command with five subcommands. Install with `uv add jaxtyc` and the CLI is available immediately.

```
jaxtyc <command> [options]
```

---

## `jaxtyc check`

Analyze files or directories for shape errors.

```
jaxtyc check <paths>... [--format full|concise|json|github]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `<paths>` | One or more `.py` files or directories. Directories are recursed for `.py` files. |
| `--format` | Output format. Default: `full`. |

**Behavior:**

1. Collects all `.py` files from the given paths (recursive for directories).
2. Applies `exclude` glob patterns from `[tool.jaxtyc]` config.
3. For each file: parses the AST, extracts jaxtyping annotations, imports the module, traces with `jax.eval_shape`, and compares shapes.
4. Filters diagnostics by `severity` threshold and `ignore_rules` from config.
5. Exits `0` if no errors, `1` if any errors remain after filtering.

**Output formats:**

=== "full"

    Human-readable, multi-line output with rule codes and indented messages.

    ```
    $ jaxtyc check src/model.py
    src/model.py:42:4: error[shape-mismatch]
      Shape mismatch in return of `forward`
        Expected: (batch, seq, d_out)
        Got:      (batch, seq, d_model)

    Found 1 error(s) in 3 function(s) checked (0.84s)
    ```

=== "concise"

    One line per error. Good for editor integration or quick scans.

    ```
    $ jaxtyc check src/ --format concise
    src/model.py:42:4: error[shape-mismatch] Shape mismatch in return of `forward`
    1 error(s) (3 checked, 0.84s)
    ```

=== "json"

    Machine-readable JSON with all diagnostics, counts, and timing.

    ```
    $ jaxtyc check src/ --format json
    {
      "diagnostics": [
        {
          "file": "src/model.py",
          "line": 42,
          "col": 4,
          "severity": "error",
          "message": "Shape mismatch in return of `forward`\n  Expected: (batch, seq, d_out)\n  Got:      (batch, seq, d_model)",
          "rule": "shape-mismatch"
        }
      ],
      "functions_checked": 3,
      "elapsed_seconds": 0.84
    }
    ```

=== "github"

    GitHub Actions workflow command format. Produces inline PR annotations.

    ```
    $ jaxtyc check src/ --format github
    ::error file=src/model.py,line=42,col=4::Shape mismatch in return of `forward`
    ```

---

## `jaxtyc trace`

Trace intermediate shapes through a single function. Useful for debugging shape propagation without running the full model.

```
jaxtyc trace <file.py::function_name>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `<target>` | Path and function name separated by `::`. |

**Behavior:**

1. Parses the file's AST to find the target function's jaxtyping annotations.
2. Imports the module and resolves the function object (supports class methods via `class_name`).
3. Builds abstract inputs from annotated parameter shapes using the `DimEnv` prime-mapping system.
4. Runs `jax.make_jaxpr` to extract every intermediate operation and its output shape.
5. Maps each operation back to its source line and prints the trace with named dimensions.

**Example:**

```
$ jaxtyc trace src/model.py::attention
attention(q: float32[batch, heads, seq, d_head], k: float32[batch, heads, seq, d_head], v: float32[batch, heads, seq, d_head])

  Line 15: transpose -> (batch, heads, d_head, seq)  [float32]
  Line 16: dot_general -> (batch, heads, seq, seq)  [float32]
  Line 17: div -> (batch, heads, seq, seq)  [float32]
  Line 18: reduce_max -> (batch, heads, seq, 1)  [float32]
  Line 18: sub -> (batch, heads, seq, seq)  [float32]
  Line 18: exp -> (batch, heads, seq, seq)  [float32]
  Line 18: reduce_sum -> (batch, heads, seq, 1)  [float32]
  Line 18: div -> (batch, heads, seq, seq)  [float32]
  Line 19: dot_general -> (batch, heads, seq, d_head)  [float32]

  Output: (batch, heads, seq, d_head) [matches]
```

The `[matches]` / `[MISMATCH]` suffix compares the traced output against the return annotation.

---

## `jaxtyc watch`

Monitor directories and re-check on every `.py` file change. Requires the `watch` extra.

```
jaxtyc watch <paths>... [--format full|concise|json|github]
```

**Arguments:**

Same as `jaxtyc check`.

**Behavior:**

1. Runs an initial `check` on all collected files.
2. Watches parent directories using [watchfiles](https://watchfiles.helpmanual.io/) (inotify/kqueue/FSEvents).
3. On each `.py` change, re-analyzes only the changed files and prints results.

**Installation:**

```bash
uv add "jaxtyc[watch]"
```

**Example:**

```
$ jaxtyc watch src/
Watching 1 directory(ies) for changes...
All checks passed: 5 function(s) checked (1.23s)
# ... edit a file ...
src/model.py:42:4: error[shape-mismatch]
  Shape mismatch in return of `forward`
    Expected: (batch, seq, d_out)
    Got:      (batch, seq, d_model)

Found 1 error(s) in 1 function(s) checked (0.31s)
```

---

## `jaxtyc lsp`

Start the Language Server Protocol server over stdio. Intended to be launched by an editor, not run manually.

```
jaxtyc lsp
```

The LSP server provides:

- **Diagnostics** on open, save, and edit (debounced at 500ms by default, configurable via `debounce_ms`).
- **Hover** showing intermediate shapes at the cursor line.
- **CodeLens** displaying traced shape summaries above annotated functions.

See [Editors](../editors/editors.md) for editor-specific setup.

---

## `jaxtyc version`

Print the installed version and exit.

```
$ jaxtyc version
jaxtyc 0.1.0
```
