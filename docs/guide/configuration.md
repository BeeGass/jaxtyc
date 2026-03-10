# Configuration

jaxtyc is configured through the standard `[tool.jaxtyc]` section in your project's `pyproject.toml`. No separate config file needed.

If no `pyproject.toml` exists or the `[tool.jaxtyc]` section is absent, all defaults apply and jaxtyc runs without configuration.

---

## Config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `severity` | `"error"` \| `"warning"` \| `"info"` | `"error"` | Minimum severity threshold. Diagnostics below this level are suppressed. |
| `ignore_rules` | `list[str]` | `[]` | Rule codes to suppress regardless of severity. |
| `exclude` | `list[str]` | `[]` | Glob patterns for files to skip during analysis. Matched with `fnmatch`. |
| `debounce_ms` | `int` | `500` | Delay in milliseconds before the LSP server re-analyzes after an edit. |
| `prefer_einops` | `bool` | `false` | When `true`, einops-style suggestions appear first in code actions. Overridable with `JAXTYC_PREFER_EINOPS=1`. Requires the `einops` extra. |
| `hover_compact` | `bool` | `true` | When `true`, compacts hover text in multiplexer mode (strips escape sequences, collapses blank lines, truncates at 1500 chars). |

---

## Full example

```toml
[tool.jaxtyc]
severity = "warning"
ignore_rules = ["trace-error"]
exclude = ["tests/**", "examples/**", "benchmarks/**"]
debounce_ms = 300
prefer_einops = true
```

---

## Severity filtering

Diagnostics have three severity levels, ranked:

| Level | Numeric rank | Typical use |
|-------|-------------|-------------|
| `error` | 3 | Shape mismatches, rank mismatches |
| `warning` | 2 | Potential issues |
| `info` | 1 | Import failures, file-not-found, resolution failures |

Setting `severity = "warning"` reports warnings and errors, suppressing info-level diagnostics. Setting `severity = "info"` reports everything.

---

## Rule codes

These are the rule codes emitted by jaxtyc, usable in `ignore_rules`:

| Rule | Severity | Description |
|------|----------|-------------|
| `shape-mismatch` | error | Traced output shape does not match the annotated return shape. |
| `rank-mismatch` | error | Traced output rank (number of dims) differs from annotation. |
| `trace-error` | error | `jax.eval_shape` raised an exception during tracing. |
| `param-inconsistency` | error | Parameter annotation conflicts with the resolved input shape. |
| `cross-function-mismatch` | error | Callee output shape contradicts its annotation at a call site. |
| `return-count-mismatch` | error | Tuple return element count differs from annotation. |
| `file-not-found` | info | The specified file path does not exist. |
| `read-error` | info | The file could not be read (permissions, encoding). |
| `import-error` | info | The module could not be imported for tracing. |
| `resolve-error` | info | A function found in the AST could not be resolved to a live object. |

---

## Filtering logic

A diagnostic is included in output when **both** conditions are met:

1. Its severity is at or above the configured `severity` threshold.
2. Its rule code is **not** in `ignore_rules`.

```toml
[tool.jaxtyc]
severity = "warning"
ignore_rules = ["trace-error"]
```

With this config:

- `shape-mismatch` (error): included -- above threshold, not ignored.
- `trace-error` (error): excluded -- above threshold, but explicitly ignored.
- `import-error` (info): excluded -- below threshold.

---

## Exclude patterns

The `exclude` list uses Python's `fnmatch` for pattern matching against file paths. Patterns are checked against the full path string collected during file discovery.

```toml
[tool.jaxtyc]
exclude = [
    "tests/**",           # skip all test files
    "**/conftest.py",     # skip conftest everywhere
    "scripts/*.py",       # skip top-level scripts
]
```

!!! tip "Exclude applies to CLI and LSP"
    The `exclude` patterns filter files in `jaxtyc check` and `jaxtyc watch`. The LSP server analyzes whatever files the editor opens, so `exclude` does not apply there.
