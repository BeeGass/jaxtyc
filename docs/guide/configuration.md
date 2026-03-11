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

## Hints configuration

The `[tool.jaxtyc.hints]` section controls inlay hint display in the LSP server.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `error_mode` | `"both"` \| `"replace"` | `"both"` | `both` shows shape and error side by side; `replace` shows only the error when a divergence is detected |
| `error_location` | `"divergence"` \| `"annotation"` \| `"return"` \| `"both"` | `"divergence"` | Where to place error hints: at the divergence point, at the annotation, at the return, or both divergence and annotation |
| `error_style` | `"pipe"` \| `"icon"` | `"pipe"` | Separator between shape and error text: `pipe` uses ` | `, `icon` uses a warning triangle |
| `dtype_style` | `"numpy"` \| `"jax"` \| `"jaxtyping"` | `"numpy"` | Dtype display format: `numpy` abbreviates (`f32`, `bf16`), `jax` passes through (`float32`), `jaxtyping` capitalizes (`Float32`, `BFloat16`) |

```toml
[tool.jaxtyc.hints]
error_mode = "replace"
dtype_style = "jaxtyping"
```

---

## Sharding configuration

The `[tool.jaxtyc.sharding]` section controls sharding display and validation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `display` | `"append"` \| `"constrained_only"` \| `"off"` | `"append"` | Sharding display in inlay hints: `append` always shows, `constrained_only` shows only for sharding_constraint primitives, `off` hides sharding info |
| `rules` | `list[str]` | all 4 rules | Allow-list of enabled sharding diagnostic rules. Only listed rules produce diagnostics. |

The default `rules` list includes all four sharding rules: `sharding-rank-mismatch`, `sharding-axis-unknown`, `sharding-conflict`, `sharding-io-mismatch`. To disable a specific rule, list only the rules you want:

```toml
[tool.jaxtyc.sharding]
display = "constrained_only"
rules = ["sharding-rank-mismatch", "sharding-axis-unknown"]
```

---

## Full example

```toml
[tool.jaxtyc]
severity = "warning"
ignore_rules = ["trace-error"]
exclude = ["tests/**", "examples/**", "benchmarks/**"]
debounce_ms = 300
prefer_einops = true

[tool.jaxtyc.hints]
error_mode = "both"
dtype_style = "numpy"

[tool.jaxtyc.sharding]
display = "append"
rules = ["sharding-rank-mismatch", "sharding-axis-unknown", "sharding-conflict", "sharding-io-mismatch"]
```

---

## Severity filtering

Diagnostics have three severity levels, ranked:

| Level | Numeric rank | Typical use |
|-------|-------------|-------------|
| `error` | 3 | Shape mismatches, rank mismatches, sharding errors |
| `warning` | 2 | Potential issues (e.g., sharding I/O mismatch) |
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
| `sharding-rank-mismatch` | error | PartitionSpec length differs from array rank. |
| `sharding-axis-unknown` | error | PartitionSpec references a non-existent mesh axis. |
| `sharding-conflict` | error | Conflicting PartitionSpecs on same shape at same line. |
| `sharding-io-mismatch` | warning | jit out_shardings contradict an inner sharding_constraint. |
| `file-not-found` | info | The specified file path does not exist. |
| `read-error` | info | The file could not be read (permissions, encoding). |
| `import-error` | info | The module could not be imported for tracing. |
| `resolve-error` | info | A function found in the AST could not be resolved to a live object. |

---

## Filtering logic

A diagnostic is included in output when **all** conditions are met:

1. Its severity is at or above the configured `severity` threshold.
2. Its rule code is **not** in `ignore_rules`.
3. For sharding diagnostics: its rule code is in the `[tool.jaxtyc.sharding] rules` allow-list (defaults to all 4 sharding rules).

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
