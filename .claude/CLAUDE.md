# jaxtyc

Static array shape checker for JAX. Uses `jax.eval_shape` with symbolic dimensions (`jax.export.symbolic_shape`) to trace jaxtyping-annotated functions and detect shape mismatches at analysis time. Supports sharding-in-types via `|axis` annotation syntax.

## Commands

```bash
uv run jaxtyc check src/          # Shape-check all Python files
uv run jaxtyc check file.py       # Check a single file
uv run jaxtyc trace file.py::fn   # Trace a specific function
uv run jaxtyc watch src/          # Watch mode (re-analyze on save)
uv run jaxtyc lsp                 # Start LSP server on stdio
uv run jaxtyc version             # Print version

uv run pytest --tb=short -q       # Run tests
uv run ruff check . && uv run ruff format --check .  # Lint
uv run ty check                   # Type check
```

## Project Structure

```
src/jaxtyc/
  __init__.py              # Public API, version
  types.py                 # Frozen dataclasses: ShapeSpec, Diagnostic, TraceResult, etc.
  config.py                # JaxtycConfig from [tool.jaxtyc] in pyproject.toml

  analyzer/
    annotations.py         # AST parser: extract jaxtyping annotations, dim locations, call sites
    dim_env.py             # DimEnv: symbolic dims via jax.export.symbolic_shape + concrete fallback
    tracer.py              # jax.eval_shape + jax.make_jaxpr tracing (with sharded support)
    checker.py             # Shape comparison: expected vs actual
    divergence.py          # Divergence detection: find first shape deviation
    sharding_checker.py    # Sharding validation: rank, axis, conflict, io-mismatch, propagation, annotation
    mesh_resolver.py       # AST-based mesh shape and axis_rules inference
    importer.py            # Import Python files as modules (CPU guard, sys.modules cleanup)
    pipeline.py            # End-to-end: parse -> import -> trace -> check (abstract-first)
    suppressions.py        # Inline # jaxtyc: ignore comments

  cli/
    main.py                # CLI entry point; _enforce_cpu_backend() before JAX import
    formatters.py          # Output: full, concise, json, github

  lsp/
    server.py              # LanguageServer instance, _analyze_and_publish (content-hash gating), start_lsp
    _state.py              # Shared mutable caches (includes content_hash_cache)
    _util.py               # uri_to_path, dim_range, dim_label, shape_summary
    _navigation.py         # hover, CodeLens, symbols, definition, references, rename, call hierarchy
    _diagnostics.py        # didOpen, didSave, didChange, didClose, pull model
    _code_actions.py       # Quick fixes with shape suggestions
    _completion.py         # Dimension name + mesh axis autocomplete in shape strings
    _semantic_tokens.py    # Semantic highlighting for dim names
    _signature_help.py     # Shape signatures for function calls
    _inlay_hints.py        # Inline shape annotations
    _linked_editing.py     # Simultaneous dim name editing
    _folding.py            # Folding ranges for annotated functions
    _configuration.py      # Config hot-reload
    suggestions.py         # Shape fix generation (JAX-native + einops)
    index.py               # WorkspaceIndex: cross-file navigation
```

## Conventions

- All value types in `types.py` are `@dataclass(frozen=True)`
- DimEnv uses `jax.export.symbolic_shape` for symbolic tracing; `get_concrete_size`/`make_concrete_shape` for NNX/equinox modules that need plain ints
- LSP handlers are in separate modules; each imports `server` and uses `@server.feature()`
- Python 3.11+ (3.12 excluded due to JAX wheel gaps), JAX >= 0.9.0
- Inline suppression: `# jaxtyc: ignore` or `# jaxtyc: ignore[rule-name]`
- Sharding annotations use `|` syntax: `"batch|dp seq|None d_model|mp"` where `|axis` sets `DimSpec.mesh_axis`
- **CPU-only by default**: `_enforce_cpu_backend()` in `cli/main.py` sets `JAX_PLATFORMS=cpu` before any JAX import. Override with `JAXTYC_BACKEND=gpu` or `JAX_PLATFORMS` env var. VSCode extension and mux also set CPU env vars on spawned processes
- **Abstract-first model tracing**: NNX uses `nnx.eval_shape` for zero-allocation construction, equinox uses `eqx.filter_eval_shape`. Abstract state is passed as explicit `jax.eval_shape`/`jax.make_jaxpr` arguments (not closure-captured). Falls back to concrete construction on CPU if abstract fails
- **Content-hash gating**: LSP skips re-analysis when SHA-256 of source text matches the cached hash for that URI
- **Cache cleanup**: `jax.clear_caches()` called after each analysis cycle (LSP and CLI watch/check) to prevent unbounded cache growth
- **sys.modules hygiene**: `import_module_from_path()` removes `_jaxtyc_user_*` entries from `sys.modules` after loading; `exec_module` runs inside `jax.default_device(cpu)`

## Diagnostic Rules

| Rule | Meaning |
|------|---------|
| `shape-mismatch` | Dimensions differ at same rank |
| `rank-mismatch` | Different number of dimensions |
| `trace-error` | `jax.eval_shape` failed |
| `param-inconsistency` | Parameter annotation conflicts with resolved shape |
| `cross-function-mismatch` | Callee output shape contradicts annotation |
| `return-count-mismatch` | Tuple return element count differs |
| `resolve-error` | Could not find function in module |
| `import-error` | Module import failed |
| `file-not-found` | File does not exist |
| `read-error` | Could not read file |
| `sharding-rank-mismatch` | PartitionSpec length differs from array rank |
| `sharding-axis-unknown` | PartitionSpec references non-existent mesh axis |
| `sharding-conflict` | Conflicting PartitionSpecs on same shape at same line |
| `sharding-io-mismatch` | jit out_shardings contradict inner sharding constraint |
| `sharding-propagation-mismatch` | JAX-propagated output sharding differs from return annotation |
| `sharding-annotation-incomplete` | Piped shape with bare dims in strict mode |
| `sharding-dim-conflict` | Same dim name sharded on different axes across params |

## Adding a New Diagnostic Rule

1. Add the check logic in `checker.py` (or `pipeline.py` for file-level)
2. Use `Diagnostic(rule="your-rule-name", ...)` with a descriptive message
3. Optionally attach `DiagnosticData` for structured LSP data
4. Add tests in `tests/test_checker.py` or integration tests
5. Document in this table above
