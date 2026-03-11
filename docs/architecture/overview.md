# Architecture

## Analysis Pipeline

```mermaid
flowchart LR
    A[Source .py] --> B[AST Parse]
    B --> C[Extract Annotations]
    C --> D[Import Module]
    D --> E[DimEnv: Assign Primes]
    E --> F[jax.eval_shape]
    F --> G[Shape Checker]
    G --> G2[Sharding Checker]
    G2 --> H[Cross-Function Check]
    H --> I[Inline Suppressions]
    I --> J[Diagnostics]
    E --> K[jax.make_jaxpr]
    K --> L[Source Mapping]
    L --> L2[Divergence Detection]
    L2 --> M[Hover/CodeLens/Inlay Hints]
```

The pipeline has two output paths from `DimEnv`. The primary path (`eval_shape` -> shape checker -> sharding checker -> cross-function check -> suppressions -> diagnostics) powers the `check` command and LSP error reporting. The secondary path (`make_jaxpr` -> source mapping -> divergence detection) powers the `trace` command, LSP hover, CodeLens, and inlay hints with error annotations.

---

## Module Responsibilities

| Module | File | Responsibility |
|--------|------|---------------|
| annotations | `analyzer/annotations.py` | AST-based jaxtyping annotation parser. Extracts `FunctionShapeSpec` from `Float[Array, "batch seq d_model"]`-style type hints. Also extracts `DimLocation` for cross-file navigation and `CallSite` for cross-function analysis. |
| dim_env | `analyzer/dim_env.py` | Prime sieve dimension environment. Maps each named dimension to a unique prime (>= 101) for unambiguous symbolic tracing. Shared per file. |
| importer | `analyzer/importer.py` | Dynamic module import with venv auto-discovery. Finds virtual environments (following ty's resolution order), adds `site-packages` to `sys.path`, and loads user `.py` files via `importlib`. |
| tracer | `analyzer/tracer.py` | `jax.eval_shape` / `jax.make_jaxpr` wrappers. Builds `ShapeDtypeStruct` inputs from specs, runs tracing, extracts output shapes and intermediates. Extracts `ShardingInfo` from `sharding_constraint` and `shard_map` jaxpr primitives. Handles single and multi-output (PyTree) returns. |
| source_map | `analyzer/source_map.py` | Jaxpr `source_info` extraction. Walks jaxpr equations to find user-code frames, filtering out JAX-internal stack frames. |
| checker | `analyzer/checker.py` | Shape comparison: expected vs actual. Emits `shape-mismatch`, `rank-mismatch`, `trace-error`, `param-inconsistency`, `cross-function-mismatch`, and `return-count-mismatch` diagnostics. Attaches `DiagnosticData` with structured shape info and suggested fixes. |
| divergence | `analyzer/divergence.py` | Divergence detection. Given a function's expected return shape and traced intermediates, finds the first intermediate whose shape deviates from expected. Returns `ErrorHintInfo` for inline error display in inlay hints. |
| sharding_checker | `analyzer/sharding_checker.py` | Sharding validation. Checks `PartitionSpec` consistency against array ranks and mesh axes. Implements 4 rules: `sharding-rank-mismatch`, `sharding-axis-unknown`, `sharding-conflict`, `sharding-io-mismatch`. |
| suppressions | `analyzer/suppressions.py` | Inline suppression comment parsing. Extracts `# jaxtyc: ignore` and `# jaxtyc: ignore[rule-name]` comments and filters matching diagnostics. |
| pipeline | `analyzer/pipeline.py` | End-to-end orchestration. Sequences parse -> import -> trace -> check -> sharding check -> cross-function -> suppress for a single file. Entry point: `analyze_file()`. Handles Flax NNX and Equinox module tracing with dimension-aware constructor kwargs. |
| config | `config.py` | `[tool.jaxtyc]` config loading from `pyproject.toml`. Supports severity filtering, rule ignoring, file exclusion, einops preferences, hover compaction, nested `[tool.jaxtyc.hints]` and `[tool.jaxtyc.sharding]` subsections. |
| formatters | `cli/formatters.py` | Output formatters: `full` (human-readable), `concise` (one-liner), `json` (machine-readable), `github` (Actions annotations). |
| server | `lsp/server.py` | pygls-based LSP server core. Runs `_analyze_and_publish` with caching, debouncing, progress reporting, and divergence detection. Imports 10 handler sub-modules that register 29 LSP method handlers. Caches error hints and source text for inlay hint rendering. |
| mux | `lsp/mux.py` | LSP multiplexer. Runs ty/pyright + jaxtyc behind a single stdio pipe with dual-send request merging, deferred server spawning, and diagnostic aggregation. |
| suggestions | `lsp/suggestions.py` | Shape fix generation. Produces JAX-native (`jnp.transpose`, `jnp.expand_dims`, `jnp.squeeze`, `jnp.reshape`) and einops-style suggestions for code actions. |
| index | `lsp/index.py` | `WorkspaceIndex` for cross-file navigation. Thread-safe index of functions, dimension locations, and call sites across all open files. |

---

## Data Flow

1. **Parse**: `extract_function_specs(source, path)` walks the AST and returns a `list[FunctionShapeSpec]` -- one per function that has at least one jaxtyping annotation. Also extracts `DimLocation` and `CallSite` data for navigation and cross-function analysis.

2. **Import**: `import_module_from_path(path)` discovers the project's virtual environment (following ty's resolution order: `VIRTUAL_ENV` -> `.venv` at project root -> walk-up), adds `site-packages` to `sys.path`, and loads the user module so JAX can trace its functions.

3. **Dimension assignment**: A shared `DimEnv` is created per file with reserved literal sizes. Each named dimension (e.g., `batch`, `seq`, `d_model`) gets a unique prime (>= 101). Fixed dimensions (literal integers in annotations) keep their original values. Sharing the `DimEnv` across all functions in a file ensures the same dimension name always maps to the same prime, enabling cross-function consistency checking.

4. **Tracing**: `jax.eval_shape(fn, **abstract_inputs)` propagates shapes through the function without running computation. The abstract inputs are `ShapeDtypeStruct` objects built from the prime-assigned shapes. For Flax NNX modules, a concrete model is constructed with prime-based dimension kwargs via `_collect_dim_kwargs`, split via `nnx.split/merge`, and traced with `jax.eval_shape`. For Equinox modules, `jax.eval_shape` traces bound methods similarly. During tracing, `_extract_intermediates` also detects sharding primitives (`sharding_constraint`, `shard_map`) and attaches `ShardingInfo` to each intermediate shape.

5. **Checking**: The checker compares the traced output shape against the return annotation's expected shape (also built from primes via the same `DimEnv`). Parameter shapes are also verified against traced inputs. Mismatches produce diagnostics with `DiagnosticData` carrying expected/actual shapes, dimension name mappings, and suggested fixes.

6. **Cross-function propagation**: `extract_call_sites()` finds calls between annotated functions. For each call site, `check_call_site()` verifies that the callee's traced output matches its annotation, emitting `cross-function-mismatch` if not.

7. **Inline suppressions**: `extract_suppressions()` parses `# jaxtyc: ignore` comments, and `filter_inline_suppressions()` removes matching diagnostics.

8. **Sharding validation**: After shape checking, `check_sharding()` validates sharding constraints found on intermediates. It checks partition spec rank against array rank, verifies mesh axis names exist, detects conflicting specs, and flags jit/constraint mismatches.

9. **Source mapping** (optional): `jax.make_jaxpr` is run separately to extract per-equation source locations. Each equation's `source_info.traceback.frames` is filtered to find the user's source line, skipping JAX internals.

10. **Divergence detection** (LSP only): `find_divergence_points()` compares traced intermediates against the expected return shape, finding the first operation whose output deviates. The result (`ErrorHintInfo`) is cached and displayed as an inline error annotation in inlay hints.

---

## Design Principles

**Zero runtime cost.** jaxtyc is a static analysis tool. It uses `jax.eval_shape` which performs no actual computation -- only shape propagation through abstract values. Analyzed code is never executed with real data.

**Unambiguous shapes via primes.** Every named dimension gets a unique prime number (>= 101). Since primes are coprime by definition, no combination of operations (transpose, reshape, matmul) can accidentally produce a shape that matches the expected shape when the actual computation is wrong. See [Internals](internals.md#prime-sieve-dimensions) for details.

**Graceful degradation.** Files without jaxtyping annotations are silently skipped (zero diagnostics, zero errors). Import failures, resolve failures, and trace failures produce info-level diagnostics rather than crashing. The tool always produces a `FileResult`, even on failure.

**Composable with existing tooling.** jaxtyc runs alongside pyright, ty, pylsp, ruff, or any other Python tool. The LSP server publishes diagnostics under the `jaxtyc` source name so they are distinguishable. The `jaxtyc mux` multiplexer merges jaxtyc with a type checker behind a single stdio pipe for editors that only support one server per language. The CLI supports `--format github` for CI integration.
