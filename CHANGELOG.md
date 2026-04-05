# Changelog

All notable changes to jaxtyc are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub issue templates (bug report, feature request) with structured YAML forms
- Pull request template with review checklist
- Security policy (SECURITY.md) with vulnerability reporting instructions
- CONTRIBUTING.md at repo root linking to full contributing guide
- CODEOWNERS file for automatic PR review assignment
- Dependabot configuration for Python, GitHub Actions, and npm dependency updates
- `.gitattributes` for line ending normalization, binary markers, and linguist overrides
- Commitlint pre-commit hook for conventional commit enforcement
- PyPI release workflow with OIDC trusted publisher (no API tokens)
- VS Code Marketplace release workflow triggered by `vscode-v*` tags
- CI coverage enforcement (80% threshold on Python 3.13)
- Coverage XML artifact upload in CI
- CI badge in README
- `pip install jaxtyc` alongside `uv add` in README
- Contributing section in README

### Changed

- Pinned Flax to `>=0.10.0,<0.10.3` for JAX 0.9.x compatibility
- Made `analyze_file` import lazy in `__init__.py` to fix CPU backend enforcement on Apple Silicon
- Fixed `site_url` in mkdocs.yml to point to deployed GitHub Pages site
- Fixed Documentation URL in pyproject.toml to point to deployed docs
- Bumped `setup-uv` from v4/v5 to v8 in all workflows (Astral recommended)
- Switched Dependabot from `pip` to native `uv` ecosystem
- Simplified PyPI publish workflow to use `uv publish` instead of `pypa/gh-action-pypi-publish`
- Consolidated `setup-uv` `python-version` input (replaces separate `uv python install` steps)
- Updated CI guide docs to reference `setup-uv@v7`

### Fixed

- 5 `TestDetectProjectRoot` failures on macOS caused by `/var` -> `/private/var` symlink resolution
- 8 NNX test failures caused by Flax 0.11.2 / JAX 0.9.2 version incompatibility
- 1 CPU backend test failure caused by eager JAX import in `__init__.py`

## [v0.7.0] — 2026-04-04

### Added

- **Einops inline hints**: Inlay hints now display dimension names from einops pattern strings instead of raw symbolic sizes. When a function contains `einops.rearrange(x, 'b c h w -> b (c h) w')`, the inlay hint shows `f32[b, c*h, w]` instead of anonymous primes. Supports `rearrange`, `reduce`, and `repeat` with both `import einops` and `from einops import rearrange` styles. Enabled by default; disable with `einops_hints = false` in `[tool.jaxtyc]` or `JAXTYC_EINOPS_HINTS=0`
- New modules: `analyzer/einops_parser.py` (pattern string parsing), `analyzer/einops_detector.py` (AST-based call detection)
- New config: `einops_hints` (bool, default `true`) gates einops dimension name detection independently from `prefer_einops`
- Einops frames added to `_INTERNAL_PATHS` in tracer so JAXPR tracebacks resolve to user call sites instead of einops internals
- New tests: `test_einops_parser.py` (16 tests), `test_einops_detector.py` (10 tests)
- New fixture: `tests/fixtures/einops_rearrange.py`

### Changed

- **JAX backend extras overhaul**: `mac` extra now uses `jax>=0.4.34` floor and constrains `jax-mps` to `python_version == '3.13'` (only published wheel). `rocm` extra rewritten with direct GitHub wheel URLs for jaxlib, jax-rocm7-pjrt, and jax-rocm7-plugin since ROCm wheels are not on PyPI at 0.9.x. `cpu` extra adds explicit `jax[cpu]` dependency. New `rocm-jax` uv index for ROCm wheel source routing
- `_INTERNAL_PATHS` in `tracer.py` expanded to include `einops/` and `site-packages/einops`
- `pipeline.py` post-processes trace results to apply einops dimension name overrides before cross-function propagation

## [v0.6.2] — 2026-03-19

### Changed

- **Abstract-first NNX/equinox tracing**: Model tracing now uses `nnx.eval_shape` and `eqx.filter_eval_shape` for zero-allocation abstract construction instead of creating concrete model instances with real weight tensors. Abstract state is passed as explicit arguments to `jax.eval_shape` and `jax.make_jaxpr` (not captured in closures), so `ShapeDtypeStruct` leaves become proper JAX tracers inside the trace context. Falls back to concrete construction on CPU if abstract tracing fails for a particular model
- **CPU-only backend**: All CLI entry points (check, trace, watch, lsp, mux) now force `JAX_PLATFORMS=cpu` before any JAX import via `_enforce_cpu_backend()` in `cli/main.py`. This prevents JAX from pre-allocating 75% of GPU VRAM — `jax.eval_shape` and `jax.make_jaxpr` produce identical results on CPU. Override with `JAXTYC_BACKEND=gpu` env var or by setting `JAX_PLATFORMS` directly
- **NNX tracing refactor**: `_trace_nnx_method` split into `_trace_nnx_abstract` (primary, zero-alloc) and `_trace_nnx_concrete` (fallback, CPU-only). Same for equinox with `_trace_eqx_abstract` and `_trace_eqx_concrete`. Shared helpers `_extract_output_shape`, `_eval_and_extract`, and `_build_mesh_context` eliminate duplicated logic

### Added

- **`backend` config option**: `backend = "cpu"` (default) in `[tool.jaxtyc]` section of `pyproject.toml`. Documents the CPU-only behavior; set to `"gpu"` to allow GPU usage
- **Content-hash gating**: LSP `_analyze_and_publish` now hashes source text (SHA-256) and skips re-analysis when content is unchanged, avoiding redundant module imports, tracing, and model instantiation on every didChange debounce and didSave event
- **JAX cache cleanup**: `jax.clear_caches()` called after each LSP analysis cycle and after CLI check/watch to prevent unbounded growth of JAX's internal trace and compilation caches
- **VSCode extension CPU env vars**: Server process spawned with `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` as defense-in-depth. New `jaxtyc.backend` setting (cpu/gpu) controls this; server restarts when changed
- **Mux CPU env vars**: jaxtyc subprocess in mux mode spawned with `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- **Import CPU guard**: `exec_module` in `importer.py` wrapped with `jax.default_device(cpu)` to prevent user module-level code from allocating on GPU
- New test files: `test_cpu_backend.py` (5 tests), `test_content_hash.py` (5 tests), `test_cache_cleanup.py` (1 test)
- Extended tests: `test_pipeline.py` (+7 abstract tracing tests), `test_config.py` (+3 backend tests), `test_importer.py` (+2 sys.modules cleanup tests)

### Fixed

- **VRAM consumption in VSCode extension**: The LSP server no longer consumes GPU VRAM. Previously, JAX auto-detected the GPU on import, NNX/equinox model constructors allocated real weight tensors on GPU, and JAX caches grew unboundedly. Now: abstract tracing eliminates weight allocation, CPU backend prevents GPU initialization, content-hash gating reduces re-analysis, and cache cleanup prevents leak
- **sys.modules memory leak**: `import_module_from_path()` now removes `_jaxtyc_user_*` modules from `sys.modules` after loading. Previously, modules were added but never removed on success, causing the LSP server to accumulate module objects over its lifetime

- Test count: 676 -> 698

## [v0.6.1] — 2026-03-14

### Changed

- **Frozen `FileIndex`**: `FileIndex` is now `@dataclass(frozen=True)` with tuple fields (was mutable with list fields), consistent with the project convention for immutable value types
- **Dependency upper bounds**: Added semver caps to all dependencies — `jax>=0.9.0,<0.10`, `jaxtyping>=0.2.28,<0.4`, `pygls>=2.0,<3`, and all optional deps (`watchfiles`, `flax`, `equinox`, `einops`) — to guard against breaking changes in private APIs like `jax._src.mesh.use_abstract_mesh`

### Added

- **Logging infrastructure**: Added `logging.getLogger(__name__)` across analyzer modules (`tracer.py`, `source_map.py`, `pipeline.py`, `config.py`). Silent `except Exception: pass` blocks now log at `debug` or `warning` level with `exc_info=True`
- **Error truncation**: `truncate_error()` utility in new `analyzer/_errors.py` takes the first line and caps at 500 chars, applied to 10 `str(e)` sites in `pipeline.py`, `tracer.py`, and `cli/main.py`
- **Config type validation**: `_validate_field_types()` in `config.py` checks TOML values match expected types before constructing `JaxtycConfig`, logging warnings for rejected keys
- New test suites: `test_pipeline.py` (21 tests), `test_lsp_handlers.py` (26 tests)
- Extended tests: `test_config.py` (+7), `test_dim_env.py` (+6)

### Fixed

- **Silent exception swallowing**: 4 bare `except Exception: pass` blocks in `tracer.py`, `source_map.py`, and `config.py` now log diagnostics instead of silently discarding errors
- **Assertions removed from runtime guards**: 9 `assert` statements in `dim_env.py` (6) and `tracer.py` (3) replaced with explicit `ValueError` raises with descriptive messages, surviving `python -O`
- **TOCTOU race in LSP diagnostic cache**: `_analyze_and_publish` in `server.py` now performs cross-file checking before the cache write, then batch-writes all caches in a single lock acquisition

- Test count: 625 -> 676

## [v0.6.0] — 2026-03-12

### Changed

- **DimEnv rewrite**: Replaced prime-based dimension sizing with symbolic dimensions via `jax.export.symbolic_shape`. Each named dimension now gets a distinct symbolic `_DimExpr` object instead of a prime number. This eliminates integer overflow for large models, makes JAX error messages use real dimension names, and removes the prime sieve entirely
- **Sharding display default**: `ShardingConfig.display` now defaults to `"all"` (was `"append"`), showing `dim|axis` sharding annotations on every inlay hint line by default. Valid values: `"all"`, `"constrained_only"`, `"off"`
- **Minimum JAX version**: Raised to `>= 0.9.0` for `jax.export.symbolic_shape` and `AbstractMesh` with `AxisType.Explicit`
- **Concrete fallback for modules**: NNX and equinox module tracing uses `DimEnv.get_concrete_size()` (unique odd ints >= 101) instead of symbolic dims, since module constructors require plain int arguments
- Test count: 498 -> 625

### Added

- **Sharding-in-types (pipe syntax)**: Annotations now support `dim|axis` syntax for per-dimension sharding: `Float[Array, "batch|dp seq|None d_model|mp"]`. The pipe separator sets `DimSpec.mesh_axis`, parsed by `parse_shape_string()` in `annotations.py`
- **Mesh configuration**: `[tool.jaxtyc.sharding]` supports `mesh` (physical axis name -> device count) and `axis_rules` (logical -> physical mapping) fields in `pyproject.toml`
- **Mesh context for tracing**: Sharded functions are traced inside `jax._src.mesh.use_abstract_mesh()` with an `AbstractMesh` built from `mesh_config`, enabling sharding propagation through `eval_shape`
- **Axis rules resolution**: Logical axis names (e.g. `dp`, `mp` from `nnx.logical_axis_rules`) are resolved to physical mesh axis names (e.g. `data`, `model`) before constructing `NamedSharding` inputs. Threaded from `MeshInfo` through pipeline to tracer via `axis_rules` parameter
- **Graceful sharding fallback**: When sharded `eval_shape` fails (e.g. scatter, advanced indexing), the tracer retries without sharding and sets `TraceResult.sharding_fallback_reason`. Pipeline emits a `trace-error` warning when fallback is used
- **`sharding-mesh-undefined` rule**: Diagnostic error when a `mesh_axis` annotation references an axis not found in the mesh config or axis_rules
- **`sharding-propagation-mismatch` rule**: Detects when JAX-propagated output sharding differs from the return annotation's sharding
- **`sharding-annotation-incomplete` rule**: Warns when a piped shape has bare (unsharded) dims in strict mode
- **`sharding-dim-conflict` rule**: Detects when the same dim name is sharded on different axes across parameters
- **`DimSize` type alias**: `DimSize: TypeAlias = Any` in `types.py` for shape fields that hold either `int` or symbolic `_DimExpr` objects. All shape-typed fields (`DiagnosticData`, `IntermediateShape`, `TraceResult`) updated
- **`TraceResult.sharding_fallback_reason`**: New optional field indicating why sharded tracing fell back to unsharded
- **`_build_abstract_mesh()` helper**: Reusable function in `tracer.py` for constructing `AbstractMesh` with `AxisType.Explicit`
- **`check_mesh_axes()`**: New function in `sharding_checker.py` validating mesh axis references against known physical and logical axes
- **`check_annotation_sharding()`**: New function checking for `sharding-annotation-incomplete` and `sharding-dim-conflict` within function annotations
- **`check_sharding_propagation()`**: Compares JAX-propagated output sharding against return annotation sharding
- **`mesh_resolver.py`**: AST-based inference of `jax.make_mesh`, `AbstractMesh`, and `nnx.logical_axis_rules` from source code
- **Synthetic dim name cleanup**: `format_named_shape()` in `_util.py` collapses internal names for user-facing display: `_ellipsis_0, _ellipsis_1` -> `...(0, 1)`, `_var_batch_0, _var_batch_1` -> `*batch`, `_anon_N` -> `_`
- **Sharding diagnostics as inlay hints**: Sharding diagnostic rules (all 8 `sharding-*` rules) now appear as error hints at divergence points in inlay hints, not just as squiggly diagnostics. Added `_diagnostics_to_error_hints()` in `server.py`
- **Mesh axis autocomplete**: Dimension name completion now also suggests mesh axis names from the resolved `MeshInfo` when typing inside pipe-syntax annotations
- **NNX/equinox sharded tracing**: Module method tracing wraps `jax.eval_shape` in mesh context when sharding annotations are present
- **Bool annotation support**: `Bool[Array, "..."]` traces to `bool` dtype through the full pipeline (verified by integration test)
- **Mux diagnostic filtering**: `--solo` flag on `jaxtyc mux` and `JAXTYC_MUX_SOLO` env var to show diagnostics from only one server. Valid values: `jaxtyc`, `ty`, `primary`, `pyright`
- **`_filter_diag_sources()`**: Filters mux diagnostic cache by server source for solo mode
- New test suites: `test_mesh_resolver.py` (14), `test_sharding_checker.py` (24)
- Extended tests: `test_dim_env.py` (+12), `test_tracer.py` (+9), `test_types.py` (+3), `test_config.py` (+3), `test_lsp.py` (+10), `test_mux.py` (+13), `test_integration.py` (+2)

### Fixed

- **Ellipsis display in inlay hints**: Synthetic dim names (`_ellipsis_0`) no longer leak into user-facing inlay hints and hover. Now shows `...(0, 1)` with indices instead of `...(_ellipsis_0, _ellipsis_1)`
- **Variadic display in inlay hints**: `_var_batch_0, _var_batch_1` collapses to `*batch` in inlay hints
- **Anonymous dim display**: `_anon_N` shows as `_` in inlay hints

## [v0.5.0] — 2026-03-11

### Changed

- **Navigation defaults**: `references_scope` now defaults to `"workspace"` (was `"file"`) and `include_external_calls` now defaults to `true` (was `false`), giving cross-file references, workspace-wide incoming calls, and full call graphs out of the box
- **External call display**: `outgoingCalls` and hover now show qualified names for library calls (e.g. `jnp.matmul` instead of `matmul`)

### Added

- **`CallSite.callee_qualified_name`**: Optional field storing the full dotted path for attribute calls (e.g. `jnp.lax.scan`), extracted from `ast.Attribute` chains via new `_dotted_name()` helper in `annotations.py`
- **External call hover**: Hovering on an external call site (e.g. `jnp.dot(x, y)`) now shows `**jnp.dot** (external)` instead of returning nothing
- **Trace error fallback hover**: Hovering on an intermediate line inside a function whose tracing failed now shows the trace error message (e.g. `Trace error in encode: dot_general requires...`)
- **Navigation docs**: `[tool.jaxtyc.navigation]` section documented in configuration guide with `references_scope` and `include_external_calls` options
- New tests: qualified name extraction, hover trace error fallback, external call hover, external qualified name in outgoing calls
- Test count: 447 -> 498

## [v0.4.0] — 2026-03-10

### Added

- **Multi-root workspace support**: Per-folder LSP clients, each discovering its own Python environment independently
- **Status bar quick pick menu**: Click the status bar for Restart Server, Check Current File, Trace Function, Show Output, Open Settings
- **Server version tooltip**: Hover over the status bar to see the connected server version
- **Trace visualization**: `jaxtyc: Trace Function` command with webview panel showing shape flow, intermediate operations, and match/mismatch status
- **jaxtyping snippets**: `jfloat`, `jint`, `jbool`, `jshaped`, `jimport`, `jignore` for Python files
- **Problem matcher**: Parses `jaxtyc check` output into VS Code's Problems panel
- **GitHub Actions CI**: Python CI (ruff, ty, pytest 3.11/3.13 matrix) and VS Code extension CI (tsc, vitest, esbuild, vsce package)
- **Pre-commit hooks**: `.pre-commit-config.yaml` with ruff + ty via prek
- **Justfile**: Development task runner (`just vscode-update`, `just test-all`, etc.)
- VS Code extension test count: 30 -> 51 (added trace-panel tests)
- **Error hints (divergence detection)**: `ErrorHintInfo` type and `find_divergence_points()` in new `analyzer/divergence.py`; inline error display at the first operation whose shape deviates from the annotated return
- **Sharding checker**: New `analyzer/sharding_checker.py` with 4 diagnostic rules: `sharding-rank-mismatch`, `sharding-axis-unknown`, `sharding-conflict`, `sharding-io-mismatch`
- **Sharding extraction in tracer**: `_extract_sharding_info()` reads `PartitionSpec` and mesh axes from `sharding_constraint` and `shard_map` jaxpr primitives; attached to `IntermediateShape` via new `ShardingInfo` dataclass
- **NNX/Equinox intermediate extraction**: `_trace_nnx_method` and `_trace_eqx_method` now return populated `intermediates` via `_extract_intermediates` (previously returned `[]`)
- **Dimension-aware module construction**: `_collect_dim_kwargs()` introspects constructor signatures and builds prime-based kwargs, replacing hardcoded `d_in=2, d_out=3`
- **Compact inlay hint format**: `dtype[dim1, dim2]` with 3 dtype styles (`numpy`/`jax`/`jaxtyping`), last-per-line deduplication, smart positioning after variable names
- **Sharding display in inlay hints**: `P('data', None)` appended to shape when `ShardingInfo` present
- **Error display in inlay hints**: Divergence error message appended after shape with configurable separator (`pipe` or `icon`)
- **Sharding in hover**: Intermediates hover shows `P(...)` and mesh axis info inline
- **`HintsConfig`**: `error_mode`, `error_location`, `error_style`, `dtype_style` config options under `[tool.jaxtyc.hints]`
- **`ShardingConfig`**: `display` and `rules` config options under `[tool.jaxtyc.sharding]`
- **`format_dtype()`**: Comprehensive dtype abbreviation utility (numpy, jax, jaxtyping styles) covering FP8, FP4/FP6, sub-byte ints
- **LSP caches**: `error_hints_cache` and `source_cache` in `_state.py` for error hint and inlay hint positioning
- Test fixtures: `nnx_sharded.py`, `sharded_rank_mismatch.py`
- New test suites: `test_divergence.py` (8 tests), `test_sharding.py` (4 tests), `test_sharding_checker.py` (8 tests)
- Extended tests: `test_types.py` (+5), `test_config.py` (+19), `test_lsp.py` (+20), `test_nnx.py` (+7), `test_integration.py` (+1)

### Changed

- VS Code extension version: 0.1.4 -> 0.2.0
- Config change handler now debounces (1 second) before restarting servers
- Extension documentation updated with multi-root, snippets, trace, all commands
- **NNX tracing**: Switched from abstract `nnx.eval_shape` to concrete model construction with `nnx.split/merge` + `jax.eval_shape`
- **Inlay hints**: Rewritten with compact `dtype[dim1, dim2]` format, smart after-variable positioning, and error/sharding display
- **Hover intermediates**: Structured display with `dtype[dim1, dim2]` format, "final" marker, inline sharding info
- **`IntermediateShape`**: Extended with optional `sharding: ShardingInfo | None` field
- **`JaxtycConfig`**: Extended with nested `hints: HintsConfig` and `sharding: ShardingConfig`
- **`filter_diagnostics`**: Now additionally filters sharding diagnostics against the `sharding.rules` allow-list
- **`didClose` handler**: Also clears `error_hints_cache` and `source_cache`

## [v0.3.1] — 2026-03-10

### Fixed

- **Prime collision with literal dimensions**: `DimEnv` now skips primes below `MIN_PRIME` (101) and reserves literal dimension values, preventing ambiguous `resolve_name()` results for annotations like `Float[Array, "batch 2"]`
- **Anonymous dimension collision**: Each anonymous dim (`_`) now gets a globally unique counter instead of position-based naming, preventing cross-function prime sharing
- **Pipeline zip misalignment**: Traced results are now indexed by function name instead of zipped with specs, preventing wrong spec/trace pairing when functions are skipped during tracing
- **sys.path pollution**: `import_module_from_path()` now saves and restores `sys.path` after module loading, preventing monotonic growth in long-running LSP sessions
- **read_message crash on IncompleteReadError**: Mux `read_message()` now catches `asyncio.IncompleteReadError` and `ConnectionResetError`, returning `None` instead of crashing the output handler
- **TextDocumentSyncKind defaulting to Incremental**: LSP server now explicitly sets `TextDocumentSyncKind.Full`, fixing `didChange` content extraction when running `jaxtyc lsp` directly (without mux)
- **Mux URI percent-decoding**: `_uri_to_path()` now uses `urllib.parse.unquote` + `urlparse` instead of naive string slicing, correctly handling paths with spaces and special characters
- **Mux hardcoded version**: Synthetic initialize response now uses `importlib.metadata.version()` instead of a hardcoded string
- **Unsynchronized cache access**: Added `cache_lock` to `_state.py` for atomic multi-cache reads and writes across threads
- **`typing.Tuple` not recognized**: Annotation parser now handles both `tuple` and `Tuple` (case-insensitive) for return type extraction
- **Missing posonly/kwonly arg parsing**: Annotation extractor now iterates `posonlyargs + args + kwonlyargs`, catching all jaxtyping-annotated parameters
- **`async def` name column offset**: `FunctionShapeSpec.name_col_offset` now accounts for 10-char `async def` prefix instead of hardcoded 4-char `def` offset
- **Dual-request timeout tasks linger**: `finalize_dual()` now cancels pending timeout tasks when both servers respond before the 3-second window
- **`$/cancelRequest` ID remapping**: Cancel requests now forward the remapped jaxtyc ID to the jaxtyc server instead of the client's original ID

### Added

- **Code action resolve handler**: Shape-fix quick actions now attach `WorkspaceEdit` via `codeAction/resolve` instead of being no-ops
- **Config-based diagnostic filtering in LSP**: `filter_diagnostics()` now applied in `_analyze_and_publish`, respecting `severity` threshold and `ignore_rules` from `[tool.jaxtyc]`
- **Mux synthetic capabilities**: Added `signatureHelpProvider`, `semanticTokensProvider`, `linkedEditingRangeProvider`, `documentHighlightProvider`, `prepareRenameProvider` to the mux initialize response
- **CLI integration tests**: 43 new tests covering config-based filtering (`severity`, `ignore_rules`, `exclude`), all output formats, `main()` direct invocation, helper functions, env var overrides, and end-to-end fixture checks
- **Mux integration tests**: 54 new tests covering all helper functions (`_detect_project_root`, `_patch_root_uri`, `_extract_file_path_from_msg`, `_find_primary_server`, `_hover_compact_enabled`, `_clean_hover_text`), merge strategies, and full subprocess lifecycle (synthetic init, server startup, diagnostics, clean exit)
- **Expanded unit tests**: `test_checker.py` (call-site checks, reserved dims), `test_dim_env.py` (literal collision, anonymous uniqueness), `test_annotations.py` (posonly/kwonly args, async def offset, `Tuple` return), `test_integration.py` (tuple return, cross-function mismatch), `test_self_analysis.py` (orphaned fixtures wired in)

### Changed

- **pyproject.toml**: Fixed `target-version` from `py313` to `py311`; added ruff rules N, A, DTZ, T10, RET; added `[tool.coverage.run]` and `[tool.coverage.report]` sections; added Python 3.14 classifier; added `[tool.jaxtyc]` self-config section; added `required-version` and `environments` to `[tool.uv]`; added Changelog URL to `[project.urls]`; added `filterwarnings` to pytest config
- Test count: 273 -> 368

## [v0.3.0] — 2026-03-09

### Added

- **Code actions**: Quick-fix suggestions for shape mismatches (transpose, expand_dims, squeeze, reshape) with einops support via `prefer_einops` config or `JAXTYC_PREFER_EINOPS` env var
- **Dimension name completion**: Autocomplete dimension names inside jaxtyping shape strings
- **Inlay hints**: Inline resolved shapes at end of lines with intermediate operations
- **Semantic tokens**: Syntax highlighting for dimension names in jaxtyping annotations
- **Linked editing range**: Simultaneous editing of dimension names within the same function
- **Signature help**: Shape signatures shown when calling jaxtyping-annotated functions
- **Folding ranges**: Collapsible ranges for functions with many shape-annotated parameters
- **Diagnostic pull model**: `textDocument/diagnostic` support alongside existing push model
- **Configuration hot-reload**: `workspace/didChangeConfiguration` and `pyproject.toml` file watching
- **Hover enhancement**: Dimension names show prime size, usage count, and locations; parameter names show shape annotations (fills the gap where ty reports `Any` for jaxtyping types); function names show full shape signatures
- **LSP multiplexer**: `jaxtyc mux` command runs both a primary type checker (ty/pyright, auto-discovered) and jaxtyc behind a single stdio pipe with dual-send merge strategy
- **Venv auto-discovery**: Importer discovers project virtual environments following ty's resolution order (`VIRTUAL_ENV` env var, `.venv` at project root, walk-up), adds `site-packages` and project `src/` to `sys.path`
- **Workspace root propagation**: Mux extracts `rootUri` from LSP `initialize` request and passes it as `cwd` to both servers for correct environment discovery
- **Cross-function shape propagation**: Detects shape mismatches across function call boundaries (`cross-function-mismatch` rule)
- **Parameter consistency checking**: Verifies parameter annotations match resolved shapes (`param-inconsistency` rule)
- **Multi-output / PyTree return checking**: Validates `tuple[Float[...], Float[...]]` return annotations (`return-count-mismatch` rule)
- **Inline suppression comments**: `# jaxtyc: ignore` and `# jaxtyc: ignore[rule]` syntax
- **Cross-file dimension references**: Find references and rename work across files
- **didClose handler**: Clears caches and diagnostics when files are closed
- **Shared DimEnv per file**: All functions in a file share one dimension environment for cross-function consistency
- **Structured diagnostic data**: `DiagnosticData` with expected/actual shapes, dim names, and suggested fixes
- **Einops integration**: `prefer_einops = true` in `[tool.jaxtyc]` or `JAXTYC_PREFER_EINOPS=1` env var switches suggestions to einops notation
- **Claude Code plugin**: Pre-built `.claude-plugin/plugin.json` for editor integration
- **Project CLAUDE.md**: Development guide with commands, structure, and conventions
- `ShapeFix` suggestion type in `suggestions.py` for programmatic fix generation
- `SuppressionComment` type for inline suppression tracking
- `DiagnosticData` type for structured diagnostic metadata
- Test suites: `test_importer.py`, `test_formatters.py`, `test_suggestions.py`
- Test fixtures: `cross_function_mismatch.py`, `tuple_return.py`, `tuple_return_mismatch.py`, `suppressed.py`

### Changed

- LSP server refactored from single `server.py` into focused modules (`_code_actions.py`, `_completion.py`, `_semantic_tokens.py`, `_signature_help.py`, `_inlay_hints.py`, `_linked_editing.py`, `_folding.py`, `_configuration.py`, `_diagnostics.py`, `_navigation.py`, `mux.py`)
- Version string now uses `importlib.metadata` instead of hardcoded value
- CI workflow adds `ty check` step and pytest coverage with 80% threshold
- `TraceResult` extended with `input_shapes`, `output_shapes`, `output_dtypes`
- `FunctionShapeSpec` extended with `end_lineno`, `return_specs`
- `WorkspaceIndex` extended with `find_function_containing()` for multi-line signature lookup
- CLI extended with `mux` subcommand for multiplexer mode
- `import_module_from_path` now auto-discovers project venvs and adds `site-packages`/`src/` to `sys.path`
- `Diagnostic` extended with optional `data: DiagnosticData`

## [v0.2.0] — 2026-03-09

### Added

- **LSP navigation for dimension names**: `textDocument/definition`, `textDocument/references`, `textDocument/documentHighlight`, `textDocument/prepareRename`, `textDocument/rename` — navigate and refactor dimension names (`batch`, `seq`, `d_model`) as first-class semantic entities
- **LSP document/workspace symbols**: `textDocument/documentSymbol` and `workspace/symbol` with shape detail summaries
- **LSP call hierarchy**: `prepareCallHierarchy`, `callHierarchy/incomingCalls`, `callHierarchy/outgoingCalls` for shape-annotated functions
- **LSP implementation**: `textDocument/implementation` (delegates to definition logic)
- `DimLocation` type for tracking exact source positions of dimension name tokens within annotation strings
- `CallSite` type for tracking call relationships between shape-annotated functions
- `FileIndex` / `WorkspaceIndex` thread-safe cross-file navigation index
- `extract_dim_locations()` and `extract_call_sites()` AST parsers in `annotations.py`
- `build_file_index()` orchestrator for per-file index construction
- `tests/fixtures/multi_function.py` fixture for call hierarchy testing
- 21 new tests (9 unit, 12 LSP integration)

### Fixed

- `PrepareRenameResult_Type1` replaced with correct `PrepareRenamePlaceholder` lsprotocol type

### Changed

- `_analyze_and_publish` now extracts function specs once and reuses them for both CodeLens and navigation index (avoids double AST parse)
- Navigation index is built alongside diagnostics on every didOpen/didSave/didChange cycle

## [v0.1.0] — 2026-02-23

### Added

- AST-based jaxtyping annotation parser supporting all dtype classes and shape patterns (named, fixed, variadic, anonymous, ellipsis, scalar)
- `jax.eval_shape` tracing with prime-based symbolic dimension sizing via `DimEnv`
- Shape checker producing `shape-mismatch`, `rank-mismatch`, and `trace-error` diagnostics
- `jax.make_jaxpr` source mapping for per-line intermediate shape extraction
- CLI with 5 subcommands: `check`, `trace`, `watch`, `lsp`, `version`
- 4 output formats: `full`, `concise`, `json`, `github` (Actions annotations)
- pygls-based LSP server with diagnostics (didOpen/didSave/didChange), hover, and CodeLens
- Configuration via `[tool.jaxtyc]` in `pyproject.toml`: severity, ignore_rules, exclude, debounce_ms
- Flax NNX and Equinox module support (auto-skips `self`/`cls` parameters)
- CI workflow and mkdocs documentation site

[v0.6.2]: https://github.com/BeeGass/jaxtyc/compare/v0.6.1...v0.6.2
[v0.6.1]: https://github.com/BeeGass/jaxtyc/compare/v0.6.0...v0.6.1
[v0.6.0]: https://github.com/BeeGass/jaxtyc/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/BeeGass/jaxtyc/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/BeeGass/jaxtyc/compare/v0.3.1...v0.4.0
[v0.3.1]: https://github.com/BeeGass/jaxtyc/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/BeeGass/jaxtyc/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/BeeGass/jaxtyc/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/BeeGass/jaxtyc/releases/tag/v0.1.0
