# Changelog

All notable changes to jaxtyc are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.4.0 — 2026-03-10

### Added

- **VS Code extension v0.2.0** (`editors/vscode/`): Native extension with auto-discovery of Python environments, status bar, and full LSP client integration
- **Multi-root workspace support**: Per-folder LSP clients, each discovering its own Python environment independently. Status bar reflects aggregate health across all folders
- **Python auto-detection**: Extension discovers jaxtyc across `VIRTUAL_ENV`, workspace `.venv` directories (including worktree subdirs), `jaxtyc` on PATH, VS Code Python extension interpreter, and `python3` fallback -- validates each candidate before use
- **Mux mode support**: `jaxtyc.mode` setting switches between `lsp` (shape checking only) and `mux` (multiplexed with ty/pyright)
- **Status bar quick pick menu**: Click the status bar to access Restart Server, Check Current File, Trace Function, Show Output, and Open Settings
- **Server version tooltip**: Hover over the status bar to see the connected jaxtyc server version
- **Trace visualization**: `jaxtyc: Trace Function` command runs `jaxtyc trace` and renders results in a webview panel with shape flow table, intermediate operations, and match/mismatch status
- **jaxtyping snippets**: 6 snippets (`jfloat`, `jint`, `jbool`, `jshaped`, `jimport`, `jignore`) for Python files
- **Problem matcher**: Parses `jaxtyc check` output format into VS Code's Problems panel
- **Config change watcher**: Automatically restarts servers (with 1-second debounce) when `jaxtyc.mode` or `jaxtyc.pythonPath` settings change
- **GitHub Actions CI**: Python CI workflow (ruff, ty, pytest matrix on 3.11/3.13) and VS Code extension CI (tsc, vitest, esbuild, vsce package with artifact upload)
- **Pre-commit hooks**: `.pre-commit-config.yaml` with ruff-check, ruff-format, and ty via prek (Rust-based pre-commit replacement)
- **Justfile**: Development task runner with `just vscode-update` pipeline (tool install, bundle, package, install)
- **Test suite**: 51 vitest tests (39 discovery + 12 trace-panel) covering discovery logic, candidate ordering, command building, trace parsing, HTML rendering, and edge cases
- **Error hints (divergence detection)**: `ErrorHintInfo` type and `find_divergence_points()` in new `analyzer/divergence.py`; inline error display at the first operation whose shape deviates from the annotated return
- **Sharding checker**: New `analyzer/sharding_checker.py` with 4 diagnostic rules: `sharding-rank-mismatch`, `sharding-axis-unknown`, `sharding-conflict`, `sharding-io-mismatch`
- **Sharding extraction in tracer**: `_extract_sharding_info()` reads `PartitionSpec` and mesh axes from `sharding_constraint` and `shard_map` jaxpr primitives; attached to `IntermediateShape` via new `ShardingInfo` dataclass
- **NNX/Equinox intermediate extraction**: Module tracing now returns populated `intermediates` via `_extract_intermediates` (previously returned `[]`)
- **Dimension-aware module construction**: `_collect_dim_kwargs()` introspects constructor signatures and builds prime-based kwargs, replacing hardcoded `d_in=2, d_out=3`
- **Compact inlay hint format**: `dtype[dim1, dim2]` with 3 dtype styles (numpy, jax, jaxtyping), last-per-line deduplication, smart positioning after variable names
- **Sharding display in inlay hints and hover**: `P('data', None)` appended to shape; hover shows mesh axis info
- **Error display in inlay hints**: Divergence error message appended with configurable separator (pipe or icon)
- **`HintsConfig`**: `error_mode`, `error_location`, `error_style`, `dtype_style` under `[tool.jaxtyc.hints]`
- **`ShardingConfig`**: `display` and `rules` under `[tool.jaxtyc.sharding]`
- **`format_dtype()`**: Dtype abbreviation utility (numpy, jax, jaxtyping styles) covering FP8, FP4/FP6, sub-byte ints
- New test suites: `test_divergence.py` (8), `test_sharding.py` (4), `test_sharding_checker.py` (8)
- Extended tests: `test_types.py` (+5), `test_config.py` (+19), `test_lsp.py` (+20), `test_nnx.py` (+7), `test_integration.py` (+1)

### Changed

- **Editors documentation**: VS Code section updated with multi-root support, snippets, trace visualization, all new commands, full configuration table
- **README**: Updated VS Code install instructions with justfile alternative
- **NNX tracing**: Switched from abstract `nnx.eval_shape` to concrete model construction with `nnx.split/merge` + `jax.eval_shape`
- **Inlay hints**: Rewritten with compact `dtype[dim1, dim2]` format, smart after-variable positioning, error/sharding display
- **Hover intermediates**: Structured display with `dtype[dim1, dim2]` format, "final" marker, inline sharding
- **`IntermediateShape`**: Extended with optional `sharding: ShardingInfo | None` field
- **`JaxtycConfig`**: Extended with nested `hints: HintsConfig` and `sharding: ShardingConfig`
- **`filter_diagnostics`**: Now additionally filters sharding diagnostics against `sharding.rules` allow-list
- **`didClose` handler**: Also clears `error_hints_cache` and `source_cache`

## v0.3.1 — 2026-03-10

### Fixed

- **Prime collision with literal dimensions**: `DimEnv` now skips primes below `MIN_PRIME` (101) and reserves literal dimension values, preventing ambiguous `resolve_name()` for annotations like `Float[Array, "batch 2"]`
- **Anonymous dimension collision**: Each anonymous dim (`_`) gets a globally unique counter instead of position-based naming
- **Pipeline zip misalignment**: Traced results indexed by function name instead of zipped with specs
- **sys.path pollution**: `import_module_from_path()` saves and restores `sys.path` after module loading
- **read_message crash on IncompleteReadError**: Mux catches incomplete reads gracefully
- **TextDocumentSyncKind defaulting to Incremental**: LSP server explicitly sets `Full` sync
- **Mux URI percent-decoding**: Uses `urllib.parse.unquote` for correct path handling
- **Mux hardcoded version**: Uses `importlib.metadata.version()` dynamically
- **Unsynchronized cache access**: Added `cache_lock` for atomic multi-cache operations
- **`typing.Tuple` not recognized**: Case-insensitive tuple return type detection
- **Missing posonly/kwonly arg parsing**: Iterates all arg types in annotation extraction
- **`async def` name column offset**: Correct 10-char offset for async functions
- **Dual-request timeout cleanup**: Cancels pending timeout tasks on early completion
- **`$/cancelRequest` ID remapping**: Forwards remapped jaxtyc ID correctly

### Added

- **Code action resolve handler**: Shape-fix quick actions now attach `WorkspaceEdit` via `codeAction/resolve`
- **Config-based diagnostic filtering in LSP**: `filter_diagnostics()` applied in `_analyze_and_publish`
- **Mux synthetic capabilities**: 5 missing capability providers added to initialize response
- **CLI integration tests**: 43 new tests covering config filtering, formats, helpers, env overrides
- **Mux integration tests**: 54 new tests covering helpers, merge strategies, subprocess lifecycle
- **Expanded unit tests**: Call-site checks, reserved dims, posonly/kwonly args, async def, Tuple returns

### Changed

- **pyproject.toml**: Fixed `target-version` to `py311`; added ruff rules N/A/DTZ/T10/RET; added coverage config; Python 3.14 classifier; `[tool.jaxtyc]` self-config; uv `required-version` and `environments`
- Test count: 273 -> 368

## v0.3.0 — 2026-03-10

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

## v0.2.0 — 2026-03-09

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
- 21 new tests (9 unit, 12 LSP integration)

### Fixed

- `PrepareRenameResult_Type1` replaced with correct `PrepareRenamePlaceholder` lsprotocol type

### Changed

- `_analyze_and_publish` now extracts function specs once and reuses them for both CodeLens and navigation index (avoids double AST parse)
- Navigation index is built alongside diagnostics on every didOpen/didSave/didChange cycle

## v0.1.0 — 2026-02-23

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
