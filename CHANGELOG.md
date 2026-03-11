# Changelog

All notable changes to jaxtyc are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — VS Code Extension v0.2.0

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

### Changed

- VS Code extension version: 0.1.4 -> 0.2.0
- Config change handler now debounces (1 second) before restarting servers
- Extension documentation updated with multi-root, snippets, trace, all commands

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

[v0.3.1]: https://github.com/BeeGass/jaxtyc/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/BeeGass/jaxtyc/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/BeeGass/jaxtyc/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/BeeGass/jaxtyc/releases/tag/v0.1.0
