# Changelog

All notable changes to jaxtyc are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[v0.3.0]: https://github.com/BeeGass/jaxtyc/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/BeeGass/jaxtyc/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/BeeGass/jaxtyc/releases/tag/v0.1.0
