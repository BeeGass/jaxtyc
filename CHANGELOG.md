# Changelog

All notable changes to jaxtyc are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[v0.2.0]: https://github.com/BeeGass/jaxtyc/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/BeeGass/jaxtyc/releases/tag/v0.1.0
