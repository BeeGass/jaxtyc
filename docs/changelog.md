# Changelog

## v0.1.0 — Initial Release

- AST-based jaxtyping annotation parser supporting all dtype classes and shape patterns (named, fixed, variadic, anonymous, ellipsis, scalar)
- `jax.eval_shape` tracing with prime-based symbolic dimension sizing via `DimEnv`
- Shape checker producing `shape-mismatch`, `rank-mismatch`, and `trace-error` diagnostics
- `jax.make_jaxpr` source mapping for per-line intermediate shape extraction
- CLI with 5 subcommands: `check`, `trace`, `watch`, `lsp`, `version`
- 4 output formats: `full`, `concise`, `json`, `github` (Actions annotations)
- pygls-based LSP server with diagnostics (didOpen/didSave/didChange), hover, and CodeLens
- Configuration via `[tool.jaxtyc]` in `pyproject.toml`: severity, ignore_rules, exclude, debounce_ms
- Flax NNX support (auto-skips `self`/`cls` parameters)
