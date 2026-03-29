# jaxtyc

Static array shape checker for JAX. Uses `jax.eval_shape` with symbolic dimensions (`jax.export.symbolic_shape`) to trace jaxtyping-annotated functions and detect shape mismatches at analysis time. Supports sharding-in-types via `|axis` syntax.

## Development

```bash
uv run jaxtyc check src/                              # Shape-check all files
uv run jaxtyc check file.py                           # Check one file
uv run jaxtyc trace file.py::fn                       # Trace a function
uv run jaxtyc watch src/                              # Watch mode
uv run jaxtyc lsp                                     # Start LSP server
uv run pytest --tb=short -q                           # Tests
uv run ruff check . && uv run ruff format --check .   # Lint
uv run ty check                                       # Type check
```

## Conventions

- Python 3.11+, 3.12 excluded (JAX wheel gaps). JAX >= 0.9.0.
- All value types in `types.py` are `@dataclass(frozen=True)`.
- CPU-only by default (`_enforce_cpu_backend()` sets `JAX_PLATFORMS=cpu`).
- Content-hash gating: LSP skips re-analysis when SHA-256 matches cached hash.
- Inline suppression: `# jaxtyc: ignore` or `# jaxtyc: ignore[rule-name]`.

## Commit Messages

Conventional commit format: `type(scope): subject`. Types: feat, fix, refactor, test, docs, perf, build, ci, chore. Scope: analyzer, lsp, cli, types, mux, config. Imperative mood, lowercase, no trailing period. Body required for non-trivial changes.
