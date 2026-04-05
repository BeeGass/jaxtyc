# Contributing

## Dev Setup

```bash
git clone https://github.com/BeeGass/jaxtyc.git
cd jaxtyc
uv sync --group dev
prek install              # install pre-commit hooks
prek install --hook-type commit-msg  # install commitlint hook
```

This installs all dev dependencies (ruff, ty, pytest, pytest-cov) and jaxtyc itself in editable mode. The pre-commit hooks run ruff and ty on every commit, and commitlint enforces conventional commit messages.

## Running Checks

```bash
# Tests
uv run pytest --tb=short -q

# Tests with coverage (must be >= 80%)
uv run pytest --cov --cov-fail-under=80 -q

# Lint
uv run ruff check . && uv run ruff format --check .

# Type check
uv run ty check

# All pre-commit hooks
prek run --all-files
```

All checks must pass before submitting a PR. CI runs these automatically.

## Project Structure

```
src/jaxtyc/
  __init__.py              # Public API: analyze_file, Diagnostic, FileResult, TraceResult
  types.py                 # Core dataclasses: DimSpec, ShapeSpec, FunctionShapeSpec, DiagnosticData, etc.
  config.py                # [tool.jaxtyc] config loading from pyproject.toml

  analyzer/
    annotations.py         # AST parser for jaxtyping annotations, dim locations, call sites
    dim_env.py             # Prime sieve dimension environment (shared per file)
    importer.py            # Dynamic module import with venv auto-discovery
    tracer.py              # jax.eval_shape / jax.make_jaxpr tracing
    source_map.py          # Jaxpr source_info frame extraction
    checker.py             # Shape comparison and diagnostic emission (14 rules)
    divergence.py          # Divergence detection: find first shape deviation
    sharding_checker.py    # Sharding validation: rank, axis, conflict, io-mismatch
    suppressions.py        # Inline # jaxtyc: ignore comment parsing
    pipeline.py            # End-to-end orchestration (analyze_file)

  cli/
    main.py                # CLI entry point: check, trace, watch, lsp, mux, version
    formatters.py          # Output formatters: full, concise, json, github

  lsp/
    server.py              # pygls-based LSP server core + _analyze_and_publish
    _state.py              # Shared mutable caches (analysis, CodeLens, diagnostics, DimEnv)
    _util.py               # uri_to_path, dim_range, shape_summary, debounce_seconds
    _diagnostics.py        # didOpen, didSave, didChange, didClose, pull model
    _navigation.py         # Hover, CodeLens, symbols, definition, references, rename, call hierarchy
    _code_actions.py       # Quick fixes with shape suggestions + suppress action
    _completion.py         # Dimension name autocomplete in shape strings
    _semantic_tokens.py    # Semantic highlighting for dim names
    _signature_help.py     # Shape signatures for function calls
    _inlay_hints.py        # Inline shape annotations
    _linked_editing.py     # Simultaneous dim name editing
    _folding.py            # Folding ranges for annotated functions
    _configuration.py      # Config hot-reload on pyproject.toml changes
    suggestions.py         # Shape fix generation (JAX-native + einops)
    index.py               # WorkspaceIndex: cross-file navigation
    mux.py                 # LSP multiplexer (ty/pyright + jaxtyc, single stdio pipe)

tests/
  conftest.py              # Shared fixtures and test configuration
  fixtures/                # Python files used as test inputs
    correct_attention.py   # Zero-diagnostic baseline
    wrong_transpose.py     # Triggers shape-mismatch
    wrong_rank.py          # Triggers rank-mismatch
    wrong_inner_dim.py     # Inner dimension alignment error
    multi_function.py      # Multiple annotated functions
    cross_function_mismatch.py  # Cross-function shape inconsistency
    tuple_return.py        # Correct tuple return
    tuple_return_mismatch.py    # Tuple return count error
    suppressed.py          # Inline suppression comments
    nnx_module.py          # Flax NNX module fixture
    eqx_module.py          # Equinox module fixture
    nnx_sharded.py         # Flax NNX module with sharding constraints
    sharded_rank_mismatch.py  # Sharding rank mismatch trigger
    ellipsis_patterns.py   # Variadic and ellipsis annotation tests
    int_annotations.py     # Int dtype annotations
    bool_annotations.py    # Bool dtype annotations
    complex_annotations.py # Complex dtype annotations
    key_annotations.py     # Key dtype annotations
    shaped_annotations.py  # Shaped dtype annotations
    untraceable.py         # Non-jaxtyping files (silently skipped)
  test_annotations.py      # Annotation parser tests
  test_dim_env.py          # DimEnv prime assignment tests
  test_checker.py          # Shape checker tests (all 10 rules)
  test_integration.py      # Full pipeline tests (analyze_file on fixtures)
  test_cli.py              # CLI invocation tests
  test_config.py           # Configuration loading tests
  test_divergence.py       # Divergence detection tests
  test_sharding.py         # Sharding extraction tests
  test_sharding_checker.py # Sharding checker tests
  test_importer.py         # Module importer + venv discovery tests
  test_lsp.py              # LSP server handler tests
  test_mux.py              # LSP multiplexer tests
  test_nnx.py              # Flax NNX tracing tests
  test_source_map.py       # Source mapping tests
  test_tracer.py           # Tracer tests
  test_types.py            # Dataclass construction tests
  test_public_api.py       # Public API surface tests
  test_formatters.py       # Output formatter tests
  test_suggestions.py      # Shape fix suggestion tests
  test_suppressions.py     # Inline suppression tests
  test_index.py            # WorkspaceIndex tests
  test_self_analysis.py    # Self-analysis (jaxtyc checking its own fixtures)
  test_benchmarks.py       # Performance benchmarks
```

## Adding a New Diagnostic Rule

1. **Define the rule string** in `checker.py`. Add a new branch to `check_function()` or `_check_shape()` that constructs a `Diagnostic` with your rule code and structured `DiagnosticData`:

    ```python
    Diagnostic(
        file=func_spec.file_path,
        line=func_spec.lineno,
        col=func_spec.col_offset,
        severity="error",  # or "warning" / "info"
        message="Description of what went wrong",
        rule="your-rule-name",
        data=DiagnosticData(
            expected_shape=expected,
            actual_shape=actual,
            expected_named=env.shape_to_names(expected),
            actual_named=env.shape_to_names(actual),
            dim_name_mapping=env.name_size_mapping(),
            suggested_fix=_suggest_fix(expected, actual, env),
            rule="your-rule-name",
        ),
    )
    ```

2. **Create a test fixture** in `tests/fixtures/` -- a minimal `.py` file with jaxtyping annotations that triggers the new rule.

3. **Add test cases** in `tests/test_checker.py` (unit-level, calling `check_function` directly) and/or `tests/test_integration.py` (end-to-end, calling `analyze_file` on your fixture).

4. **Document the rule** in `docs/reference/diagnostics.md` -- add a row to the rule catalog table and, if it is error-severity, include a code example.

## Adding a New Annotation Pattern

1. **Update the parser** in `annotations.py`:
    - For a new dtype class (e.g., `Sparse`): add an entry to `_DTYPE_MAP`.
    - For new shape syntax (e.g., a new dim kind): update `parse_shape_string()` to handle the new token pattern and add a corresponding `DimSpec` kind in `types.py`.

2. **Update DimEnv** if the new kind requires special prime assignment logic (add a new `case` branch in `make_shape()`).

3. **Add tests** in `tests/test_annotations.py` covering parsing of the new pattern, and in `tests/test_integration.py` with a fixture that uses it end-to-end.

## Branch Workflow

- **`dev`** is the active development branch. All work happens here.
- **`main`** is the stable release branch. It only receives updates via `just pr-to-main`.
- AI assistant config files (`.claude/`, `.codex/`) live on `dev` only and are stripped before merging to `main`.

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). The commitlint pre-commit hook enforces this format:

```
<type>(<scope>): <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## PR Process

1. Fork the repository.
2. Create a feature branch from `dev` (not `main`).
3. Make your changes. Run all checks (`pytest`, `ruff`, `ty`).
4. Submit a pull request against `dev`.

CI will auto-lint your PR with ruff. The following checks must pass:

- `lint-and-type-check` (ruff + ty)
- `test (3.11)` and `test (3.13)` (pytest with 80% coverage on 3.13)
- `pre-commit` (prek hooks)

!!! tip
    Keep PRs focused -- one diagnostic rule or annotation pattern per PR. This simplifies review and makes the commit history useful for bisecting.

## Releases

Releases are automated. When a version bump in `pyproject.toml` is merged to `main`:

- **Patch/minor bumps** create a GitHub Release automatically, which triggers PyPI publishing.
- **Major bumps** create a draft release requiring manual approval before publishing.
