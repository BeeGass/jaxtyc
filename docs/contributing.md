# Contributing

## Dev Setup

```bash
git clone https://github.com/BeeGass/jaxtyc.git
cd jaxtyc
uv sync
```

This installs all dev dependencies (ruff, ty, pytest, pytest-cov) and jaxtyc itself in editable mode.

## Running Checks

```bash
# Tests
uv run pytest

# Lint
uv run ruff check . && uv run ruff format --check .

# Type check
uv run ty check
```

All three must pass before submitting a PR.

## Project Structure

```
src/jaxtyc/
  __init__.py              # Public API: analyze_file, Diagnostic, FileResult, TraceResult
  types.py                 # Core dataclasses: DimSpec, ShapeSpec, FunctionShapeSpec, etc.
  config.py                # [tool.jaxtyc] config loading from pyproject.toml
  analyzer/
    annotations.py         # AST parser for jaxtyping annotations
    dim_env.py             # Prime sieve dimension environment
    importer.py            # Dynamic module import via importlib
    tracer.py              # jax.eval_shape / jax.make_jaxpr tracing
    source_map.py          # Jaxpr source_info frame extraction
    checker.py             # Shape comparison and diagnostic emission
    pipeline.py            # End-to-end orchestration (analyze_file)
  cli/
    main.py                # CLI entry point: check, trace, watch, lsp, version
    formatters.py          # Output formatters: full, concise, json, github
  lsp/
    server.py              # pygls-based LSP server

tests/
  fixtures/                # Python files used as test inputs
    correct_attention.py   # Zero-diagnostic baseline
    wrong_transpose.py     # Triggers shape-mismatch
    wrong_rank.py          # Triggers rank-mismatch
    untraceable.py         # Non-jaxtyping files (silently skipped)
    ellipsis_patterns.py   # Variadic and ellipsis annotation tests
  test_annotations.py      # Annotation parser tests
  test_dim_env.py          # DimEnv prime assignment tests
  test_checker.py          # Shape checker tests
  test_integration.py      # Full pipeline tests (analyze_file on fixtures)
  test_cli.py              # CLI invocation tests
  test_lsp.py              # LSP server tests
  test_source_map.py       # Source mapping tests
  test_tracer.py           # Tracer tests
  test_types.py            # Dataclass construction tests
  test_public_api.py       # Public API surface tests
```

## Adding a New Diagnostic Rule

1. **Define the rule string** in `checker.py`. Add a new branch to `check_function()` or `_check_shape()` that constructs a `Diagnostic` with your rule code:

    ```python
    Diagnostic(
        file=func_spec.file_path,
        line=func_spec.lineno,
        col=func_spec.col_offset,
        severity="error",  # or "warning" / "info"
        message="Description of what went wrong",
        rule="your-rule-name",
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

## PR Process

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes. Run all checks (`pytest`, `ruff`, `ty`).
4. Submit a pull request against `main`.

!!! tip
    Keep PRs focused -- one diagnostic rule or annotation pattern per PR. This simplifies review and makes the commit history useful for bisecting.
