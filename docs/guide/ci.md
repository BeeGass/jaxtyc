# CI Integration

jaxtyc is designed to run in CI pipelines. The `check` command returns a nonzero exit code on shape errors and supports output formats tailored for automated environments.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No errors (after config filtering). |
| `1` | One or more errors found. |

---

## GitHub Actions

Use `--format github` to produce [workflow commands](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/workflow-commands-for-github-actions#setting-an-error-message) that render as inline annotations on pull request diffs.

```yaml
name: Shape Check
on: [push, pull_request]

jobs:
  jaxtyc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run jaxtyc check src/ --format github
```

Each shape error appears as an annotation on the exact file and line in the PR:

```
::error file=src/model.py,line=42,col=4::Shape mismatch in return of `forward`
```

!!! tip "Combine with other linters"
    jaxtyc checks shapes, not types or style. Run it alongside `ruff`, `mypy`, and your test suite in parallel jobs for fast feedback.

---

## JSON output for custom reporting

Use `--format json` when you need structured output for custom CI tooling, dashboards, or Slack notifications.

```yaml
- run: uv run jaxtyc check src/ --format json > jaxtyc-report.json
- name: Upload report
  uses: actions/upload-artifact@v4
  with:
    name: jaxtyc-report
    path: jaxtyc-report.json
```

The JSON schema:

```json
{
  "diagnostics": [
    {
      "file": "src/model.py",
      "line": 42,
      "col": 4,
      "severity": "error",
      "message": "Shape mismatch in return of `forward`\n  Expected: (batch, seq, d_out)\n  Got:      (batch, seq, d_model)",
      "rule": "shape-mismatch"
    }
  ],
  "functions_checked": 12,
  "elapsed_seconds": 2.341
}
```

---

## Configuration in CI

CI runs respect `[tool.jaxtyc]` in your `pyproject.toml`. Common patterns:

=== "Strict (default)"

    Report only errors. This is the default and requires no config.

    ```toml
    [tool.jaxtyc]
    severity = "error"
    ```

=== "Warnings too"

    Catch potential issues before they become errors.

    ```toml
    [tool.jaxtyc]
    severity = "warning"
    ```

=== "Ignore known issues"

    Suppress specific rules while you fix them incrementally.

    ```toml
    [tool.jaxtyc]
    severity = "error"
    ignore_rules = ["trace-error"]
    exclude = ["legacy/**"]
    ```

---

## Full pipeline example

A complete workflow running shape checks, type checks, linting, and tests in parallel:

```yaml
name: CI
on: [push, pull_request]

jobs:
  shapes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run jaxtyc check src/ --format github

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest --tb=short
```

The `shapes` job fails the build if any annotated function has a shape error, and each error appears as an inline annotation on the PR diff.
