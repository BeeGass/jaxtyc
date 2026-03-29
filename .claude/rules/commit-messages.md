---
description: Conventional Commits format with scoped headers and issue references
---

# Commit Messages

Write high-quality, detailed commit messages following Conventional Commits.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Header

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `build`, `ci`, `chore`
- **Scope**: the module or area affected (e.g., `analyzer`, `lsp`, `cli`, `types`, `mux`, `config`)
- **Subject**: imperative mood, lowercase, no period, concise but specific

## Body (required for non-trivial changes)

- Explain *what* changed and *why*, not just *how*
- Describe the problem being solved
- Describe the approach taken and any trade-offs or alternatives considered
- Separate from header with a blank line
- Wrap at 72 characters

## Footer

- Reference issues: `Closes #123`, `Fixes #45`
- Note breaking changes: `BREAKING CHANGE: <description>`

## Example

```
feat(analyzer): detect cross-function shape mismatches via callee tracing

When a function calls another jaxtyping-annotated function, the caller's
expected output shape may conflict with what the callee actually produces.

Add cross-function tracing in checker.py that compares the callee's
traced output shape against the caller's annotation. This catches
mismatches that single-function tracing misses, such as when an inner
function reshapes a tensor differently than the outer annotation claims.

Closes #42
```
