# Contributing to jaxtyc

Thank you for your interest in contributing!

See the full contributing guide: **[Contributing Guide](https://beegass.github.io/jaxtyc/contributing/)**

## Quick Start

```bash
git clone https://github.com/BeeGass/jaxtyc.git
cd jaxtyc
uv sync --group dev
prek install && prek install --hook-type commit-msg
```

## Running Checks

```bash
uv run pytest --tb=short -q
uv run ruff check . && uv run ruff format --check .
uv run ty check
```

## Branch Workflow

- Branch from **`dev`** (not `main`)
- Submit PRs against **`dev`**
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
