# Contributing to jaxtyc

Thank you for your interest in contributing!

See the full contributing guide: **[Contributing Guide](https://beegass.github.io/jaxtyc/contributing/)**

## Quick Start

```bash
git clone https://github.com/BeeGass/jaxtyc.git
cd jaxtyc
uv sync --group dev
uv run pytest --tb=short -q
uv run ruff check . && uv run ruff format --check .
uv run ty check
```
