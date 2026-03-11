# jaxtyc development tasks

# Default: list available recipes
default:
    @just --list

# Install jaxtyc as a uv tool
tool-install:
    uv tool install --editable . --force

# Bundle the VS Code extension
vscode-bundle:
    cd editors/vscode && npm run bundle

# Package the VS Code extension as .vsix
vscode-package: vscode-bundle
    rm -f editors/vscode/jaxtyc-*.vsix
    cd editors/vscode && npx @vscode/vsce package --allow-missing-repository

# Install the VS Code extension and reload
vscode-install: vscode-package
    code --install-extension editors/vscode/jaxtyc-$(node -p "require('./editors/vscode/package.json').version").vsix --force
    @echo "Reload VS Code: Ctrl+Shift+P → Developer: Reload Window"

# Full pipeline: install uv tool, bundle, package, install extension
vscode-update: tool-install vscode-install

# Run VS Code extension tests
vscode-test:
    cd editors/vscode && npm test

# Run Python tests
test:
    uv run pytest --tb=short -q

# Run all tests (Python + VS Code)
test-all: test vscode-test

# Lint and format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Type check
typecheck:
    uv run ty check

# Run pre-commit hooks on all files
check:
    prek run --all-files
