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

# Create a clean PR from dev to main (strips AI assistant config files)
pr-to-main:
    #!/usr/bin/env bash
    set -euo pipefail

    # Files/dirs to exclude from main
    EXCLUDE=(
        .claude
        .claude-plugin
        .codex
        .opencode
        CLAUDE.md
        AGENTS.md
    )

    BRANCH="release/$(date +%Y%m%d-%H%M%S)"

    echo "Creating clean release branch: $BRANCH"
    git checkout -b "$BRANCH" dev

    # Remove AI config files
    REMOVED=false
    for path in "${EXCLUDE[@]}"; do
        if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
            git rm -rf "$path"
            REMOVED=true
        fi
    done

    if [ "$REMOVED" = true ]; then
        git commit -m "chore: remove AI assistant config files for main"
    fi

    git push -u origin "$BRANCH"
    gh pr create --base main --head "$BRANCH" \
        --title "Merge dev into main" \
        --body "$(cat <<'EOF'
    ## Summary

    Merge latest changes from `dev` into `main`.

    AI assistant configuration files (`.claude/`, `.codex/`, etc.) have been
    stripped from this branch. They remain on `dev` for development use.
    EOF
    )"

    echo ""
    echo "PR created. After merge, clean up with:"
    echo "  git checkout dev && git branch -d $BRANCH"
