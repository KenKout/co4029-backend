#!/usr/bin/env bash
# scripts/lint-changed.sh — run lint tools only against files changed since main HEAD
# Usage: bash scripts/lint-changed.sh [--all]
#   --all   bypass change detection and run on whole tree (slow, full check)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ "${1:-}" = "--all" ]; then
    echo "==> ruff check (all)"
    uv run ruff check .
    echo "==> mypy (all)"
    uv run mypy abridgeai/
    echo "==> lint-imports (all)"
    uv run lint-imports
    exit 0
fi

# Detect changed files vs HEAD
CHANGED=$(git diff --name-only HEAD -- '*.py' 2>/dev/null || true)
STAGED=$(git diff --cached --name-only -- '*.py' 2>/dev/null || true)
UNTRACKED=$(git ls-files --others --exclude-standard -- '*.py' 2>/dev/null || true)
ALL_CHANGED=$(echo "$CHANGED $STAGED $UNTRACKED" | tr ' ' '\n' | sort -u | grep -v '^$' || true)

if [ -z "$ALL_CHANGED" ]; then
    echo "No Python changes detected."
    exit 0
fi

echo "==> ruff check (changed files)"
echo "$ALL_CHANGED" | xargs uv run ruff check

# Filter mypy to abridgeai/ files only
ABRIDGEAI_CHANGED=$(echo "$ALL_CHANGED" | grep '^abridgeai/' || true)
if [ -n "$ABRIDGEAI_CHANGED" ]; then
    echo "==> mypy (changed abridgeai files)"
    echo "$ABRIDGEAI_CHANGED" | xargs uv run mypy
fi

# lint-imports MUST run on whole tree (it analyzes the dependency graph globally)
echo "==> lint-imports (whole graph - cannot be scoped)"
uv run lint-imports

echo "==> done"
