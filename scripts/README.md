# Lint scripts

`bash scripts/lint-changed.sh` — fast lint of files changed since HEAD.
`bash scripts/lint-changed.sh --all` — full repo lint (slower; CI runs this).

Note: lint-imports always runs on the full graph because it analyzes
cross-module dependencies. ruff and mypy can be scoped to changed files.
