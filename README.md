# abridgeai-backend (restructured)

Feature-first restructure of the aBridgeAI backend, scaffolded as a sibling to `backend/`. This package will gradually replace `backend/app/` per the migration plan at `.sisyphus/plans/backend-restructure.md`. Until that plan completes, `backend/` remains the source of truth — code here is a work-in-progress and not yet wired to the application entry point.

## Secret scanning (gitleaks)

A gitleaks config lives in **two** locations because only `backend/` and `backend-new/` are git roots (the parent `co4029_projects/` directory is not a repo):

- `../.gitleaks.toml` — canonical copy at the workspace root (covers all sibling repos)
- `./.gitleaks.toml` — committed copy inside this repo so CI/pre-commit hooks find it

Both files are kept in sync. To scan before pushing, from this directory:

```bash
gitleaks detect --source . --config .gitleaks.toml --no-git
```

To scan the whole workspace from this directory:

```bash
gitleaks detect --source .. --config ../.gitleaks.toml --no-git
```

Custom rules cover OpenAI (`sk-proj-…`, `sk-…`), Anthropic (`sk-ant-…`), and Garage S3 (`GK<hex>`) keys. Allowlists exclude `tests/fixtures/`, `docs/`, and `.env.example`-style files. See `.gitleaks.toml` for the full ruleset.

## Pre-commit hooks

`.pre-commit-config.yaml` lives in this repo (not the parent workspace, which is not a git repo). The hooks mirror the CI lint/security jobs so violations are caught before they reach a PR.

Install once per fresh clone, from this directory:

```bash
uv sync --extra dev
uv run pre-commit install
```

Run all hooks against the full tree at any time:

```bash
uv run pre-commit run --all-files
```

The config wires 13 hooks across 5 SHA-pinned upstream repos:

- `pre-commit-hooks` v6.0.0: `check-yaml`, `check-toml`, `check-json`, `check-merge-conflict`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files` (`--maxkb=500`), `check-ast`
- `ruff-pre-commit` v0.15.13: `ruff --fix` + `ruff-format`
- `mirrors-mypy` v1.20.2 (matches the `mypy>=1.18,<2` range in `pyproject.toml`): runs `mypy abridgeai` with `additional_dependencies` pinned to the same versions as `uv.lock`
- `bandit` 1.9.4: `bandit -r abridgeai/ -c pyproject.toml`
- `gitleaks` v8.30.1: scoped to this repo's `.gitleaks.toml` (the workspace-root copy is unreachable from inside a git hook)

`gitleaks` covers only this repo, not `backend/` (frozen) or the workspace root. The duplicated `.gitleaks.toml` setup keeps both surfaces auditable.

Skipping hooks for a single commit (`git commit --no-verify`) is discouraged. Per `AGENTS.md`: never skip hooks unless the user explicitly requests it, and the commit message must justify why.
