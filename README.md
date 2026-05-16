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
