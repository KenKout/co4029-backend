# Run results

Per-run outputs: `run-<timestamp>.json` (gitignored) and `baseline-*.json` (committed).

Schema mirrors `eval.runner.ScenarioResult` plus run-level metadata (run_id, started_at, finished_at, budget_usd, spent_usd, backend, judge_model, dry_run, results). See top-level `eval/README.md` for the full shape.

`run-*.json` is treated as ephemeral diagnostic output. Promote a run to a baseline by copying it to `baseline-<descriptive>.json` and committing.
