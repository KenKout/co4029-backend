# aBridgeAI eval framework

Real-LLM quality regression harness for the AI pipelines in `abridgeai.features.*.ai`.

## Trigger

**MANUAL ONLY.** This framework is intentionally NOT part of CI:
- it makes real outbound LLM calls and costs real money;
- it is non-deterministic and would generate flaky CI signals;
- it is meant for spot-checks before/after AI pipeline changes, not continuous validation.

## Hard cost ceiling

Every invocation requires `--budget-usd <float>`. The runner refuses to start if the budget is missing. While running, every LLM call must pass through `Budget.spend()`, which raises `BudgetExceededError` the moment the next call would push spend over the ceiling. There is no override flag.

`--dry-run` skips LLM calls entirely; useful for verifying scenario discovery and cost estimates.

## How to run

```sh
# from backend-new/
uv run python -m eval.run --scenarios all --backend new --budget-usd 5.00
uv run python -m eval.run --scenarios quiz_generation --backend both --judge-model gpt-4o-mini
uv run python -m eval.run --budget-usd 0 --dry-run
```

CLI flags:
- `--scenarios`  comma-separated scenario names, or `all` (default `all`).
- `--backend`    `old` | `new` | `both` (default `new`).
- `--budget-usd` hard ceiling in USD (required).
- `--judge-model` LLM-as-judge model id; must be SMALLER than the subject model so judge stays independent (default `gpt-4o-mini`).
- `--output`     output JSON path (default `eval/results/run-<timestamp>.json`).
- `--dry-run`    skip LLM calls; print scenarios and estimated cost.

## Output schema

Each run writes a JSON document of the form:

```json
{
  "run_id": "20260517T120000Z",
  "started_at": "2026-05-17T12:00:00+00:00",
  "finished_at": "2026-05-17T12:03:21+00:00",
  "budget_usd": 5.00,
  "spent_usd": 1.27,
  "backend": "new",
  "judge_model": "gpt-4o-mini",
  "results": [
    {
      "scenario_name": "quiz_generation_basic",
      "inputs": { "...": "..." },
      "outputs_old": null,
      "outputs_new": { "...": "..." },
      "judge_scores_old": null,
      "judge_scores_new": { "faithfulness": 0.9, "coverage": 0.8 },
      "cost_breakdown": [
        { "stage": "subject", "role": "completion", "tokens": 1200, "cost_usd": 0.012 }
      ],
      "latency_ms_old": null,
      "latency_ms_new": 4321,
      "errors": []
    }
  ]
}
```

Schema mirrors `eval.runner.ScenarioResult` plus run-level metadata. Scoring keys (`faithfulness`, `coverage`, etc.) are scenario-defined and ship in T8.3.

## Adding scenarios

See `scenarios/README.md`. T8.1 ships scaffold only; T8.2 ships fixtures + scenario YAMLs.

## Judge independence rule

The judge model MUST be a strictly smaller / cheaper class than the subject model (e.g. judge `gpt-4o-mini` for subject `gpt-4o`). This prevents the same model from grading itself and dampens self-preference bias. The runner does not enforce the inequality automatically; the operator is responsible for picking models that respect it.

## Results retention

`results/run-*.json` is gitignored (see `backend-new/.gitignore`). Only baselines explicitly named `results/baseline-*.json` are committed.
