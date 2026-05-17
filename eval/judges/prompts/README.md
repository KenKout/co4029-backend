# Judge prompts

Jinja2 templates rendered with scenario inputs + outputs to produce LLM-as-judge prompts.

Naming convention: `<criterion>.j2`, e.g. `faithfulness.j2`, `coverage.j2`, `difficulty_alignment.j2`.

Each prompt MUST instruct the judge model to return a JSON object with numeric scores in `[0, 1]` (one key per criterion the prompt covers). The runner parses the JSON and feeds it into `ScenarioResult.judge_scores_*`.

T8.1 ships no prompts. T8.3 ships the first batch.

## Independence rule

Judge models MUST be smaller / cheaper than the subject model (e.g. `gpt-4o-mini` for subject `gpt-4o`). The CLI does not enforce this automatically; the operator picks the model via `--judge-model`. Document the chosen pairing in the scenario YAML.
