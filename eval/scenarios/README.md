# Scenarios

Scenario YAMLs describe a single eval case: input fixture, expected capability (quiz_generation, interview_generation, knowledge_graph_extraction, ...), judge criteria, and cost estimate.

T8.1 ships no scenarios. T8.2 will populate this directory.

## Schema (T8.2 will finalize)

```yaml
name: quiz_generation_basic
capability: quiz_generation
inputs:
  material_path: fixtures/quiz/textbook_excerpt.md
  num_questions: 5
estimated_cost_usd: 0.05
judge:
  prompt: judges/prompts/quiz_faithfulness.j2
  criteria:
    - faithfulness
    - coverage
    - difficulty_alignment
```

The runner discovers `*.yaml` files here, optionally filtered by `--scenarios <name>,<name>`. Filenames (without extension) act as scenario ids and must match `[a-z][a-z0-9_]*`.
