# Scenario fixtures

Static inputs (course materials, transcripts, prompts) used by scenario YAMLs in `../scenarios/`. Keep fixtures small and committed; large media fixtures should be referenced by URL or generated lazily.

T8.1 ships no fixtures. T8.2 will populate this directory with inputs for the first batch of scenarios (quiz generation, interview question generation, knowledge graph extraction, etc.).

## How to add a fixture

1. Place the file under `eval/fixtures/<scenario_family>/<descriptive_name>.<ext>`.
2. Reference it from a scenario YAML via a path relative to `eval/`.
3. Keep the file under 50 KB if possible; commit larger fixtures only if they exercise a specific edge case that smaller fixtures cannot.
