# Fixture License

All material in this directory (`backend-new/eval/fixtures/`) is **self-authored for the purpose of the aBridgeAI eval framework** and released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (Public Domain Dedication).

## Declaration

- Every fixture file has been written from scratch by aBridgeAI contributors.
- No third-party copyrighted text, audio, or other content has been incorporated.
- No LLM-generated content has been used to author the fixture bodies (LLM-generated text would bias the eval signal).
- The "advanced_system_design.md" fixture is a fictional lecture transcript shape — it represents what an audio-extraction pipeline would produce, but no real lecture was transcribed.

## Why CC0

The eval framework runs LLM calls against these fixtures and may include excerpts in test logs, results JSON, and judge prompts. Releasing fixtures into the public domain removes any ambiguity about how the resulting eval artifacts may be stored, redistributed, or analyzed.

## Adding New Fixtures

When adding a new fixture:

1. Author the content yourself, or use a pre-existing CC0 / public-domain source.
2. Do **not** include text that is copyrighted (textbook excerpts, proprietary tutorials, transcribed talks from conferences with restrictive licenses, etc.).
3. Do **not** generate fixture bodies with an LLM — fixtures are the ground truth the eval framework grades against, so LLM-authored content would create circular signal.
4. Add a YAML frontmatter block with `fixture_id`, `material_type`, `expected_chunks`, `language`, and `license` keys.
5. Update this file's declaration above if the new fixture introduces a different license than CC0.
