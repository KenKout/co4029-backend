# Interview prompt-injection red-team corpus

A versioned, labeled dataset of prompt-injection attempts (and benign-adversarial
answers) used to **measure** the interview security guard's coverage and to gate
changes against regressions.

## Why this exists

Prompt injection is an open problem — no filter is complete. This corpus turns
"we think it's covered" into "we measured N% on this vector," and lets every
rule/prompt change be proven a net improvement with no regression (see
`../test_corpus_coverage.py` and `scripts/security_replay.py`).

## File layout

Each `*.json` file is a list of cases for one bucket. `loader.py` merges them.

| File | Bucket | Targeted by |
|---|---|---|
| `covered_input.json` | Attacks the current rules already catch | Phase 0 (baseline) |
| `benign.json` | Legitimate academic answers that *look* scary | Phase 0 (false-positive guard) |
| `other_language.json` | Injections in languages other than EN/VI | Phase 1.1 |
| `delimiter.json` | Fake role/delimiter/format markers | Phase 1.2 |
| `encoding.json` | ROT13 / URL / leetspeak / reversed / base64 / hex | Phase 1.3 |
| `multiturn.json` | Split payloads assembled across turns | Phase 1.4 |
| `social_engineering.json` | Authority spoofing, urgency plea, jailbreak personas | Phase 4 |
| `output_leakage.json` | Proposed AI outputs that leak protected content | Phase 3 |

## Case schema

```json
{
  "id": "unique-stable-slug",
  "text": "the student utterance (or, for output_leakage, the proposed AI text)",
  "lang": "en | vi | es | zh | ar | de | fr | ru | mixed | n/a",
  "layer": "input | output",
  "expected_category": "one of SecurityCategory values (input) or protected category (output)",
  "expected_block": true,
  "expected_action": "optional: refuse_and_redirect | warn_and_redirect | end_and_flag | allow | ...",
  "tags": ["freeform", "labels"],
  "target_phase": "0 | 1.1 | 1.2 | 1.3 | 1.4 | 2 | 3 | 4",
  "status": "covered | gap",
  "notes": "why this case matters / boundary being tested"
}
```

- `status: "covered"` — expected to pass against the CURRENT code (baseline).
- `status: "gap"` — a known gap a later phase is expected to close. The suite
  reports these as `xfail`-style expected gaps until the owning phase lands,
  then flips them to `covered`.

## Invariants every case must respect

- No case's `text` is ever decoded/executed by the guard — canonicalization is
  classify-only.
- Output-leakage cases assert the guard blocks/falls back; the secret text is
  never returned.
- Benign cases assert the guard does NOT block (false-positive protection is a
  first-class metric, not an afterthought).
