# Adaptive Interviewer — Staged Rollout Runbook

How to take the adaptive interviewer from OFF to fully live, one controlled step
at a time, and how to detect + reverse a bad stage. It ties together the flag
matrix (`abridgeai.core.config.Settings`) and the rollback-signal evaluator
(`abridgeai.features.interviews.realtime.voice_report`, Slice 6).

The guiding principle: **every stage is reversible by flipping one env var, and
every stage has an objective pass/fail signal derived from real traffic.** No
stage advances on vibes.

---

## 0. The flag matrix (what each control does)

All flags are `Settings` fields read from the environment (`.env`), consumed by
both the backend and the voice-agent process. Restart both after any change:

```
pm2 restart abridgeai-backend abridgeai-interview-agent
```

| Env var | Default | Role |
|---|---|---|
| `ADAPTIVE_INTERVIEWER_ENABLED` | `false` | **Master / kill switch.** OFF → every mode runs legacy, regardless of per-mode flags. This is the one-flip rollback for the whole feature. |
| `ADAPTIVE_INTERVIEWER_TEXT_ENABLED` | `true` | Per-mode gate for text. Only matters when the master switch is ON. |
| `ADAPTIVE_INTERVIEWER_HYBRID_ENABLED` | `true` | Per-mode gate for hybrid. |
| `ADAPTIVE_INTERVIEWER_VOICE_ENABLED` | `false` | Per-mode gate for voice. Requires human mic sign-off (see `voice-adaptive-smoke-tests.md` §5) before enabling. |
| `ADAPTIVE_INTERVIEWER_SHADOW_ENABLED` | `false` | Compute adaptive but never drive the student; logs `voice.decision shadow=true`. Only meaningful for a mode NOT already live. |
| `ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT` | `100` | Fraction (0–100) of students who get adaptive on an already-enabled mode, by stable hash of (student_id, config_id). Applied ON TOP of the mode gate. |

Effective per-mode rule: `ENABLED AND <mode>_flag AND (rollout_bucket < PERCENT)`.

The resolver methods are `Settings.adaptive_enabled_for_mode(mode)` (static gate)
and `Settings.adaptive_enabled_for_student(mode, student_id, config_id)` (full
gate incl. percentage). Do not re-implement these anywhere — call them.

---

## 1. The staged sequence

Each stage runs for a defined window (suggest ≥ 50 completed interview turns of
real traffic, or one teaching day, whichever is larger), then you evaluate the
rollback signal (§2) before advancing. Text/hybrid default ON in code, so a
fresh deployment that flips only the master switch is effectively already at
Stage 2 — if you want true staging, set the per-mode flags OFF first.

| Stage | Flags | What students get | Advance when |
|---|---|---|---|
| **0. Dark** | master OFF | 100% legacy | baseline metrics captured |
| **1. Shadow** | master OFF, `SHADOW_ENABLED=true` | 100% legacy; adaptive computed + logged | shadow decisions look sane in the report; no compute errors |
| **2. Text canary** | master ON, text ON, hybrid OFF, voice OFF, `ROLLOUT_PERCENT=10` | 10% of text students adaptive | rollback signal clean over the window |
| **3. Text ramp** | as above, `ROLLOUT_PERCENT=50` then `100` | 50% → 100% of text | rollback signal clean at each step |
| **4. Hybrid** | text 100%, hybrid ON at `10 → 50 → 100` | text live, hybrid ramping | rollback signal clean |
| **5. Voice** | voice ON at `10 → 50 → 100` | all modes live | **requires** mic sign-off first |

`ROLLOUT_PERCENT` is global, not per-mode, so ramp one mode to 100% and confirm
it is healthy before turning the next mode's flag ON (the newly-enabled mode
then inherits the current percent — set percent back down before flipping the
next flag if you want that mode to canary independently).

---

## 2. The rollback signal (objective pass/fail)

Slice 6 emits transcript-free `voice.decision` / `voice.fallback_activated`
events on the LIVE adaptive path (REST + voice) and the aggregator derives
rollback triggers from them. To evaluate a window:

```bash
# From a log export / pm2 dump of the backend + agent processes:
pm2 logs abridgeai-backend abridgeai-interview-agent --lines 20000 --nostream --raw \
  | python -m abridgeai.features.interviews.realtime.voice_report --rollback
```

Output is a `RollbackSignals` JSON object. **Exit code 2 = at least one trigger
breached = pause/roll back the current stage.** Exit 0 = clean, advance.

Thresholds (single source of truth in `voice_report.py`):

| Signal | Trigger | Meaning |
|---|---|---|
| `turn_error_rate_breached` | turn errors / turns started > **1%** | adaptive pipeline is throwing |
| `utterance_fallback_rate_breached` | fallback utterances / adaptive turns > **5%** | LLM utterance layer failing; students see canned text |
| `legacy_fallback_rate_breached` | legacy decisions / all decisions > **5%** | adaptive is silently declining to legacy too often |
| `question_loop_detected` | reserved (defaults false) | follow-up cap already guaranteed by the decision invariants (Slice 1) |

Drop `--rollback` to get the full `VoiceOpsReport` (latency percentiles, action
histogram, adaptive-success rate, shadow histogram) for a deeper look.

Per-session sign-off (e.g. a specific reported bad interview):

```bash
python -m abridgeai.features.interviews.realtime.voice_report --session <session_id> events.jsonl
```

---

## 3. Rollback procedure

Fastest first:

1. **Whole feature:** set `ADAPTIVE_INTERVIEWER_ENABLED=false`, restart backend +
   agent. Every mode instantly reverts to legacy. In-flight sessions are safe —
   the runtime falls back per-turn and a session is never lost.
2. **One mode:** set that mode's flag OFF (e.g. `ADAPTIVE_INTERVIEWER_VOICE_ENABLED=false`).
3. **Dial back exposure without disabling:** lower `ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT`
   (e.g. `100 → 10`). A given student's experience stays consistent because the
   bucket is a stable hash.

None of these require a deploy or migration — they are env flips + a pm2 restart.
Because the gate is evaluated per turn, a mid-interview flip cleanly moves
subsequent turns to legacy without corrupting persisted runtime state.

---

## 4. Safety invariants that make staging safe

- **Fallback ladder:** any adaptive failure (exception, malformed LLM output,
  runtime-state load failure) rolls back its savepoint and runs the legacy
  advance without re-inserting the answer. The student always gets a turn.
- **Idempotency:** each turn carries a `turn_key`; a retry/replay never
  double-inserts an answer or re-runs the pipeline.
- **Flag-gated behaviour is additive:** the new structured response fields
  (`action`, `reason_code`, `pending_confirmation`, `interaction_state`, …) are
  optional; legacy clients ignore them.
- **Final grading is independent:** the post-session evaluator re-judges the
  transcript and is never bound by provisional adaptive coverage/scores, so a
  bad adaptive window cannot corrupt a verdict.
