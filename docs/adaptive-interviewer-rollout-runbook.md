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

---

## 5. Adaptive v2 (real-life interview upgrades)

v2 layers seven human-interviewer behaviours on top of the v1 adaptive engine.
It is a **second staged rollout that only begins once v1 is at 100% on a mode**
and healthy per §2. All v2 behaviour is additive and flag-gated; with every v2
flag OFF the adaptive path is byte-for-byte v1.

### 5.1 The v2 flag matrix

All flags are `Settings` fields (`.env`), read by backend + agent. Restart both
after any change. Every v2 flag requires the v1 master switch
(`ADAPTIVE_INTERVIEWER_ENABLED`) ON **and** the per-mode gate ON for that mode —
v2 rides inside the v1 adaptive path, so if v1 is off for a mode, v2 is off too.

| Env var | Default | Behaviour |
|---|---|---|
| `ADAPTIVE_INTERVIEWER_V2_ENABLED` | `false` | **v2 master / kill switch.** OFF → every sub-flag below is inert; the engine runs exactly as v1. One-flip v2 rollback. |
| `ADAPTIVE_V2_PHASES_ENABLED` | `false` | Real phase progression (opening → warmup → core → deep-probe → closing) with per-phase difficulty bias. |
| `ADAPTIVE_V2_DEPTH_PROBE_ENABLED` | `false` | Probe a strong answer for its ceiling (extend / edge-case) instead of advancing. Consumes the follow-up budget. |
| `ADAPTIVE_V2_HINT_LADDER_ENABLED` | `false` | Escalating hints + non-repeating rephrasing on the same question (via the adaptive decision path). |
| `ADAPTIVE_V2_AFFECT_ENABLED` | `false` | Detect candidate affect (nervous/terse/rambling/confident) and warm the utterance TONE only. |
| `ADAPTIVE_V2_CROSS_TURN_ENABLED` | `false` | Feed the candidate's own prior claims into answer analysis to catch cross-turn contradictions. |
| `ADAPTIVE_V2_PER_OUTCOME_DIFFICULTY_ENABLED` | `false` | Per-outcome competence (EWMA) calibrates question difficulty per topic instead of one global level. |
| `ADAPTIVE_V2_RICH_CLOSING_ENABLED` | `false` | Closing sub-sequence: self-reflection prompt → invite candidate questions → graceful sign-off. |

Resolver: `Settings.adaptive_v2_feature_enabled(mode, feature)` — returns True
only when the v1 gate for `mode` is on AND the v2 master is on AND that
sub-flag is on. Never re-implement; always call it.

### 5.2 Shadow-first sub-flag order

Bring sub-features up **one at a time, in this order**, each in shadow for a
window (compute + log, don't drive — reuse `ADAPTIVE_INTERVIEWER_SHADOW_ENABLED`
on a mode not yet live, or enable on a low `ROLLOUT_PERCENT` canary), then live:

1. `phases` — foundational; the phase drives depth-probe and closing eligibility.
2. `depth_probe` — verify `test_interview_decision_invariants` loop caps hold.
3. `hint_ladder`
4. `affect` — phrasing-only; lowest risk.
5. `cross_turn`
6. `per_outcome_difficulty`
7. `rich_closing` — last; it changes how sessions end.

Evaluate the §2 rollback signal between each. The new v2 actions
(`extend_answer`, `probe_edge_case`, `prompt_self_reflection`,
`invite_candidate_questions`, `answer_candidate_question`) surface in the report
action histogram automatically; because depth probes and hints consume the
follow-up budget, they cannot trip `question_loop_detected`.

### 5.3 v2 rollback

Fastest first — all env flips + `pm2 restart abridgeai-backend abridgeai-interview-agent`:

1. **All of v2:** set `ADAPTIVE_INTERVIEWER_V2_ENABLED=false` → the engine
   reverts to v1 (which, via its own master switch, can revert to legacy).
   In-flight sessions are safe: the per-turn fallback ladder is unchanged.
2. **One sub-feature:** set that sub-flag OFF (e.g.
   `ADAPTIVE_V2_RICH_CLOSING_ENABLED=false`).

### 5.4 v2 safety invariants

- **Additive + gated:** every v2 field on the runtime state is optional with a
  safe default; an old persisted session loads with neutral v2 values.
- **Deterministic core:** all v2 policy (phase, affect, competence, closing
  sub-state) is pure and unit-tested; the LLM only ever refines phrasing.
- **Evaluator independence still holds:** competence estimates, claims, affect
  and phase are provisional signals and never bind the final verdict.
- **Loop protection preserved:** depth probes / hints consume the follow-up
  budget, so the Slice 1 decision invariants continue to cap follow-ups.
