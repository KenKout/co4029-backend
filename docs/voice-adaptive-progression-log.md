# Voice Adaptive Interviewer — Delivery Progression Log

Running log of the 11-step voice-adaptive delivery plan. Newest phases at the
bottom. Each entry records what changed, what was verified (real tool output,
not assertions), and any bug the work surfaced.

**Environment (stable facts)**
- Backend: `/root/co4029/backend` (FastAPI, Python 3.11, PostgreSQL localhost:5433, db=abridgeai)
- Frontend: `/root/co4029/frontend` (Vite + React + TanStack Router + Tailwind v4)
- pm2 processes: `abridgeai-backend` (8000), `abridgeai-frontend` (5173), `abridgeai-worker`, `abridgeai-interview-agent` (LiveKit)
- Backend venv: `/root/co4029/backend/.venv/bin/python`
- Alembic head: `0023_interview_turn_key_idempotency`
- Persona→voice: strict→onyx, neutral→alloy, supportive→shimmer
- **Default safety posture:** `adaptive_interviewer_enabled` OFF; voice mode flag OFF. Instant rollback = flip flag off + `pm2 restart`.
- Agent flag restore method: `pm2 delete <name> && pm2 start ecosystem.config.cjs --only <name>` (avoids stale env merge).

**Required delivery order:** strict harness → mode flags → strict EN/VI runs →
diagnostics → human smoke → playback-aware closing → observability → question
metadata → policy tuning → shadow mode → rollout.

---

## Status at a glance

| # | Step | Status |
|---|------|--------|
| 1 | Strict adaptive harness verification | ✅ Done |
| 2 | Mode-specific feature flags | ✅ Done |
| 3 | Strict EN/VI live runs | ✅ Done |
| 4 | Diagnostic results + latency timeline | ✅ Done |
| 5 | Human smoke-test (EN/VI listening) | 🟡 Prepared — needs human reviewer |
| 6 | Human sign-off record + gate | 🟡 Prepared — needs human reviewer |
| 7 | Question metadata surfacing (type/difficulty in decision event + report) | ✅ Done |
| 8 | Playback-aware closing | ✅ Done |
| 9 | Voice observability | ✅ Done |
| 10 | Shadow mode (compute-only, never drives the student) | ✅ Built (OFF by default; validate under human sign-off) |
| 11 | Controlled rollout (deterministic percentage gate) | ✅ Built (100% default = no behaviour change; gated on human sign-off) |

**The single most important open number:** decision latency **p50 ≈ 10 s**
(STT-final → decision-ready, all adaptive LLM stages + phrasing) against the
local gateway. A student waits ~10 s after speaking before the agent replies.
This must be reviewed and accepted (or mitigated) in the human sign-off before
any rollout — see §6.2 of `voice-adaptive-smoke-tests.md`.

---

## Phase: Steps 1–3 — strict harness, mode flags, strict EN/VI runs

**Goal:** prove the adaptive voice path can be verified under the harness with a
source of truth that can't be faked by utterance text, gate voice behind its
own flag, and get both languages passing live.

**Delivered**
- Strict verification in `scripts/voice_harness/run_harness.py`: strict
  `ScenarioResult` fields, `_strict_adaptive_ok`, `_strict_failure_reason`,
  extended `_collect_db_signals`, `--require-adaptive` CLI + `REQUIRE_ADAPTIVE_VOICE` env, verdict wiring.
- Mode-specific flags in `abridgeai/core/config.py`: global master switch AND
  per-mode flags; voice defaults OFF. Resolver `adaptive_enabled_for_mode(mode)` = `global AND mode_flag`.
- Strict source of truth = persisted state, NOT utterance text: AI turns tagged
  `metadata_json.kind="adaptive"` carrying `action` / `reason_code` /
  `utterance_status`, plus the `interview_runtime_states.state_version` counter.

**Bug found & fixed (strict EN first run FAILED, fallback 3/3):**
`utterance_system.j2` was rendered with no args but the template uses
`{{ language }}` → `UndefinedError` → every turn silently fell back to the
deterministic template. Fixed in `orchestrator/utterance_logic.py` (~line 71):
pass `language=(language or "en")` to the system-prompt render. Regression test
`tests/unit/test_utterance_language_render.py` (EN+VI).

**Verified (real runs)**
- Strict EN — PASS: `decision_count=3`, actions `[ask_for_example, repeat_question, transition_topic]`, `state_version=3`, `fallback_count=0`.
- Strict VI — PASS: `decision_count=3`, first action `redirect_to_topic` (flagged as a VI STT-quality concern for human scrutiny), `fallback_count=0`.
- Test suites: strict-logic (6) + strict-wiring (4) + flag-matrix (8) + adaptive-step (18) + utterance-render (2) all pass; ruff/mypy clean on changed files; agent restored to safe default; zero leaked DB/LiveKit resources.

---

## Phase: Step 4 — diagnostics + latency timeline

**Goal:** expose an honest event timeline + latency baselines from the harness,
populated even on timeout/error.

**Delivered** (`run_harness.py` + `room_client.py`)
- `room_client.py`: wall-clock UTC timing hooks — room join, agent join,
  per-turn student audio start/end, agent audio first-frame / first-frame-after-prompt / last-frame, disconnect.
- `HarnessTimeline` + `LatencyMetrics` dataclasses, `result_schema_version=1`,
  `selected_question_ids`, `ai_turn_committed_at` added to DB signals.
- Latencies computed in the `finally` block so they populate even on
  timeout/error. Warnings-only — no thresholds enforced (baseline collection).
- Every metric carries a `notes` entry: client-observable vs DB-proxy vs
  agent-internal(null).

**Measurement flaw the baseline caught:** first run reported
`end_of_speech_to_agent_audio_s = 0.005 s` — deceptive, because the agent
publishes a *continuous* audio track, so "first frame after prompt" just catches
the next streamed frame. Replaced the headline turn-latency with
`end_of_speech_to_decision_committed_s` (DB proxy: last student speech end → last
AI-turn commit), kept the frame-based metric but flagged it **UNRELIABLE** in
code, docs, and `notes`. Re-ran: honest metric ≈ **1.8 s**.

**Verified:** 2 live EN baselines (strict-PASS); new
`tests/unit/test_voice_harness_timeline.py` (helper math, timeline assembly,
missing-event tolerance, schema-version JSON serialization). Also fixed a test
hermeticity bug — autouse fixture clearing `ADAPTIVE_INTERVIEWER_*` env so the
flag-matrix tests aren't polluted by the vars used to run the live agent. 54
pass; ruff clean; DB + LiveKit cleaned; agent safe.

---

## Phase: Step 9 — voice observability

**Goal:** measure what the harness structurally can't — the agent-internal
per-turn latency and decision flow — via structured events, plus an offline
operational report.

**Delivered**
- `realtime/observability.py`: 16 `voice.*` events over the existing structlog
  pipeline. Privacy-safe (no raw transcripts — only `*_chars` lengths + control
  signals). Never raises. Correlated per turn by a `turn_id`. No LiveKit import
  (usable from API process + unit tests).
- Instrumented three layers: `agent.py` (dispatch language, room-join ok/fail,
  disconnect), `session_runtime.py` (turn_started/STT-final, turn_completed +
  decision latency, tts_started/completed, turn_error), `orchestration_bridge.py`
  (decision action/reason/selected-q/state_version, fallback, closing_emitted,
  default_closing_suppressed, session_submitted, evaluation_enqueued).
- `realtime/voice_report.py`: pure aggregator over event dicts + tolerant JSONL
  reader → adaptive success rate, legacy fallback rate, utterance-fallback rate,
  decision/TTS latency p50/p95, room-join failure rate, turn-error rate,
  disconnect rate, per-action counts.

**Two real bugs caught**
1. `emit` double-passed `event` (`logger.info(event, event=event, ...)`) → `TypeError`, swallowed by `contextlib.suppress`. In production **every voice event would have silently vanished.** Unit test caught it before the live run; fixed by stamping the queryable field as `event_type`.
2. Aggregator returned 0 events on real logs — pm2/structlog wraps the event dict as a Python-repr string inside a `message` field, not top-level JSON. Fixed `parse_jsonl` to recover that shape (json → `ast.literal_eval` fallback) + regression test with the exact production format.

**The number Step 9 was built to expose:** live EN run measured **decision
latency p50 ≈ 10.2 s** — the STT-final → decision-ready span inside the agent
(all four adaptive LLM stages + phrasing). The harness's DB proxy reported ~0.9 s
because it measures a narrower span.

**Verified:** live EN run, 13 events emitted, aggregator produced a correct
report from the real agent log (2 adaptive decisions, 100% adaptive rate, real
p50/p95). 12 observability tests + 54 in the full sweep; ruff/mypy clean; agent
safe; 0 leaks.

---

## Phase: Steps 5–6 — human sign-off preparation

**Goal:** turn the human listening sign-off from "it felt fine" into an
objective, repeatable procedure backed by the Step 9 events. (Cannot complete
the actual listening — that needs a human, and a native VI speaker for §5B.)

**Delivered**
- `voice_report.py`: `--session <id>` filter + argparse CLI + `filter_by_session`
  helper, so a reviewer gets metrics scoped to their own session.
- `Makefile`: `voice-signoff-report SESSION=<id>` target (pipes agent log →
  aggregator).
- `docs/voice-adaptive-smoke-tests.md` rewritten for sign-off:
  - §5.0 "How to run a sign-off session" (enable flags → note session id → run →
    produce report → restore safe default) + a table mapping each by-ear check to
    the report field that corroborates it.
  - Per-test "report expectations" for §5A (EN), §5B (VI), §5C (barge-in/closing).
  - §6.1 sign-off record template (copy-paste block for the release ticket).
  - §6.2 latency acceptance rubric (≤3 s conversational / 3–6 s needs a "thinking"
    cue / >6 s don't launch wide without mitigation) — the gate now *requires*
    recording this decision.

**Verified:** `make voice-signoff-report SESSION=…` works end-to-end on the real
log; 14 observability tests (added filter + CLI tests); 46 in the full sweep;
ruff/mypy clean; agent safe.

**Still needs a human:** §5A/5B/5C listening sessions + the §6.2 latency
acceptance decision. These are the true gate before rollout.

---

## Phase: Step 8 — playback-aware closing

**Goal:** ensure the closing utterance is heard in full and can't be cut short.

**Two problems found**
1. Shutdown could fire mid-closing — `ctx.shutdown()` ran right after
   `session.say(closing)`; nothing guaranteed playout had finished.
2. The **adaptive** closing was interruptible — only the canned fallback remark
   used `allow_interruptions=False`; the adaptive closing (now the common path)
   used the session default, so it could be cut short.

**Delivered** (`session_runtime.py` + `observability.py`)
- Force the closing non-interruptible on any finished turn; non-final
  probes/advances keep the session default (unchanged behavior).
- Playback-aware shutdown: capture the closing `SpeechHandle`, `await
  wait_for_playout()` bounded by a 30 s timeout, THEN `ctx.shutdown()`.
- New `voice.closing_playout` event (`completed` / `timed_out` / `playout_ms`).

**Verified (live, reached a real closing via an explicit end-request answer):**
`begin_closing` fired → `closing_emitted` (adaptive, 99 chars) →
`closing_playout completed=true, timed_out=false` → clean shutdown;
`default_closings_suppressed=1` (no double closing), `sessions_submitted=1`,
`evaluations_enqueued=1`, zero errors. Room log confirms the agent finished the
closing before disconnect.

Honest note: `playout_ms ≈ 0` because `await say()` already blocks for playout in
livekit-agents 1.5.x — the explicit wait is belt-and-suspenders against SDK
drift; the behavioral win is the non-interruptible closing.

**Tests:** 5 new (playout happy-path, timeout guard, None-handle, error-swallow,
non-interruptible-closing regression). 43 pass; ruff/mypy clean; agent safe; 0
leaks.

---

## Phase: Steps 7, 10, 11 — question metadata, shadow mode, rollout gate

**Goal:** finish all remaining pre-sign-off engineering. Everything here is
OFF/no-op by default — no student-facing behaviour changes until the human
sign-off flips a flag.

### Step 7 — Question-metadata surfacing
The `voice.decision` event now carries the SELECTED question's
`selected_question_type` / `selected_question_difficulty` (populated only on an
advance — probe/clarify/repeat/closing turns reuse the current question and
leave them null). The offline ops report (`voice_report.py`) aggregates these
into `selected_question_type_counts` / `selected_question_difficulty_counts`, so
sign-off can see WHAT KIND of questions the adaptive brain is choosing without
reading transcripts (privacy contract preserved — still lengths + control
signals only). Metadata is read from the ORM row while attached to the session,
before commit.

### Step 10 — Shadow mode (`ADAPTIVE_INTERVIEWER_SHADOW_ENABLED`, default OFF)
When shadowing a mode that is NOT already live, `take_session_step` serves the
student the **legacy** path, then ALSO computes the adaptive decision purely for
comparison inside a savepoint that **always rolls back** — so shadow can never
persist an AI turn, bump `state_version`, or otherwise leak into the live
session (identical isolation to the proven adaptive-fallback path). The shadow
decision is emitted as `voice.decision` with `shadow=true`; the aggregator
counts these separately (`shadow_decisions` / `shadow_action_counts`) so they
never inflate the live adaptive/legacy rates. `shadow_enabled_for_mode()` is a
no-op for any mode that is already statically adaptive-enabled — you can't
shadow a path that's already driving. Best-effort + fully swallowed: a shadow
failure can never affect the interview the student is actually taking.

### Step 11 — Controlled rollout (`ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT`, default 100)
`adaptive_enabled_for_student(mode, student_id, config_id)` layers a
deterministic percentage gate ON TOP of the static mode gate. The bucket is a
salt-independent SHA-256 hash of `(student_id, config_id) % 100`, so a given
student's experience is stable across turns/attempts and never flips
mid-interview. `percent=100` (default) = everyone who passes the static gate is
enabled → **identical to prior behaviour**; `percent=0` = nobody, without
touching the mode flag. `take_session_step` now calls the per-student gate
instead of the raw mode gate.

**Tests:**
- `tests/unit/test_adaptive_rollout_shadow.py` — 12 tests: rollout determinism,
  0/100 boundaries, field clamping [0,100], ~50% population split, shadow
  precedence (no-op when live / independent of master switch / unknown mode).
- `tests/unit/test_voice_observability.py` — +2 tests: question-metadata
  histograms populated on advances, empty when no advance (never fabricated).
- `tests/integration/test_interview_adaptive_step.py` — +2 real-DB tests: shadow
  serves legacy AND persists zero AI turns / zero runtime-state rows (answer
  still recorded once); shadow OFF by default runs pure legacy.

**Verified:** 46 interview adaptive/realtime tests pass (incl. the shadow real-DB
tests); 24 unit config/observability tests pass; 437 in the broad interview+
realtime sweep pass (see "Known pre-existing failures" below); ruff clean on all
changed modules.

**Known pre-existing failures (NOT caused by this work):**
- `test_no_god_files_in_interviews` — `routers/authoring.py` is 820 LOC at HEAD,
  already over the 800 cap; untouched by Steps 7/10/11 (my largest file,
  `services/taking.py`, is 798, under the cap).
- 3 collection ImportErrors in `test_courses_authoring_queries`,
  `test_courses_routers_deps_snapshot`, `test_lesson_gating_enforcement` —
  reference symbols (`get_course_content_authoring`, `_resolve_lesson_to_course`,
  `InterviewPassRequiredError`) unrelated to this work.
- mypy 1.20.2 crashes with an INTERNAL ERROR on this Python 3.14 env (typeshed
  incompat) regardless of file — Pyright (LSP) is clean on all edits.

**How to use these before rollout:**
```
# Shadow voice while it's still off for real (collect adaptive-vs-legacy signal):
ADAPTIVE_INTERVIEWER_SHADOW_ENABLED=true         # voice stays legacy for students
# then read shadow_decisions / shadow_action_counts from the ops report.

# Canary a mode to 10% of students after sign-off:
ADAPTIVE_INTERVIEWER_ENABLED=true
ADAPTIVE_INTERVIEWER_VOICE_ENABLED=true
ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT=10
```

---

## Observed dev latency baselines (reference, not SLOs)

| Metric | Value | Source |
|--------|-------|--------|
| room join → agent join | ~1.6–2.6 s | harness (client-observable) |
| end-of-speech → decision committed | ~0.9–1.8 s | harness (DB proxy) |
| **decision latency p50 (STT-final → decision-ready)** | **~10 s** | **agent observability event** |
| decision latency p95 | ~11–11.6 s | agent observability event |
| TTS p50 / p95 | ~5–7 s / ~9–14 s | agent observability event |

Single dev runs against the local gateway. Collect a spread across EN/VI and
load before setting any threshold.

---

## Immediate next actions

All pre-sign-off ENGINEERING is now complete (Steps 1–4, 7, 8, 9 done; 10 and 11
built and OFF by default). The remaining work is the human gate — code cannot
substitute for a person listening to the audio:

1. **Human sign-off (the only remaining blocker):** run §5A (EN), §5B (VI —
   native speaker), §5C (barge-in/closing); attach the per-session ops reports;
   record the §6.2 latency acceptance decision (~10 s p50 dev baseline).
2. **Optional pre-sign-off signal:** turn on shadow mode
   (`ADAPTIVE_INTERVIEWER_SHADOW_ENABLED=true`) to collect adaptive-vs-legacy
   comparison data on real traffic with zero student-facing risk — read
   `shadow_decisions` / `shadow_action_counts` from the ops report.
3. **After sign-off — controlled rollout:** flip the mode flag on and start with
   `ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT=10` (or similar), widening as the ops
   report stays clean.

Do NOT flip a mode to 100% before the human sign-off: rolling out an experience
no human has validated is premature. Shadow mode + the percentage gate exist
precisely so you never have to make that jump blind.
