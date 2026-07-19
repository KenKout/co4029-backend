# Voice Adaptive Interviewer — Manual Smoke-Test Procedures

Phase 18 wired the adaptive interviewer into the LiveKit voice path and threaded
per-session language (EN/VI) end-to-end. Everything up to a live spoken session
is automated and green (336 interview tests, ruff/mypy clean, all processes
boot). **The remaining verification requires a human at a microphone** — a mock
model cannot speak into STT or listen to TTS. This document is that checklist.

Do NOT enable voice adaptive in production until every box in §5 is signed off.

---

## 0. Preconditions & environment

- Backend, worker, voice agent, frontend all `online` in pm2:
  ```
  pm2 status
  # expect: abridgeai-backend, abridgeai-worker, abridgeai-interview-agent, abridgeai-frontend all "online"
  ```
- Voice agent registered with LiveKit cloud (check once):
  ```
  pm2 logs abridgeai-interview-agent --lines 40 --nostream
  # expect a line: {"message": "registered worker", "agent_name": "interview-agent", ...}
  ```
- LiveKit + LLM/STT/TTS credentials present in `.env`:
  `LIVEKIT_WS_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LLM_API_KEY`,
  `LLM_BASE_URL`, `WHISPER_MODEL`. Gateway reachable at
  `http://192.168.1.21:3000/v1`.
- **Feature flag ON for the test window only.** Adaptive is gated by
  `adaptive_interviewer_enabled`. For a staging smoke test set it in the env the
  backend AND the voice-agent process read (both call `take_session_step` /
  `get_settings`), then restart both:
  ```
  # in .env (staging):  ADAPTIVE_INTERVIEWER_ENABLED=true
  pm2 restart abridgeai-backend abridgeai-interview-agent
  ```
  Confirm it took effect:
  ```
  curl -s localhost:8000/openapi.json >/dev/null   # backend up
  # then run the flag probe in §0.1
  ```

### 0.1 Flag probe (no mic needed)
Confirms the running backend actually sees the flag as ON before you spend time
on a live call:
```
cd /root/co4029/backend
.venv/bin/python -c "from abridgeai.core.config import get_settings; print('adaptive:', get_settings().adaptive_interviewer_enabled)"
# expect: adaptive: True
```

### 0.2 Prepare a voice-eligible interview
- A published interview config with `supported_modes` in (`voice`, `hybrid`) and
  at least 3 approved questions across ≥2 outcomes (so selection/advance has
  somewhere to go).
- A student account enrolled in the course, no active cooldown.
- Note the `course slug` + `interview_config_id` for the join URL:
  `/courses/<slug>/interview/<interview_config_id>`.

---

## 4. Semi-automated harness (no mic, runs the full audio loop)

A scripted LiveKit participant lives at `scripts/voice_harness/`. It joins a real
room as a synthetic student, publishes **real spoken audio** (student answers are
synthesized with the same gateway TTS the agent uses, so Whisper STT transcribes
them), and captures every agent TTS utterance to WAV. This exercises the ENTIRE
path — token mint → agent dispatch → STT → orchestration bridge → adaptive brain
→ agent TTS — without a human at a microphone.

**What it proves (and what it does not):**
- ✅ Agent dispatches, joins, and speaks; audio is produced in both directions.
- ✅ Student answers transcribe and drive real adaptive decisions (verified via
  DB: `interview_session_messages` rows + one `interview_runtime_states` row).
- ✅ Per-session language threads to the agent (run with `--language vi`).
- ❌ Subjective audio quality / naturalness — a human still signs off §5.
- ❌ True acoustic barge-in (echo cancellation / mic VAD) — approximated only.

**Critical prerequisites (learned the hard way):**
1. The **agent process** reads `adaptive_interviewer_enabled`, NOT the harness.
   To exercise adaptive you must enable the flag in the env the pm2
   `abridgeai-interview-agent` process reads, then `pm2 restart
   abridgeai-interview-agent`. With the flag off the harness still runs but the
   agent takes the legacy path (`runtime rows: 0`).
2. The student track MUST be published with `source=SOURCE_MICROPHONE` — the
   agent's RoomIO only feeds microphone-sourced tracks into STT. (The harness
   already does this; noted so nobody "simplifies" it away.)
3. Each answer clip needs **trailing silence** so silero VAD endpoints the turn.
   (The harness appends 1.5 s; do not remove it or turns never complete.)

**Run it:**
```
cd /root/co4029/backend
# enable adaptive on the AGENT, then:
INTERVIEW_VOICE_ENABLED=true .venv/bin/python -m scripts.voice_harness.run_harness \
  --language en \
  --answers "A fact table stores quantitative measurements" "It links to dimension tables via foreign keys" \
  --turn-gap 9 --reply-wait 12 \
  --out ./voice-harness-out --cleanup
```
- Omit `--config-id/--student-id` and it provisions a throwaway published
  voice config (+2 outcomes, +2 approved questions, a student). `--cleanup`
  deletes that throwaway data (and only harness-provisioned `vh-` data) at the end.
- `--language vi` runs the Vietnamese path. `--barge-in` starts the first answer
  during agent speech (approximate barge-in only).
- Captured agent audio lands in `--out` as `agent_utterance_*.wav`.
- Exit code 0 + `HARNESS PASS` means ≥1 non-empty agent utterance was captured.

**Verify it actually ran adaptive (not legacy):**
```
.venv/bin/python - <<'PY'
import asyncio
from sqlalchemy import text
from abridgeai.core.db import get_sessionmaker
SID = "<session id printed by the harness>"
async def main():
    async with get_sessionmaker()() as db:
        msgs = (await db.execute(text(
            "SELECT role, left(content_text,70) FROM interview_session_messages "
            "WHERE session_id=:s ORDER BY created_at"), {"s": SID})).all()
        for m in msgs: print(m)
        rt = (await db.execute(text(
            "SELECT count(*) FROM interview_runtime_states WHERE session_id=:s"), {"s": SID})).scalar_one()
        print("runtime rows (1 = adaptive ran, 0 = legacy):", rt)
asyncio.run(main())
PY
```

> This harness was dry-run during development: with the agent flag ON it produced
> a live session with 4 messages (2 student answers transcribed, 2 adaptive AI
> probes: "Could you give a concrete example?" / "Which part … rephrase?") and
> one runtime-state row — confirming the voice→adaptive loop works end-to-end.
> Verify the captured WAV has real speech energy, not silence:
> ```
> .venv/bin/python -c "import wave,audioop; w=wave.open('voice-harness-out/agent_utterance_01.wav'); d=w.readframes(w.getnframes()); print('peak amplitude:', audioop.max(d,2))"
> # expect a peak in the thousands (silence would be ~0)
> ```

### 4.1 Discoverable entry points (Make + pytest)

The harness has one canonical implementation (`scripts/voice_harness/run_harness.py::run_scenario`).
Both the Make targets and the pytest wrapper call it — no logic is duplicated.

**Make targets** (from `backend/`):
```
make voice-harness-en      # English scenario  → voice-harness-out-en/  + result.json
make voice-harness-vi      # Vietnamese scenario → voice-harness-out-vi/ + result.json
make voice-harness         # both, in sequence (VI still runs if EN fails; target fails if either did)
make voice-harness-clean   # rm -rf local captured-audio dirs (safe, local only)
make voice-live-test       # run the opt-in pytest wrapper
```
Override defaults inline, e.g. `make voice-harness-en TURN_GAP=12 EN_ANSWERS='"answer one" "answer two"'`.

**pytest wrapper** — `tests/integration/test_voice_live_harness.py`, marker `voice_live`:
```
RUN_LIVEKIT_VOICE_TESTS=1 .venv/bin/pytest tests/integration/test_voice_live_harness.py \
  -m voice_live -o addopts="" -p no:cov -s -v
```
It is excluded from the normal suite **two ways**: the default `addopts` carries
`-m 'not destructive and not voice_live'`, AND a `skipif` requires
`RUN_LIVEKIT_VOICE_TESTS=1` plus all credentials. A bare `pytest` never collects
it; `pytest -m voice_live` self-skips unless explicitly armed. It therefore
never runs in normal CI.

### 4.2 Machine-readable result

Every run produces a `ScenarioResult` (printed as JSON, and written to `--json <path>`):

| Field | Meaning |
|---|---|
| `ok` | overall pass: agent produced audio AND turns persisted |
| `session_id`, `config_id` | the (throwaway or supplied) IDs |
| `language` | `en` / `vi` |
| `message_count` | persisted `interview_session_messages` rows |
| `runtime_state_count` / `adaptive_ran` | **authoritative** "did the adaptive brain run" (1 row ⇒ yes; 0 ⇒ legacy) |
| `adaptive_actions` | ordered list of adaptive actions taken (`ask_for_example`, `clarify`, …) |
| `agent_utterance_count`, `captured_audio_seconds`, `peak_amplitude` | captured agent TTS (peak > ~200 ⇒ real speech, not silence) |
| `adaptive_flag_during_run` / `adaptive_flag_after_restoration` | flag as this process saw it at start / end — replaces the old ambiguous single `adaptive flag: False` line |
| `adaptive_required` / `adaptive_requirement_met` / `fallback_count` / `state_version` / `decision_count` | strict-mode verification (see §4.5) |
| `selected_question_ids` | next-question ids the adaptive selector chose, in ask order |
| `db_cleanup_status`, `livekit_cleanup_status` | `cleaned`/`deleted`/`failed: …` — cleanup always attempted in `finally` |
| `audio_kept`, `audio_dir` | WAVs kept on failure; deleted on success only with `--delete-audio-on-success` |
| `result_schema_version` | integer; bump signals a breaking result-shape change (currently `1`) |
| `timeline` | wall-clock UTC (ISO-8601) event timeline — see §4.2.1 |
| `latency` | derived latency seconds + per-metric `notes` — see §4.2.1 |

#### 4.2.1 Timeline & latency (baseline collection only)

The `timeline` block records **client-observable** wall-clock UTC events (room
join, agent join, per-turn student audio start/end, agent audio first frame /
end, disconnect) plus `ai_turn_committed_at` (AI-turn DB `created_at` — a proxy
for "decision committed"). Timestamps are UTC ISO-8601 so they line up with DB
rows.

The `latency` block derives seconds from those events. **No aggressive
thresholds are enforced yet** — this phase collects baselines and emits data,
not failures. Each metric carries a `notes` entry stating its provenance:

| Metric | Provenance | Trust |
|---|---|---|
| `end_of_speech_to_decision_committed_s` | DB proxy: last student speech end → last AI-turn `created_at` | **PREFERRED** turn latency. Folds STT + decision + phrasing to persistence; excludes TTS playout. |
| `room_join_to_agent_join_s` | client-observable | reliable (dispatch + join cost) |
| `agent_audio_span_s` | client-observable | reliable (first→last agent frame) |
| `total_scenario_s` | client-observable | reliable (whole run wall-clock) |
| `end_of_speech_to_agent_audio_s` | client-observable frame timing | **UNRELIABLE** — the agent publishes a *continuous* audio track, so "first frame after prompt" catches the next streamed frame (~ms), not the reply onset. Kept for completeness; do not use as turn latency. |
| `stt_final_at`, `decision_completed_at` | agent-internal | **not observable** from the harness (null); use `ai_turn_committed_at` as the proxy |

Observed dev baseline (single EN run, 2 turns, adaptive on, local gateway):
`room_join_to_agent_join_s ≈ 1.6`, `end_of_speech_to_decision_committed_s ≈ 1.8`.
These are a starting reference, not a threshold — collect a spread across EN/VI
and load conditions before setting any SLO.

### 4.3 External services & estimated cost

A live run touches **real, metered** services. Rough per-scenario estimate
(2 student turns, default pacing):

| Service | Usage per scenario | Notes |
|---|---|---|
| **LiveKit Cloud** | 1 room, 2 participants (agent + synthetic student), ~1–2 connected minutes | Billed by participant-minutes; a room is created and **deleted in `finally`**. Requires `LIVEKIT_WS_URL/API_KEY/API_SECRET`. |
| **Gateway TTS** (`/audio/speech`, `tts-1`) | student answer clips + every agent utterance (agent-side TTS) | ~1–2 KB chars total; small but non-zero. |
| **Gateway STT** (Whisper) | transcription of each student clip | ~2 short clips (~4 s each). |
| **Gateway LLM** (adaptive brain) | perception + analysis + utterance calls per turn, ONLY when the agent runs adaptive | 0 tokens if the agent is on the legacy path; a handful of small completions per turn when adaptive. |
| **Postgres** | 1 throwaway org/course/module/config + 2 questions/outcomes + 1 session | deleted in `finally` when `--cleanup`. |

Ballpark: **a few cents of gateway spend + a couple of LiveKit participant-minutes
per scenario.** `make voice-harness` runs two (EN + VI). This is cheap enough for
pre-release smoke runs but should NOT be wired into per-commit CI (cost + external
flakiness + it needs the agent process running).

Combined wall-clock: ~1.5–2 min per scenario at default pacing (`TURN_GAP=9`,
`REPLY_WAIT=12`), bounded by `--scenario-timeout` (default 240 s).

### 4.4 This is not a substitute for human sign-off

The harness proves the **pipeline runs and produces non-empty, correctly-routed
audio**. It does **not** judge audio quality, naturalness, VI pronunciation,
true acoustic barge-in, or the closing experience. §5A (English), §5B
(Vietnamese), and §5C (barge-in & closing) still require a human and remain the
gate before enabling voice adaptive in production.

### 4.5 Voice observability (structured events + operational report)

The live voice path emits compact **structured events** via the shared
structlog pipeline (`abridgeai.features.interviews.realtime.observability`).
Every event is one log line tagged `event=voice.<name>` plus a stable field
set. **No raw transcripts are logged** — only lengths (`*_chars`) and control
signals (action, reason_code, question/outcome ids, latencies, error class).

Events emitted:

| Event | Emitted by | Key fields |
|---|---|---|
| `voice.agent_dispatch` | agent entrypoint | `language` |
| `voice.room_join` | agent entrypoint | `ok`, `error_class` |
| `voice.turn_started` | runtime (STT final) | `turn_id`, `transcript_chars`, `language` |
| `voice.decision` | bridge | `turn_id`, `adaptive`, `action`, `reason_code`, `state_version`, `selected_question_id`, `answer_chars` |
| `voice.fallback_activated` | bridge | `turn_id`, `action` (utterance degraded to template) |
| `voice.turn_completed` | runtime | `turn_id`, `decision_latency_ms`, `will_speak`, `finished` |
| `voice.tts_started` / `voice.tts_completed` | runtime | `turn_id`, `speak_chars`, `tts_ms` |
| `voice.closing_emitted` / `voice.default_closing_suppressed` | bridge | `turn_id`, `adaptive_closing` |
| `voice.closing_playout` | runtime | `turn_id`, `completed`, `timed_out`, `playout_ms` — the closing finished (or timed out) playing out BEFORE shutdown |
| `voice.session_submitted` / `voice.evaluation_enqueued` | bridge | `turn_id` |
| `voice.turn_error` | runtime | `turn_id`, `error_class`, `latency_ms` |
| `voice.disconnect` | agent entrypoint | `reason` |

`turn_id` (a uuid4 minted per student turn) correlates the runtime's I/O events
with the bridge's decision events for the same turn.

**Operational report.** `voice_report` aggregates these events into a compact
rollup — adaptive success rate, legacy fallback rate, utterance-fallback rate,
decision-latency p50/p95, TTS p50/p95, room-join failure rate, turn-error rate,
disconnect rate, and per-action counts. It's a pure function over event dicts
plus a tolerant JSONL reader (handles clean JSON, log-prefixed lines, and the
pm2/structlog wrapper that repr's the event dict inside a `message` field):

```
pm2 logs abridgeai-interview-agent --lines 5000 --nostream --raw \
  | .venv/bin/python -m abridgeai.features.interviews.realtime.voice_report
# or point it at a captured file:
.venv/bin/python -m abridgeai.features.interviews.realtime.voice_report events.log
```

**Observed dev baseline (single EN run, 2 adaptive turns, local gateway):**
`decision_latency_ms` p50 ≈ 10.2 s. This is the **STT-final → decision-ready**
span (all four adaptive LLM stages + phrasing) as measured *inside the agent* —
much larger than the harness's DB-proxy latency (~0.9 s), which measures a
narrower span. This ~10 s brain latency is the single most important number for
the human sign-off and any future SLO: at default gateway speed a student waits
~10 s after speaking before the agent replies. Collect a spread across EN/VI
and load before setting a threshold; the report deliberately enforces none yet.

---

## 5.0 How to run a sign-off session (read first)

Each §5 test below is a **human** listening test. To make the sign-off
objective rather than "it felt fine", every session now produces a machine
report from the structured `voice.*` events (§4.5). Run each session like this:

1. **Enable adaptive voice on the agent** (it is OFF by default) and restart it:
   ```
   # in the abridgeai-interview-agent process env:
   ADAPTIVE_INTERVIEWER_ENABLED=true
   ADAPTIVE_INTERVIEWER_VOICE_ENABLED=true
   pm2 restart abridgeai-interview-agent --update-env
   pm2 logs abridgeai-interview-agent --lines 20 --nostream   # expect "registered worker"
   ```
2. **Note the interview session id** (visible in the URL / Network tab on the
   `realtime-token` POST, or the first `voice.agent_dispatch` log line).
3. **Run the interview** and fill the checklist by ear.
4. **Produce the per-session report** (scopes metrics to just your run):
   ```
   cd /root/co4029/backend
   make voice-signoff-report SESSION=<your-session-id>
   ```
5. **Attach the JSON to the sign-off record** (§6.1) and check it against the
   "report expectations" under each test.
6. **When done, restore the safe default:** flip both flags OFF and
   `pm2 restart abridgeai-interview-agent --update-env` (see §6 Rollback).

The report turns each subjective check into a cross-checkable number:

| You are verifying (by ear) | Report field that corroborates it |
|---|---|
| Adaptive brain actually ran (not legacy) | `adaptive_decisions` > 0, `legacy_decisions` == 0, `adaptive_success_rate` == 1.0 |
| Utterances were genuine, not template fallbacks | `utterance_fallback_rate` == 0.0 (`fallback_activations` == 0) |
| No turn crashed | `turn_errors` == 0, `turn_error_rate` == 0.0 |
| Single clean closing | `closings_emitted` == 1 AND (`default_closings_suppressed` == 1 for adaptive closing) |
| Session finalised + evaluation queued | `sessions_submitted` == 1, `evaluations_enqueued` == 1 |
| Which adaptive actions fired | `action_counts` (e.g. `ask_for_example`, `repeat_question`) |
| **How long the student waited after speaking** | **`decision_latency_ms_p50` / `p95`** — see the latency gate in §6 |

> The report is corroboration, not a substitute: it proves the *control path*
> (what the brain decided, how long it took). Only your ears prove *audio
> quality, naturalness, VI pronunciation, and true barge-in*.

---

## 5A. English voice smoke test

**Goal:** a spoken EN session runs the adaptive brain end-to-end with correct
audio, correct language, and no double-utterance.

Set the browser/app UI language to English before joining (this drives the
`Accept-Language` header on the realtime-token request → dispatch metadata →
agent → `take_session_step(language="en")`).

| # | Step | Expected | Pass? |
|---|------|----------|-------|
| 1 | Join the interview in **voice** mode | Agent connects; you HEAR the first question spoken (not just see it) | ☐ |
| 2 | Give a **complete, correct** answer | Agent acknowledges briefly then either probes OR advances; audio matches on-screen transcript | ☐ |
| 3 | Give a **vague/partial** answer to the next question | Agent asks a follow-up probe ("Could you give a concrete example?" etc.) and does NOT advance | ☐ |
| 4 | Say **"Could you repeat the question?"** | Agent re-speaks the SAME question; no new question selected | ☐ |
| 5 | On an **advance**, listen closely | The question is spoken **exactly once** (not once in the ack/transition and again as the question) | ☐ |
| 6 | Answer remaining questions until closing | Agent speaks a **single** closing utterance (adaptive closing OR canned remark, never both) | ☐ |
| 7 | After closing | Session finalizes; evaluation is enqueued; UI moves to results after the closing was heard | ☐ |

**Backend cross-checks (run during/after the call):**
```
cd /root/co4029/backend
# One AI turn per adaptive turn, tagged adaptive, with action/reason:
.venv/bin/python - <<'PY'
import asyncio, json
from sqlalchemy import text
from abridgeai.core.db import get_sessionmaker
SESSION_ID = "<paste session id>"
async def main():
    async with get_sessionmaker()() as db:
        rows = (await db.execute(text(
            "SELECT role, metadata_json->>'action' AS action, "
            "metadata_json->>'utterance_status' AS utt, left(content_text,60) AS txt "
            "FROM interview_session_messages WHERE session_id=:s ORDER BY created_at"), {"s": SESSION_ID})).all()
        for r in rows: print(r)
        rt = (await db.execute(text(
            "SELECT version, left(state_json::text, 120) FROM interview_runtime_states WHERE session_id=:s"), {"s": SESSION_ID})).all()
        print("runtime_state:", rt)
asyncio.run(main())
PY
```
Expected: alternating `user` / `ai` rows; each adaptive `ai` row has a non-null
`action`; `utterance_status` is `llm` (or `fallback` if the gateway hiccuped —
still acceptable, just note it); exactly one runtime_state row whose `version`
equals the number of committed adaptive turns.

**Report expectations** (`make voice-signoff-report SESSION=<id>`):
`adaptive_decisions` == number of turns you took, `legacy_decisions` == 0,
`utterance_fallback_rate` == 0.0, `turn_errors` == 0, `closings_emitted` == 1,
`sessions_submitted` == 1, `evaluations_enqueued` == 1. `action_counts` should
include `ask_for_example` (step 3) and `repeat_question` (step 4). Record
`decision_latency_ms_p50/p95` — this is the EN latency data point for §6.

**Watch the agent log live for errors:**
```
pm2 logs abridgeai-interview-agent
# red flags: "interview turn failed", "pipeline failed ... legacy fallback",
# any traceback. A one-off "legacy fallback" means adaptive degraded safely —
# note which turn and why; a storm of them means the gateway or DB is unhappy.
```

---

## 5B. Vietnamese voice smoke test

**Goal:** the same, but the interviewer SPEAKS VIETNAMESE.

Set the UI language to **Tiếng Việt** before joining. This must produce
`Accept-Language: vi-*` on the realtime-token request. (You can confirm the
header in the browser Network tab on the `realtime-token` POST.)

| # | Step | Expected | Pass? |
|---|------|----------|-------|
| 1 | Join in voice mode (UI = VI) | First question spoken; interviewer's **acknowledgements/transitions are in Vietnamese** (the question text itself is whatever the author wrote) | ☐ |
| 2 | Answer partially | Vietnamese probe, e.g. "Bạn có thể cho một ví dụ cụ thể không?" | ☐ |
| 3 | Say **"Bạn nhắc lại câu hỏi được không?"** (repeat) | Interviewer re-speaks the question, prefaced in Vietnamese ("Câu hỏi được nhắc lại: …") | ☐ |
| 4 | Advance | Vietnamese transition ("Chúng ta tiếp tục nhé.") then the question, spoken once | ☐ |
| 5 | Reach closing | Vietnamese closing, spoken once | ☐ |
| 6 | Audio quality | TTS voice pronounces Vietnamese acceptably (note any mispronunciation of diacritics; this is a TTS-model quality judgment, log examples) | ☐ |

**Metadata cross-check (confirm language actually reached the agent):**
```
pm2 logs abridgeai-interview-agent --lines 200 --nostream | grep -i "agent starting"
# The dispatch metadata carried language="vi"; the voice.agent_dispatch event
# also records language=vi — confirm in the per-session report's source events.
```

**Report expectations** (`make voice-signoff-report SESSION=<id>`): same control
signals as EN (`adaptive_decisions` > 0, `legacy_decisions` == 0,
`utterance_fallback_rate` == 0.0, `turn_errors` == 0). The report can NOT judge
VI pronunciation or answer-STT quality — those stay ear-only. Compare
`decision_latency_ms_p50/p95` against the EN run: a large VI/EN gap is worth
flagging (extra tokenisation cost). If VI answer transcription is poor you may
see unexpected `action_counts` (e.g. `redirect_to_topic` when STT garbled the
answer) — note it as an STT-quality signal, not an adaptive-logic failure.

> Known limitation: STT (`whisper_model`) must also handle Vietnamese speech for
> the student's *answers* to transcribe well. If answer transcription is poor in
> VI, that's an STT-model issue, separate from the interviewer-language work —
> log it but don't block adaptive on it unless it's unusable.

---

## 5C. Barge-in & closing behavior

**Goal:** verify turn-taking and the closing sequence hold up in real audio.

| # | Step | Expected | Pass? |
|---|------|----------|-------|
| 1 | While the agent is **mid-question**, start speaking over it | Agent stops speaking and listens (barge-in). The first question is spoken with `allow_interruptions=False`, so barge-in on the VERY FIRST question may be suppressed by design — test barge-in on a LATER turn | ☐ |
| 2 | Barge-in on an adaptive **probe/advance** utterance | Agent yields to you; your answer is transcribed and processed | ☐ |
| 3 | After barge-in, confirm no desync | The next agent utterance corresponds to the answer you actually gave (not a stale turn) | ☐ |
| 4 | Let the interview reach its **natural end** | Exactly ONE closing is heard; `suppress_default_closing` prevented the canned remark from stacking on the adaptive closing | ☐ |
| 5 | Confirm the closing is **fully heard before shutdown** | The room does not cut off mid-closing. The runtime is now playback-aware: it forces the closing non-interruptible and awaits `wait_for_playout()` (bounded by a 30s timeout) BEFORE `ctx.shutdown`. Confirm the `voice.closing_playout` event shows `completed=true` (not `timed_out=true`) | ☐ |
| 6 | End early via UI "End interview" (if exposed for voice) | Session finalizes cleanly; evaluation enqueued | ☐ |

**Known limitation to watch for (log, don't necessarily block):**
The adaptive combined utterances (ack + transition + question) are *longer* than
the old bare-question utterances. Confirm the longer TTS playback doesn't cause
awkward VAD turn-detection (agent thinking you're done when you pause
mid-thought). If it does, note it — a turn-detector model may be needed
(currently silero VAD only, per `build_agent_session`).

**Report expectations** (`make voice-signoff-report SESSION=<id>`): exactly one
`closings_emitted`; if the adaptive path produced the closing,
`default_closings_suppressed` == 1 (this is the machine proof of "no double
closing" — step 4). The `voice.closing_playout` event must show `completed=true`
/ `timed_out=false` (machine proof of step 5 — the closing played fully before
shutdown). `disconnects` should reflect only your intentional end/leave. A barge-in that desyncs would usually show up as a `turn_error` or a
`decision` whose `action` doesn't match the answer you gave — cross-check the
`turn_id` ordering if step 3 felt wrong.

---

## 6. Production-enable checklist (gate)

Only flip `ADAPTIVE_INTERVIEWER_ENABLED=true` in production after ALL of:

- [ ] §5A English smoke test — all rows pass, no double-utterance, single closing.
- [ ] §5B Vietnamese smoke test — interviewer speaks VI; language confirmed in
      metadata/Network tab.
- [ ] §5C barge-in & closing — turn-taking works; exactly one closing; no cutoff.
- [ ] Backend cross-check: one AI turn + one version bump per adaptive turn;
      answer rows never duplicated (idempotency holds).
- [ ] Agent log during all three sessions shows **no** unexpected tracebacks; any
      "legacy fallback" occurrences understood and acceptable.
- [ ] Evaluation ran to a verdict on at least one completed voice session
      (the post-session pipeline still works with adaptive-produced turns).
- [ ] Rollback rehearsed: setting the flag OFF + `pm2 restart abridgeai-backend
      abridgeai-interview-agent` returns voice to the legacy sequential path
      within one turn (verify with a quick legacy voice turn).
- [ ] Decide monitoring: watch `interview turn failed` / `pipeline failed` log
      rates for the first N production sessions before widening rollout.
- [ ] **Per-session ops report attached** for each of §5A/§5B/§5C
      (`make voice-signoff-report SESSION=<id>`), and each meets its "report
      expectations": `legacy_decisions` == 0, `utterance_fallback_rate` == 0.0,
      `turn_errors` == 0, single `closings_emitted`.
- [ ] **Latency decision recorded (§6.2).** `decision_latency_ms_p95` from the
      sign-off runs was reviewed and the wait-after-speaking is judged
      acceptable for launch (or a mitigation is agreed). The dev baseline was
      ~10 s p50 against the local gateway — do NOT skip this; it is the most
      likely reason a real user finds voice mode frustrating.
- [ ] **Question mix sane (Step 7).** `selected_question_type_counts` /
      `selected_question_difficulty_counts` in the ops report show the adaptive
      brain selecting a reasonable spread (not, e.g., only the easiest question
      every time). No hard threshold — a sanity read.

### 6.0a Prefer a staged rollout over an all-at-once flip (Steps 10–11)

You no longer have to jump from OFF to 100%. Two dials make the launch gradual:

- **Shadow first (optional, zero risk).** Set
  `ADAPTIVE_INTERVIEWER_SHADOW_ENABLED=true` with the voice mode flag still OFF.
  Students keep getting the legacy path; the adaptive decision is computed and
  logged (`voice.decision` with `shadow=true`) purely for comparison. Read
  `shadow_decisions` / `shadow_action_counts` from the ops report to see what
  the adaptive brain WOULD have done on real traffic before it drives anyone.
- **Then canary by percentage.** After sign-off, set
  `ADAPTIVE_INTERVIEWER_ROLLOUT_PERCENT` to a small number (e.g. 10) alongside
  the mode flag. A deterministic hash of `(student_id, config_id)` picks a
  stable fraction of students, so a given student's experience never flips
  mid-interview. Widen (25 → 50 → 100) as the ops report stays clean.
  `100` (default) preserves the current "everyone" behaviour; `0` gates everyone
  out without touching the mode flag.

### 6.1 Sign-off record (fill one per reviewer session)

Copy this block into the PR / release ticket for EN, VI, and barge-in runs:

```
Voice adaptive sign-off — <EN | VI | barge-in>
  reviewer:            <name>
  date:                <YYYY-MM-DD>
  interview_session_id:<id>
  agent commit/build:  <git sha or pm2 restart time>
  checklist (§5x):     <all pass? note any ☐ left unchecked + why>
  audio quality notes: <naturalness, VI pronunciation, cutoffs, awkward VAD>
  ops report (make voice-signoff-report SESSION=<id>):
    adaptive_decisions / legacy_decisions:   <n> / <n>
    utterance_fallback_rate:                 <x>
    turn_errors:                             <n>
    closings_emitted / default_suppressed:   <n> / <n>
    sessions_submitted / evaluations_enqueued:<n> / <n>
    decision_latency_ms  p50 / p95:          <ms> / <ms>
    action_counts:                           <dict>
  verdict:             <PASS | PASS-with-notes | FAIL>
```

### 6.2 Latency acceptance

The sign-off's job is not only "did it work" but "is the pause bearable". Use
the p95 from the sign-off runs:

| p95 decision latency | Guidance |
|---|---|
| ≤ ~3 s | Launch-acceptable; feels conversational. |
| ~3–6 s | Acceptable with a spoken/again UI "thinking" cue; note as a follow-up. |
| > ~6 s (dev baseline was ~10 s) | Do not launch wide without mitigation: faster gateway/model for the phrasing+decision stages, streaming the acknowledgement before the full decision resolves, or a "one moment" filler. Record the decision either way. |

These bands are a starting rubric, not a hard SLO — set a real threshold once
you have a spread across EN/VI and realistic load.

### Rollback (instant, no deploy)
```
# .env (prod):  ADAPTIVE_INTERVIEWER_ENABLED=false
pm2 restart abridgeai-backend abridgeai-interview-agent
```
All three input modes revert to the exact legacy path. No schema change is
needed to roll back (the Slice-4 idempotency index and additive fields are inert
when the flag is off).

---

## What is already verified automatically (context)

- Language threads token-router → dispatch metadata → agent → bridge →
  `take_session_step` (unit + integration).
- `test_voice_bridge_honors_language[en|vi]` — a real-DB call through
  `bridge.handle_student_turn(..., language=...)` produces EN vs VI utterances.
- `test_voice_session_now_uses_adaptive_path` — voice is no longer hard-gated.
- Bridge unit tests — adaptive advance speaks the **combined** utterance; closing
  sets `suppress_default_closing`; `submit_session` still enqueued.
- 336 interview tests pass together; ruff + mypy clean; all processes boot.

These prove the *logic and wiring*. They do NOT prove the *audio experience* —
that is what §5A–5C cover, and only a human can sign them off.
