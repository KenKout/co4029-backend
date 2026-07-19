# aBridgeAI backend — developer convenience targets.
#
# The voice-harness targets drive the LiveKit voice interview end-to-end (a
# synthetic student joins a real room, publishes TTS speech, captures the
# agent's TTS). They hit REAL external services (LiveKit Cloud + the LLM/STT/TTS
# gateway) and cost money — see docs/voice-adaptive-smoke-tests.md §7. They are
# NOT part of `make test` / CI.
#
# Adaptive path: the pm2 `abridgeai-interview-agent` process governs whether the
# live agent runs the adaptive brain. To exercise adaptive, enable the flag in
# that process's env and restart it BEFORE running these targets. The harness
# reports `runtime-state rows` / `adaptive ran` as the authoritative signal.

PY := .venv/bin/python
HARNESS := $(PY) -m scripts.voice_harness.run_harness
OUT_DIR ?= ./voice-harness-out

# Default answer sequences (override with EN_ANSWERS=... / VI_ANSWERS=...).
EN_ANSWERS ?= "A fact table stores quantitative measurements for analysis" "It connects to dimension tables through foreign keys"
VI_ANSWERS ?= "Bảng dữ kiện lưu trữ các số đo định lượng" "Nó liên kết với các bảng chiều thông qua khóa ngoại"

# Pacing / safety knobs (override on the command line).
TURN_GAP ?= 9
REPLY_WAIT ?= 12
SCENARIO_TIMEOUT ?= 240

.PHONY: test lint typecheck voice-harness voice-harness-en voice-harness-vi voice-harness-clean voice-live-test voice-signoff-report

test:
	INTERVIEW_VOICE_ENABLED=$${INTERVIEW_VOICE_ENABLED:-false} $(PY) -m pytest

lint:
	.venv/bin/ruff check .

typecheck:
	.venv/bin/mypy abridgeai

# --- Live voice harness (real external services; not part of `make test`) -----

## Run the English voice scenario. Captures agent audio to $(OUT_DIR)-en and
## writes a machine-readable result to $(OUT_DIR)-en/result.json.
voice-harness-en:
	INTERVIEW_VOICE_ENABLED=true $(HARNESS) \
	  --language en \
	  --answers $(EN_ANSWERS) \
	  --turn-gap $(TURN_GAP) --reply-wait $(REPLY_WAIT) \
	  --scenario-timeout $(SCENARIO_TIMEOUT) \
	  --out $(OUT_DIR)-en --json $(OUT_DIR)-en/result.json \
	  --cleanup

## Run the Vietnamese voice scenario.
voice-harness-vi:
	INTERVIEW_VOICE_ENABLED=true $(HARNESS) \
	  --language vi \
	  --answers $(VI_ANSWERS) \
	  --turn-gap $(TURN_GAP) --reply-wait $(REPLY_WAIT) \
	  --scenario-timeout $(SCENARIO_TIMEOUT) \
	  --out $(OUT_DIR)-vi --json $(OUT_DIR)-vi/result.json \
	  --cleanup

## Run BOTH EN and VI scenarios in sequence (combined smoke run). Each cleans
## its own DB + LiveKit resources. A failure in EN does NOT prevent VI from
## running; the combined target then fails if EITHER scenario failed (so a
## masked EN failure can't pass CI-by-hand). Check each scenario's result.json
## for the authoritative per-language verdict.
voice-harness:
	@rc=0; \
	$(MAKE) voice-harness-en || rc=1; \
	$(MAKE) voice-harness-vi || rc=1; \
	exit $$rc

## Remove all local captured-audio directories (safe, local only).
voice-harness-clean:
	rm -rf $(OUT_DIR) $(OUT_DIR)-en $(OUT_DIR)-vi

## Run the opt-in live pytest wrapper (skipped unless RUN_LIVEKIT_VOICE_TESTS=1).
## Explicitly selects the excluded marker so it runs despite the default filter.
voice-live-test:
	RUN_LIVEKIT_VOICE_TESTS=1 INTERVIEW_VOICE_ENABLED=true \
	  $(PY) -m pytest tests/integration/test_voice_live_harness.py \
	  -m voice_live -o addopts="" -p no:cov -s -v

# --- Human sign-off support (§5A/5B/5C) ---------------------------------------

## Produce the operational report for a HUMAN voice sign-off session from the
## agent's structured voice.* events. Pass SESSION=<interview_session_id> to
## scope the report to just that session (recommended for a single reviewer
## run); omit it to summarise everything currently in the agent log buffer.
## LINES controls how far back to read (default 5000).
##
##   make voice-signoff-report SESSION=070ca448-b2a9-46df-92e4-46590443d3fc
LINES ?= 5000
voice-signoff-report:
	@pm2 logs abridgeai-interview-agent --lines $(LINES) --nostream --raw \
	  | $(PY) -m abridgeai.features.interviews.realtime.voice_report \
	    $(if $(SESSION),--session $(SESSION),)
