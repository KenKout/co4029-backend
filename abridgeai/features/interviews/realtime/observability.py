"""Structured voice-interview observability (Phase 9).

A thin, dependency-light event emitter for the LiveKit voice path. It logs
compact structured events via the shared structlog pipeline
(:func:`abridgeai.core.observability.get_logger`) so a log backend
(Loki/Datadog/CloudWatch) or the offline aggregator in
:mod:`abridgeai.features.interviews.realtime.voice_report` can derive
operational metrics without bespoke plumbing.

Design rules
------------
* **No raw transcripts by default.** Student/agent utterance *content* is
  never logged; we log lengths and structured control signals (action,
  reason_code, question/outcome ids, latencies, error class). A future
  explicit debug flag could add content, but the default is privacy-safe.
* **Every event is a single structlog ``info`` call** tagged
  ``event="voice.<name>"`` plus a stable field set, so events are trivially
  filterable (``event:voice.*``) and machine-parseable.
* **Correlation.** Events carry ``session_id`` and, for per-turn events, a
  ``turn_id`` (a uuid4 minted by the runtime at the start of each student
  turn) so I/O-level events (STT/TTS, emitted by the runtime) line up with
  decision-level events (action/selected question, emitted by the bridge).
* **Never raises.** Emitting telemetry must not crash an interview; every
  public function swallows its own errors.

This module imports nothing from LiveKit, so it is safe to import from the API
process and unit tests (the heavy ``interview-agent`` extra is not required).
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from abridgeai.core.observability import get_logger

logger = get_logger(__name__)

# ── Event name constants (stable identifiers for log queries/dashboards) ──────
# Lifecycle
EV_ROOM_JOIN = "voice.room_join"
EV_AGENT_DISPATCH = "voice.agent_dispatch"
EV_DISCONNECT = "voice.disconnect"
# Per-turn I/O (emitted by the runtime)
EV_TURN_STARTED = "voice.turn_started"  # STT produced a final transcript
EV_TTS_STARTED = "voice.tts_started"
EV_TTS_COMPLETED = "voice.tts_completed"
EV_TURN_COMPLETED = "voice.turn_completed"
EV_BARGE_IN = "voice.barge_in"
EV_TTS_INTERRUPTED = "voice.tts_interrupted"
# Per-turn decision (emitted by the bridge)
EV_DECISION = "voice.decision"
EV_FALLBACK = "voice.fallback_activated"
EV_CLOSING_EMITTED = "voice.closing_emitted"
EV_DEFAULT_CLOSING_SUPPRESSED = "voice.default_closing_suppressed"
# The closing utterance finished (or timed out) playing out BEFORE shutdown —
# proof the student heard the whole closing, not a room cut off mid-sentence.
EV_CLOSING_PLAYOUT = "voice.closing_playout"
EV_SESSION_SUBMITTED = "voice.session_submitted"
EV_EVALUATION_ENQUEUED = "voice.evaluation_enqueued"
EV_TURN_ERROR = "voice.turn_error"
# Prompt-injection security guard (Phase S1). Transport-agnostic (text/hybrid/
# voice all route through take_session_step), so these are named "interview.*"
# rather than "voice.*". Privacy contract: these events carry only category /
# confidence band / action / counts / fingerprint — never raw student content.
EV_SECURITY_ASSESSED = "interview.security.assessed"
EV_SECURITY_BLOCKED = "interview.security.blocked"
EV_SECURITY_OUTPUT_LEAKAGE_BLOCKED = "interview.security.output_leakage_blocked"
EV_SECURITY_REPEATED_ATTEMPT = "interview.security.repeated_attempt"
EV_SECURITY_SESSION_FLAGGED = "interview.security.session_flagged"

# The complete catalogue — used by the aggregator + tests to validate coverage.
ALL_EVENTS = frozenset(
    {
        EV_ROOM_JOIN,
        EV_AGENT_DISPATCH,
        EV_DISCONNECT,
        EV_TURN_STARTED,
        EV_TTS_STARTED,
        EV_TTS_COMPLETED,
        EV_TURN_COMPLETED,
        EV_BARGE_IN,
        EV_TTS_INTERRUPTED,
        EV_DECISION,
        EV_FALLBACK,
        EV_CLOSING_EMITTED,
        EV_DEFAULT_CLOSING_SUPPRESSED,
        EV_CLOSING_PLAYOUT,
        EV_SESSION_SUBMITTED,
        EV_EVALUATION_ENQUEUED,
        EV_TURN_ERROR,
        EV_SECURITY_ASSESSED,
        EV_SECURITY_BLOCKED,
        EV_SECURITY_OUTPUT_LEAKAGE_BLOCKED,
        EV_SECURITY_REPEATED_ATTEMPT,
        EV_SECURITY_SESSION_FLAGGED,
    }
)


def monotonic() -> float:
    """Monotonic clock for latency spans (never goes backwards on NTP steps)."""
    return time.monotonic()


def latency_ms(started_monotonic: float | None) -> float | None:
    """Elapsed milliseconds since ``started_monotonic`` (None-safe, rounded)."""
    if started_monotonic is None:
        return None
    return round((monotonic() - started_monotonic) * 1000.0, 1)


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop None values so log records stay compact and consistent."""
    return {k: v for k, v in fields.items() if v is not None}


def emit(event: str, *, session_id: Any, **fields: Any) -> None:  # noqa: ANN401 - session_id is UUID|str; fields are arbitrary telemetry
    """Emit one structured voice event. Never raises.

    ``session_id`` is always present; ``turn_id`` and any metric fields are
    passed through when provided. Content-bearing keys (transcript text) must
    NOT be passed here — callers log ``*_chars`` lengths instead.
    """
    # Telemetry must never crash an interview — swallow any logging failure.
    # NB: the event NAME is structlog's positional arg; we also stamp it as a
    # queryable ``event_type`` field. We must NOT pass ``event=`` as a kwarg —
    # structlog already binds the positional as ``event``, and passing both
    # raises TypeError (multiple values for 'event').
    with contextlib.suppress(Exception):
        logger.info(event, event_type=event, session_id=str(session_id), **_clean(fields))


__all__ = [
    "ALL_EVENTS",
    "EV_AGENT_DISPATCH",
    "EV_BARGE_IN",
    "EV_CLOSING_EMITTED",
    "EV_CLOSING_PLAYOUT",
    "EV_DECISION",
    "EV_DEFAULT_CLOSING_SUPPRESSED",
    "EV_DISCONNECT",
    "EV_EVALUATION_ENQUEUED",
    "EV_FALLBACK",
    "EV_ROOM_JOIN",
    "EV_SECURITY_ASSESSED",
    "EV_SECURITY_BLOCKED",
    "EV_SECURITY_OUTPUT_LEAKAGE_BLOCKED",
    "EV_SECURITY_REPEATED_ATTEMPT",
    "EV_SECURITY_SESSION_FLAGGED",
    "EV_SESSION_SUBMITTED",
    "EV_TTS_COMPLETED",
    "EV_TTS_INTERRUPTED",
    "EV_TTS_STARTED",
    "EV_TURN_COMPLETED",
    "EV_TURN_ERROR",
    "EV_TURN_STARTED",
    "emit",
    "latency_ms",
    "monotonic",
]
