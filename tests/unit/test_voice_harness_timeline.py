"""Unit tests for the voice harness diagnostic timeline + latency metrics.

These exercise the pure helpers ``_iso`` / ``_delta_s`` /
``_build_timeline_and_latency`` against a fake room object — NO live services,
NO DB. They lock in:
  * the schema version is exposed on the result,
  * client-observable latencies are computed from observed timestamps,
  * missing/negative deltas degrade to None (never a crash, never a bogus
    negative latency),
  * every latency carries a source note distinguishing client-observable from
    DB-proxy from agent-internal.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from scripts.voice_harness.run_harness import (
    RESULT_SCHEMA_VERSION,
    ScenarioResult,
    _build_timeline_and_latency,
    _delta_s,
    _iso,
)

_T0 = datetime(2026, 7, 14, 12, 0, 0, tzinfo=UTC)


def _fake_room(**overrides: object) -> SimpleNamespace:
    """A stand-in for HarnessRoom exposing only the attributes the timeline
    builder reads."""
    cap = SimpleNamespace(
        first_frame_at=_T0 + timedelta(seconds=3),
        last_frame_at=_T0 + timedelta(seconds=40),
        first_frame_after_prompt_at=_T0 + timedelta(seconds=30),
    )
    room = SimpleNamespace(
        room_joined_at=_T0,
        agent_joined_at=_T0 + timedelta(seconds=2),
        disconnected_at=_T0 + timedelta(seconds=60),
        student_turns=[
            {"started_at": _T0 + timedelta(seconds=10), "ended_at": _T0 + timedelta(seconds=15)},
            {"started_at": _T0 + timedelta(seconds=25), "ended_at": _T0 + timedelta(seconds=28)},
        ],
        capture=cap,
    )
    for k, v in overrides.items():
        setattr(room, k, v)
    return room


def test_iso_and_delta_helpers() -> None:
    assert _iso(None) is None
    assert _iso(_T0) == _T0.isoformat()
    assert _delta_s(_T0 + timedelta(seconds=5), _T0) == 5.0
    # Missing endpoint → None.
    assert _delta_s(None, _T0) is None
    assert _delta_s(_T0, None) is None
    # Negative delta (clock went backwards / out of order) → None, not a bogus value.
    assert _delta_s(_T0, _T0 + timedelta(seconds=5)) is None


def test_build_timeline_from_observed_events() -> None:
    room = _fake_room()
    # Two AI-turn commits; the LAST (at 32s) pairs with last speech end (28s).
    signals = {
        "ai_turn_committed_at": [
            "2026-07-14T12:00:16+00:00",
            "2026-07-14T12:00:32+00:00",
        ]
    }
    timeline, latency = _build_timeline_and_latency(room, signals)

    assert timeline.room_joined_at == _T0.isoformat()
    assert timeline.agent_joined_at == (_T0 + timedelta(seconds=2)).isoformat()
    assert timeline.first_student_audio_started_at == (_T0 + timedelta(seconds=10)).isoformat()
    assert timeline.last_student_audio_ended_at == (_T0 + timedelta(seconds=28)).isoformat()
    assert timeline.agent_audio_first_frame_at == (_T0 + timedelta(seconds=3)).isoformat()
    assert timeline.ai_turn_committed_at == [
        "2026-07-14T12:00:16+00:00",
        "2026-07-14T12:00:32+00:00",
    ]

    # Latencies from observed spans.
    assert latency.room_join_to_agent_join_s == 2.0
    # PREFERRED (DB proxy): end-of-last-speech (28s) → last AI-turn commit (32s) = 4s.
    assert latency.end_of_speech_to_decision_committed_s == 4.0
    # UNRELIABLE frame-based metric still computed: end-of-speech (28s) →
    # first agent frame after prompt (30s) = 2s.
    assert latency.end_of_speech_to_agent_audio_s == 2.0
    # agent audio span: 3s → 40s = 37s.
    assert latency.agent_audio_span_s == 37.0
    # total scenario: 0 → 60s.
    assert latency.total_scenario_s == 60.0
    # Source notes: preferred metric labelled DB-PROXY, deceptive one flagged UNRELIABLE.
    assert "DB-PROXY" in latency.notes["end_of_speech_to_decision_committed_s"]
    assert "UNRELIABLE" in latency.notes["end_of_speech_to_agent_audio_s"]
    assert "agent-internal" in latency.notes["stt_final_at"]


def test_build_timeline_tolerates_missing_events() -> None:
    """A run that never got agent audio (e.g. legacy/timeout) still builds a
    timeline with Nones rather than crashing."""
    room = _fake_room(
        agent_joined_at=None,
        student_turns=[],
        capture=SimpleNamespace(
            first_frame_at=None, last_frame_at=None, first_frame_after_prompt_at=None
        ),
    )
    timeline, latency = _build_timeline_and_latency(room, {})
    assert timeline.agent_audio_first_frame_at is None
    assert timeline.first_student_audio_started_at is None
    assert latency.end_of_speech_to_agent_audio_s is None
    assert latency.agent_audio_span_s is None
    # Room lifecycle latency still computes.
    assert latency.total_scenario_s == 60.0


def test_result_exposes_schema_version() -> None:
    r = ScenarioResult(ok=True, language="en")
    assert r.result_schema_version == RESULT_SCHEMA_VERSION == 1
    import json

    payload = json.loads(r.to_json())
    # Nested dataclasses serialize; schema version + timeline/latency present.
    assert payload["result_schema_version"] == 1
    assert "timeline" in payload
    assert "latency" in payload
    assert "notes" in payload["latency"]
