"""Unit tests for the voice observability emitter + operational aggregator.

No LiveKit, no DB. The emitter is tested by swapping in a fake logger so we can
assert the exact field shape (and that transcript CONTENT never appears); the
aggregator is tested as a pure function over event dicts.
"""

from __future__ import annotations

from typing import Any

from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import voice_report as vr


class _FakeLogger:
    """Captures info() calls as (event_name, kwargs) tuples."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


def _swap_logger(monkeypatch: Any) -> _FakeLogger:
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)
    return fake


def test_emit_produces_one_record_with_core_fields(monkeypatch: Any) -> None:
    fake = _swap_logger(monkeypatch)
    obs.emit(
        obs.EV_DECISION,
        session_id="sess-1",
        turn_id="turn-1",
        adaptive=True,
        action="ask_for_example",
    )
    assert len(fake.calls) == 1
    name, kw = fake.calls[0]
    assert name == obs.EV_DECISION
    assert kw["event_type"] == obs.EV_DECISION
    assert kw["session_id"] == "sess-1"
    assert kw["turn_id"] == "turn-1"
    assert kw["adaptive"] is True
    assert kw["action"] == "ask_for_example"


def test_emit_drops_none_fields(monkeypatch: Any) -> None:
    fake = _swap_logger(monkeypatch)
    obs.emit(
        obs.EV_TURN_COMPLETED,
        session_id="s",
        turn_id="t",
        decision_latency_ms=None,  # should be dropped
        will_speak=True,
    )
    _name, kw = fake.calls[0]
    assert "decision_latency_ms" not in kw
    assert kw["will_speak"] is True


def test_emit_coerces_session_id_to_str(monkeypatch: Any) -> None:
    import uuid

    fake = _swap_logger(monkeypatch)
    sid = uuid.uuid4()
    obs.emit(obs.EV_ROOM_JOIN, session_id=sid, ok=True)
    _name, kw = fake.calls[0]
    assert kw["session_id"] == str(sid)
    assert isinstance(kw["session_id"], str)


def test_emit_never_raises_on_bad_logger(monkeypatch: Any) -> None:
    class _Boom:
        def info(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("logging backend down")

    monkeypatch.setattr(obs, "logger", _Boom())
    # Must not propagate — telemetry can never crash an interview.
    obs.emit(obs.EV_TURN_ERROR, session_id="s", error_class="X")


def test_emit_only_accepts_length_not_content(monkeypatch: Any) -> None:
    """Contract guard: callers pass *_chars lengths, never transcript text.

    We assert the emitter faithfully passes through whatever it's given, and
    that our own call sites (exercised in the aggregator fixture below) carry
    only length fields — no 'transcript'/'text' content keys.
    """
    fake = _swap_logger(monkeypatch)
    obs.emit(obs.EV_TURN_STARTED, session_id="s", turn_id="t", transcript_chars=42)
    _name, kw = fake.calls[0]
    assert kw["transcript_chars"] == 42
    assert "transcript" not in kw
    assert "text" not in kw


def test_latency_ms_helper() -> None:
    assert obs.latency_ms(None) is None
    start = obs.monotonic()
    val = obs.latency_ms(start)
    assert val is not None
    assert val >= 0.0


# ── Aggregator ───────────────────────────────────────────────────────────────


def _events() -> list[dict[str, Any]]:
    """A synthetic two-turn adaptive session + one legacy turn in another."""
    return [
        {"event": obs.EV_AGENT_DISPATCH, "session_id": "s1", "language": "en"},
        {"event": obs.EV_ROOM_JOIN, "session_id": "s1", "ok": True},
        {"event": obs.EV_TURN_STARTED, "session_id": "s1", "turn_id": "t1", "transcript_chars": 30},
        {
            "event": obs.EV_DECISION,
            "session_id": "s1",
            "turn_id": "t1",
            "adaptive": True,
            "action": "ask_for_example",
            "selected_question_type": "conceptual",
            "selected_question_difficulty": "junior",
        },
        {
            "event": obs.EV_TURN_COMPLETED,
            "session_id": "s1",
            "turn_id": "t1",
            "decision_latency_ms": 1800.0,
        },
        {"event": obs.EV_TTS_COMPLETED, "session_id": "s1", "turn_id": "t1", "tts_ms": 900.0},
        {"event": obs.EV_TURN_STARTED, "session_id": "s1", "turn_id": "t2", "transcript_chars": 12},
        {
            "event": obs.EV_DECISION,
            "session_id": "s1",
            "turn_id": "t2",
            "adaptive": True,
            "action": "ask_for_example",
            "selected_question_type": "conceptual",
            "selected_question_difficulty": "senior",
        },
        {
            "event": obs.EV_FALLBACK,
            "session_id": "s1",
            "turn_id": "t2",
            "action": "ask_for_example",
        },
        {
            "event": obs.EV_TURN_COMPLETED,
            "session_id": "s1",
            "turn_id": "t2",
            "decision_latency_ms": 2200.0,
        },
        {
            "event": obs.EV_CLOSING_EMITTED,
            "session_id": "s1",
            "turn_id": "t2",
            "adaptive_closing": True,
        },
        {"event": obs.EV_DEFAULT_CLOSING_SUPPRESSED, "session_id": "s1", "turn_id": "t2"},
        {"event": obs.EV_SESSION_SUBMITTED, "session_id": "s1", "turn_id": "t2"},
        {"event": obs.EV_EVALUATION_ENQUEUED, "session_id": "s1", "turn_id": "t2"},
        # Second session: one legacy turn.
        {"event": obs.EV_ROOM_JOIN, "session_id": "s2", "ok": True},
        {"event": obs.EV_TURN_STARTED, "session_id": "s2", "turn_id": "t3", "transcript_chars": 5},
        {"event": obs.EV_DECISION, "session_id": "s2", "turn_id": "t3", "adaptive": False},
        {
            "event": obs.EV_TURN_COMPLETED,
            "session_id": "s2",
            "turn_id": "t3",
            "decision_latency_ms": 400.0,
        },
        {"event": obs.EV_DISCONNECT, "session_id": "s2", "reason": "participant_disconnected"},
    ]


def test_report_rates_and_counts() -> None:
    r = vr.build_report(_events())
    assert r.sessions == 2
    assert r.decisions == 3
    assert r.adaptive_decisions == 2
    assert r.legacy_decisions == 1
    # 2 of 3 decisions adaptive (rates rounded to 3dp by _rate).
    assert r.adaptive_success_rate == round(2 / 3, 3)
    assert r.legacy_fallback_rate == round(1 / 3, 3)
    # 1 utterance fallback out of 2 adaptive decisions.
    assert r.utterance_fallback_rate == 0.5
    # 1 disconnect over 2 sessions.
    assert r.disconnect_rate == 0.5
    assert r.turns_started == 3
    assert r.default_closings_suppressed == 1
    assert r.action_counts.get("ask_for_example") == 2


def test_report_question_metadata_histograms() -> None:
    """Phase 7: advance turns carry the selected question's type/difficulty,
    which the report aggregates into histograms. The two adaptive turns in the
    fixture both selected a 'conceptual' question (junior + senior)."""
    r = vr.build_report(_events())
    assert r.selected_question_type_counts == {"conceptual": 2}
    assert r.selected_question_difficulty_counts == {"junior": 1, "senior": 1}


def test_report_question_metadata_absent_when_no_advance() -> None:
    """Probe/legacy turns without question metadata → empty histograms (never
    fabricated)."""
    events = [
        {"event": obs.EV_DECISION, "session_id": "s1", "adaptive": True, "action": "probe_deeper"},
        {"event": obs.EV_DECISION, "session_id": "s2", "adaptive": False},
    ]
    r = vr.build_report(events)
    assert r.selected_question_type_counts == {}
    assert r.selected_question_difficulty_counts == {}


def test_report_counts_v2_actions_and_no_false_loop_rollback() -> None:
    """Slice 14: new v2 actions (depth probe, closing sub-steps) surface in the
    action histogram, and consuming the follow-up budget does NOT trip a false
    question_loop_detected rollback (the histogram is dynamic; loop detection is
    not action-name based)."""
    events = [
        {"event": obs.EV_DECISION, "session_id": "s1", "adaptive": True, "action": "extend_answer"},
        {
            "event": obs.EV_DECISION,
            "session_id": "s1",
            "adaptive": True,
            "action": "prompt_self_reflection",
        },
        {
            "event": obs.EV_DECISION,
            "session_id": "s1",
            "adaptive": True,
            "action": "invite_candidate_questions",
        },
    ]
    r = vr.build_report(events)
    assert r.action_counts.get("extend_answer") == 1
    assert r.action_counts.get("prompt_self_reflection") == 1
    assert r.action_counts.get("invite_candidate_questions") == 1
    # These are normal adaptive decisions → no rollback trigger.
    signals = vr.evaluate_rollback(r)
    assert signals.question_loop_detected is False
    assert signals.should_rollback is False


def test_report_latency_percentiles() -> None:
    r = vr.build_report(_events())
    # decision latencies: [1800, 2200, 400] → p50 nearest-rank = 1800, p95 = 2200.
    assert r.decision_latency_ms_p50 == 1800.0
    assert r.decision_latency_ms_p95 == 2200.0
    assert r.tts_ms_p50 == 900.0


def test_report_ignores_unknown_and_nonvoice_events() -> None:
    events = [
        {"event": "some.other.thing", "session_id": "x"},
        {"event": obs.EV_ROOM_JOIN, "session_id": "s1", "ok": True},
        {"nope": "no event key"},
    ]
    r = vr.build_report(events)
    assert r.events_parsed == 1
    assert r.room_joins == 1


def test_parse_jsonl_tolerates_log_prefixes() -> None:
    lines = [
        '2026-07-14 12:00:00 INFO {"event": "voice.room_join", "session_id": "s1", "ok": true}',
        "not json at all",
        '{"event": "voice.turn_started", "session_id": "s1", "turn_id": "t1"}',
    ]
    events = vr.parse_jsonl(lines)
    assert len(events) == 2
    assert events[0]["event"] == "voice.room_join"


def test_parse_jsonl_recovers_wrapped_message_dict() -> None:
    """Regression: the real agent log wraps the event dict inside a JSON record's
    ``message`` field as a Python-repr string (single quotes). A live run showed
    the parser returned 0 events until it learned to recover this shape."""
    # As emitted in production: outer JSON record, no top-level "event" key, the
    # event dict repr'd (single-quoted) inside "message".
    wrapped = (
        "{\"message\": \"{'event': 'voice.decision', "
        "'session_id': 's1', 'turn_id': 't1', 'adaptive': True, "
        "'action': 'ask_for_example'}\", "
        '"level": "INFO", "name": "abridgeai...observability"}'
    )
    events = vr.parse_jsonl([wrapped])
    assert len(events) == 1
    assert events[0]["event"] == "voice.decision"
    assert events[0]["adaptive"] is True
    assert events[0]["action"] == "ask_for_example"
    # And it aggregates correctly.
    r = vr.build_report(events)
    assert r.adaptive_decisions == 1


def test_percentile_edges() -> None:
    assert vr._percentile([], 50) is None
    assert vr._percentile([5.0], 95) == 5.0
    assert vr._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_filter_by_session_scopes_events() -> None:
    events = vr.parse_jsonl(
        [
            '{"event": "voice.room_join", "session_id": "s1", "ok": true}',
            '{"event": "voice.room_join", "session_id": "s2", "ok": true}',
            '{"event": "voice.decision", "session_id": "s1", "adaptive": true, "action": "ask_for_example"}',
        ]
    )
    only_s1 = vr.filter_by_session(events, "s1")
    assert len(only_s1) == 2
    assert {e["session_id"] for e in only_s1} == {"s1"}
    r = vr.build_report(only_s1)
    assert r.sessions == 1
    assert r.decisions == 1


def test_cli_main_reads_file_and_filters_session(tmp_path: Any, capsys: Any) -> None:
    """The CLI reads a log file, filters to --session, and prints report JSON."""
    import json as _json

    log = tmp_path / "agent.log"
    log.write_text(
        "\n".join(
            [
                '{"event": "voice.room_join", "session_id": "sA", "ok": true}',
                '{"event": "voice.decision", "session_id": "sA", "adaptive": true, "action": "ask_for_example"}',
                '{"event": "voice.decision", "session_id": "sB", "adaptive": false}',
            ]
        ),
        encoding="utf-8",
    )
    rc = vr.main([str(log), "--session", "sA"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    # Only sA's events counted: its 1 adaptive decision, not sB's legacy one.
    assert out["sessions"] == 1
    assert out["decisions"] == 1
    assert out["adaptive_decisions"] == 1


# ── rollback signal evaluation (Slice 6) ─────────────────────────────────────


def test_evaluate_rollback_clean_window_does_not_trip() -> None:
    # All-adaptive, no errors, no fallback → no rollback.
    report = vr.build_report(
        [
            {"event": obs.EV_TURN_STARTED, "session_id": "s"},
            {"event": obs.EV_TURN_COMPLETED, "session_id": "s"},
            {
                "event": obs.EV_DECISION,
                "session_id": "s",
                "adaptive": True,
                "action": "ask_main_question",
            },
        ]
    )
    signals = vr.evaluate_rollback(report)
    assert signals.should_rollback is False
    assert signals.turn_error_rate_breached is False
    assert signals.utterance_fallback_rate_breached is False
    assert signals.legacy_fallback_rate_breached is False


def test_evaluate_rollback_trips_on_high_legacy_fallback() -> None:
    # 1 adaptive vs 9 legacy decisions → 90% legacy fallback, way over 5%.
    events: list[dict[str, Any]] = [
        {
            "event": obs.EV_DECISION,
            "session_id": "s",
            "adaptive": True,
            "action": "ask_main_question",
        },
    ]
    events += [{"event": obs.EV_DECISION, "session_id": "s", "adaptive": False} for _ in range(9)]
    signals = vr.evaluate_rollback(vr.build_report(events))
    assert signals.legacy_fallback_rate_breached is True
    assert signals.should_rollback is True


def test_evaluate_rollback_trips_on_turn_errors() -> None:
    # 100 turns, 5 errored → 5% > 1% threshold.
    events: list[dict[str, Any]] = []
    for _ in range(100):
        events.append({"event": obs.EV_TURN_STARTED, "session_id": "s"})
    for _ in range(5):
        events.append({"event": obs.EV_TURN_ERROR, "session_id": "s"})
    signals = vr.evaluate_rollback(vr.build_report(events))
    assert signals.turn_error_rate_breached is True
    assert signals.should_rollback is True


def test_evaluate_rollback_empty_window_never_trips() -> None:
    signals = vr.evaluate_rollback(vr.build_report([]))
    assert signals.should_rollback is False


def test_main_rollback_flag_exit_code(tmp_path: Any, capsys: Any) -> None:
    import json as _json

    # A window with only legacy decisions → rollback should fire → exit 2.
    log = tmp_path / "ev.jsonl"
    log.write_text(
        "\n".join(
            '{"event": "voice.decision", "session_id": "s", "adaptive": false}' for _ in range(4)
        ),
        encoding="utf-8",
    )
    rc = vr.main([str(log), "--rollback"])
    assert rc == 2
    out = _json.loads(capsys.readouterr().out)
    assert out["legacy_fallback_rate_breached"] is True
    assert out["should_rollback"] is True
