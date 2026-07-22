"""Offline operational report for the voice-interview path (Phase 9).

Parses the structured ``voice.*`` events emitted by
:mod:`abridgeai.features.interviews.realtime.observability` (as JSON log lines)
and derives a compact operational summary: adaptive success rate, legacy
fallback rate, turn-latency percentiles, STT/TTS failure rates, and disconnect
rate.

It is intentionally a *pure function over a list of event dicts* plus a thin
JSONL reader, so it can run against:
  * a captured ``pm2 logs`` dump,
  * a log-backend export (Loki/Datadog query results), or
  * a live harness run's stdout,
without any DB or LiveKit dependency. This is the "compact operational report"
half of the observability deliverable; a full dashboard would consume the same
``voice.*`` events from the log backend.

Usage::

    pm2 logs abridgeai-interview-agent --lines 5000 --nostream --raw \\
      | python -m abridgeai.features.interviews.realtime.voice_report

or point it at a file::

    python -m abridgeai.features.interviews.realtime.voice_report events.jsonl
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from abridgeai.features.interviews.realtime import observability as obs


def _append_num(target: list[float], value: Any) -> None:  # noqa: ANN401 - event field is untyped JSON
    """Append ``value`` to ``target`` iff it is a real number (not bool/None)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target.append(float(value))


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in [0,100]); None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    # Nearest-rank: rank = ceil(pct/100 * N), 1-indexed.
    import math

    rank = max(1, math.ceil((pct / 100.0) * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


def _rate(numerator: int, denominator: int) -> float | None:
    """Ratio in [0,1] rounded to 3dp; None when there's nothing to divide."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


@dataclass
class VoiceOpsReport:
    """Compact operational rollup over a batch of voice events."""

    events_parsed: int = 0
    sessions: int = 0
    # Turn counts
    turns_started: int = 0
    turns_completed: int = 0
    turn_errors: int = 0
    # Adaptive vs legacy (from voice.decision events)
    decisions: int = 0
    adaptive_decisions: int = 0
    legacy_decisions: int = 0
    fallback_activations: int = 0
    adaptive_success_rate: float | None = None  # adaptive / decisions
    legacy_fallback_rate: float | None = None  # legacy / decisions
    utterance_fallback_rate: float | None = None  # fallback / adaptive_decisions
    # Latency (ms) percentiles
    decision_latency_ms_p50: float | None = None
    decision_latency_ms_p95: float | None = None
    tts_ms_p50: float | None = None
    tts_ms_p95: float | None = None
    # Reliability
    room_joins: int = 0
    room_join_failures: int = 0
    room_join_failure_rate: float | None = None
    turn_error_rate: float | None = None  # turn_errors / turns_started
    disconnects: int = 0
    disconnect_rate: float | None = None  # disconnects / sessions
    # Flow
    closings_emitted: int = 0
    default_closings_suppressed: int = 0
    sessions_submitted: int = 0
    evaluations_enqueued: int = 0
    # Action histogram (adaptive actions taken)
    action_counts: dict[str, int] = field(default_factory=dict)
    # Question-metadata histograms (Phase 7) — WHAT KIND of question the adaptive
    # brain selected on advances. Only advance turns carry question metadata;
    # probe/clarify/repeat/closing turns reuse the current question and don't.
    selected_question_type_counts: dict[str, int] = field(default_factory=dict)
    selected_question_difficulty_counts: dict[str, int] = field(default_factory=dict)
    # Shadow mode (Phase 10) — decisions COMPUTED but not driving the student
    # (voice.decision with shadow=true). Counted separately so they never inflate
    # the live adaptive/legacy rates above. ``shadow_action_counts`` shows what
    # the adaptive brain *would* have done on the shadowed traffic.
    shadow_decisions: int = 0
    shadow_action_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _tally_decision(
    ev: dict[str, Any],
    r: VoiceOpsReport,
    *,
    action_counts: dict[str, int],
    qtype_counts: dict[str, int],
    qdiff_counts: dict[str, int],
    shadow_action_counts: dict[str, int],
) -> None:
    """Tally one ``voice.decision`` event into the report + histograms.

    Shadow decisions (Phase 10) are COMPUTED but never drive the student, so
    they are counted separately and must NOT inflate the live adaptive/legacy
    rates. Live adaptive decisions also feed the action + question-metadata
    histograms; legacy decisions just bump the legacy counter.
    """
    if ev.get("shadow"):
        r.shadow_decisions += 1
        shadow_action = ev.get("action")
        if shadow_action:
            shadow_action_counts[str(shadow_action)] += 1
        return
    r.decisions += 1
    if not ev.get("adaptive"):
        r.legacy_decisions += 1
        return
    r.adaptive_decisions += 1
    action = ev.get("action")
    if action:
        action_counts[str(action)] += 1
    qtype = ev.get("selected_question_type")
    if qtype:
        qtype_counts[str(qtype)] += 1
    qdiff = ev.get("selected_question_difficulty")
    if qdiff:
        qdiff_counts[str(qdiff)] += 1


def build_report(events: list[dict[str, Any]]) -> VoiceOpsReport:
    """Aggregate a list of parsed voice-event dicts into a VoiceOpsReport."""
    r = VoiceOpsReport()
    session_ids: set[str] = set()
    decision_latencies: list[float] = []
    tts_latencies: list[float] = []
    action_counts: dict[str, int] = defaultdict(int)
    qtype_counts: dict[str, int] = defaultdict(int)
    qdiff_counts: dict[str, int] = defaultdict(int)
    shadow_action_counts: dict[str, int] = defaultdict(int)

    # Simple "increment one counter" events → attribute name on the report.
    _counter_attr = {
        obs.EV_TURN_STARTED: "turns_started",
        obs.EV_TURN_COMPLETED: "turns_completed",
        obs.EV_TURN_ERROR: "turn_errors",
        obs.EV_FALLBACK: "fallback_activations",
        obs.EV_DISCONNECT: "disconnects",
        obs.EV_CLOSING_EMITTED: "closings_emitted",
        obs.EV_DEFAULT_CLOSING_SUPPRESSED: "default_closings_suppressed",
        obs.EV_SESSION_SUBMITTED: "sessions_submitted",
        obs.EV_EVALUATION_ENQUEUED: "evaluations_enqueued",
    }

    for ev in events:
        name = ev.get("event")
        if name not in obs.ALL_EVENTS:
            continue
        r.events_parsed += 1
        sid = ev.get("session_id")
        if sid:
            session_ids.add(str(sid))

        counter = _counter_attr.get(str(name))
        if counter is not None:
            setattr(r, counter, getattr(r, counter) + 1)

        # Events that also carry a latency sample or sub-fields.
        if name == obs.EV_TURN_COMPLETED:
            _append_num(decision_latencies, ev.get("decision_latency_ms"))
        elif name == obs.EV_TTS_COMPLETED:
            _append_num(tts_latencies, ev.get("tts_ms"))
        elif name == obs.EV_DECISION:
            _tally_decision(
                ev,
                r,
                action_counts=action_counts,
                qtype_counts=qtype_counts,
                qdiff_counts=qdiff_counts,
                shadow_action_counts=shadow_action_counts,
            )
        elif name == obs.EV_ROOM_JOIN:
            r.room_joins += 1
            if ev.get("ok") is False:
                r.room_join_failures += 1

    r.sessions = len(session_ids)
    r.action_counts = dict(sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    r.selected_question_type_counts = dict(
        sorted(qtype_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    r.selected_question_difficulty_counts = dict(
        sorted(qdiff_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    r.shadow_action_counts = dict(
        sorted(shadow_action_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    # Derived rates
    r.adaptive_success_rate = _rate(r.adaptive_decisions, r.decisions)
    r.legacy_fallback_rate = _rate(r.legacy_decisions, r.decisions)
    r.utterance_fallback_rate = _rate(r.fallback_activations, r.adaptive_decisions)
    r.room_join_failure_rate = _rate(r.room_join_failures, r.room_joins)
    r.turn_error_rate = _rate(r.turn_errors, r.turns_started)
    r.disconnect_rate = _rate(r.disconnects, r.sessions)
    # Latency percentiles
    r.decision_latency_ms_p50 = _percentile(decision_latencies, 50)
    r.decision_latency_ms_p95 = _percentile(decision_latencies, 95)
    r.tts_ms_p50 = _percentile(tts_latencies, 50)
    r.tts_ms_p95 = _percentile(tts_latencies, 95)
    return r


# Staged-rollout rollback thresholds (Slice 6/7). Breaching any of these on a
# rollout stage is the documented signal to pause/roll back that stage. Kept
# here as the single source of truth so the report and the rollout runbook agree.
ROLLBACK_TURN_ERROR_RATE_MAX = 0.01  # >1% of turns erroring
ROLLBACK_UTTERANCE_FALLBACK_RATE_MAX = 0.05  # >5% of adaptive turns using fallback text
ROLLBACK_LEGACY_FALLBACK_RATE_MAX = 0.05  # >5% adaptive→legacy fallback


@dataclass
class RollbackSignals:
    """Boolean rollback triggers derived from a :class:`VoiceOpsReport`.

    Each flag is True when the corresponding metric BREACHES its threshold on
    the analysed window. ``should_rollback`` is the OR of all triggers — a
    single True means the stage should be paused/rolled back. A None metric
    (no data) is treated as "not breached" so an empty window never trips.
    """

    turn_error_rate_breached: bool = False
    utterance_fallback_rate_breached: bool = False
    legacy_fallback_rate_breached: bool = False
    question_loop_detected: bool = False
    should_rollback: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _over(value: float | None, threshold: float) -> bool:
    """True iff ``value`` is present AND strictly over ``threshold``."""
    return value is not None and value > threshold


def evaluate_rollback(report: VoiceOpsReport) -> RollbackSignals:
    """Derive rollback triggers from an aggregated report (pure).

    Flags each metric that breaches its documented threshold. A None metric
    (no data in the window) never trips. ``question_loop_detected`` is reserved
    for a future dedicated loop-counter event and defaults False — the decision
    policy already caps follow-ups (Slice 1 invariants), so an offline proxy
    would be noisier than the guarantee it duplicates.
    """
    signals = RollbackSignals(
        turn_error_rate_breached=_over(report.turn_error_rate, ROLLBACK_TURN_ERROR_RATE_MAX),
        utterance_fallback_rate_breached=_over(
            report.utterance_fallback_rate, ROLLBACK_UTTERANCE_FALLBACK_RATE_MAX
        ),
        legacy_fallback_rate_breached=_over(
            report.legacy_fallback_rate, ROLLBACK_LEGACY_FALLBACK_RATE_MAX
        ),
        question_loop_detected=False,
    )
    signals.should_rollback = any(
        (
            signals.turn_error_rate_breached,
            signals.utterance_fallback_rate_breached,
            signals.legacy_fallback_rate_breached,
            signals.question_loop_detected,
        )
    )
    return signals


def parse_jsonl(lines: list[str]) -> list[dict[str, Any]]:
    """Extract voice-event dicts from raw log lines.

    Each line may be a bare JSON object or have a log prefix; we find the first
    ``{`` and try to parse from there. Non-JSON / non-voice lines are skipped.
    """
    events: list[dict[str, Any]] = []
    for line in lines:
        obj = _extract_event_dict(line)
        if obj is not None:
            events.append(obj)
    return events


def _extract_event_dict(line: str) -> dict[str, Any] | None:
    """Pull one voice-event dict out of a raw log line, or None.

    Handles the formats seen in practice:
      1. a bare/clean JSON object with a top-level ``event`` key;
      2. a JSON log record whose ``message`` field carries the event dict —
         either as nested JSON or (structlog+stdlib bridge) a Python-repr
         string with single quotes, which we recover via ``ast.literal_eval``.
    A dict qualifies only if it has a string ``event`` key.
    """
    brace = line.find("{")
    if brace < 0:
        return None
    fragment = line[brace:]
    try:
        obj = json.loads(fragment)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("event"), str):
        return obj
    # Log-wrapper case: the event dict is stringified inside "message".
    msg = obj.get("message")
    if isinstance(msg, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                inner = loader(msg)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(inner, dict) and isinstance(inner.get("event"), str):
                return inner
    return None


def filter_by_session(events: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """Keep only events for one ``session_id`` (for per-session sign-off)."""
    return [e for e in events if str(e.get("session_id")) == session_id]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Aggregate voice.* observability events into an ops report."
    )
    p.add_argument(
        "path",
        nargs="?",
        help="Log/JSONL file to read (default: stdin). Accepts pm2 raw logs.",
    )
    p.add_argument(
        "--session",
        help="Only aggregate events for this interview session id (per-session sign-off).",
    )
    p.add_argument(
        "--rollback",
        action="store_true",
        help="Print the rollback-signal evaluation (staged-rollout gate) instead "
        "of the full report; exit code 2 when any trigger is breached.",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.path:
        with open(args.path, encoding="utf-8") as fh:
            lines = fh.readlines()
    else:
        lines = sys.stdin.readlines()
    events = parse_jsonl(lines)
    if args.session:
        events = filter_by_session(events, args.session)
    report = build_report(events)
    if args.rollback:
        signals = evaluate_rollback(report)
        print(signals.to_json())  # noqa: T201 - CLI writes to stdout by design
        # Non-zero exit lets CI / a rollout script gate on the signal.
        return 2 if signals.should_rollback else 0
    print(report.to_json())  # noqa: T201 - CLI report writes to stdout by design
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
