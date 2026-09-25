"""Phase 6: pure timing/deadline math tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from abridgeai.features.quizzes.services.timing import (
    EffectiveTiming,
    compute_deadline,
    is_overdue,
    resolve_attempt_timing,
    timing_snapshot,
)

T0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)


def test_deadline_is_start_plus_limit_when_no_close():
    t = EffectiveTiming(time_limit_seconds=600, available_until=None)
    assert compute_deadline(T0, t) == T0 + timedelta(seconds=600)


def test_deadline_capped_by_available_until():
    close = T0 + timedelta(seconds=120)
    t = EffectiveTiming(time_limit_seconds=600, available_until=close)
    assert compute_deadline(T0, t) == close


def test_no_deadline_when_unbounded():
    t = EffectiveTiming(time_limit_seconds=None, available_until=None)
    assert compute_deadline(T0, t) is None


def test_hard_due_caps_deadline():
    due = T0 + timedelta(seconds=60)
    t = EffectiveTiming(time_limit_seconds=600, available_until=None)
    assert compute_deadline(T0, t, due_at=due, hard_due=True) == due
    assert compute_deadline(T0, t, due_at=due, hard_due=False) == T0 + timedelta(seconds=600)




def test_is_overdue_respects_grace():
    t = EffectiveTiming(time_limit_seconds=100, available_until=None)
    at_deadline = T0 + timedelta(seconds=101)
    assert is_overdue(T0, t, now=at_deadline) is True
    assert is_overdue(T0, t, grace_period_seconds=60, now=at_deadline) is False
    assert is_overdue(T0, t, grace_period_seconds=60, now=T0 + timedelta(seconds=200)) is True


def test_attempt_timing_snapshot_wins_over_later_quiz_values():
    quiz = type("Quiz", (), {"time_limit_seconds": 600, "available_until": None})()
    attempt = type(
        "Attempt",
        (),
        {
            "timing_snapshot": {
                "version": 1,
                "time_limit_seconds": 3600,
                "available_until": None,
            }
        },
    )()

    resolved = resolve_attempt_timing(quiz, attempt)

    assert resolved == EffectiveTiming(time_limit_seconds=3600, available_until=None)


def test_timing_snapshot_round_trips_close_time():
    timing = EffectiveTiming(time_limit_seconds=3600, available_until=T0 + timedelta(hours=2))
    attempt = type("Attempt", (), {"timing_snapshot": timing_snapshot(timing)})()
    quiz = type("Quiz", (), {"time_limit_seconds": 600, "available_until": None})()

    assert resolve_attempt_timing(quiz, attempt) == timing
