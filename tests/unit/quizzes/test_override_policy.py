"""Phase 5 — pure override precedence resolver tests."""

from __future__ import annotations

from datetime import datetime, timezone

from abridgeai.features.quizzes.services.override_policy import (
    EffectivePolicy,
    OverrideValues,
    resolve_effective_policy,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _base() -> EffectivePolicy:
    return EffectivePolicy(
        available_from=_dt("2026-01-10T00:00:00"),
        available_until=_dt("2026-01-20T00:00:00"),
        due_at=_dt("2026-01-18T00:00:00"),
        time_limit_seconds=3600,
        max_attempts=2,
        allow_retakes=True,
        cooldown_hours=24,
    )


def test_no_overrides_returns_base():
    base = _base()
    assert resolve_effective_policy(base, user=None, groups=[]) == base


def test_user_override_beats_base_field_by_field():
    base = _base()
    user = OverrideValues(max_attempts=5, time_limit_seconds=7200)
    out = resolve_effective_policy(base, user=user, groups=[])
    assert out.max_attempts == 5
    assert out.time_limit_seconds == 7200
    assert out.cooldown_hours == 24
    assert out.available_from == base.available_from


def test_user_override_beats_group_override():
    base = _base()
    out = resolve_effective_policy(
        base, user=OverrideValues(max_attempts=5), groups=[OverrideValues(max_attempts=10)]
    )
    assert out.max_attempts == 5


def test_group_only_applies_when_user_field_is_null():
    base = _base()
    out = resolve_effective_policy(
        base,
        user=OverrideValues(time_limit_seconds=7200),
        groups=[OverrideValues(max_attempts=10)],
    )
    assert out.time_limit_seconds == 7200
    assert out.max_attempts == 10


def test_most_generous_group_wins():
    base = _base()
    g1 = OverrideValues(
        available_from=_dt("2026-01-05T00:00:00"),
        available_until=_dt("2026-01-19T00:00:00"),
        max_attempts=3,
        cooldown_hours=48,
    )
    g2 = OverrideValues(
        available_from=_dt("2026-01-08T00:00:00"),
        available_until=_dt("2026-01-25T00:00:00"),
        max_attempts=6,
        cooldown_hours=6,
    )
    out = resolve_effective_policy(base, user=None, groups=[g1, g2])
    assert out.available_from == _dt("2026-01-05T00:00:00")
    assert out.available_until == _dt("2026-01-25T00:00:00")
    assert out.max_attempts == 6
    assert out.cooldown_hours == 6


def test_allow_retakes_true_beats_false_in_group_merge():
    base = _base()
    out = resolve_effective_policy(
        base,
        user=None,
        groups=[OverrideValues(allow_retakes=False), OverrideValues(allow_retakes=True)],
    )
    assert out.allow_retakes is True
