"""Timing service (Phase 6): effective deadline + overdue math.

Pure functions (no DB) so the deadline / overdue rules are trivially unit-
testable, plus a thin async resolver that layers Phase 5 per-user/group
overrides on top of the quiz's base timing columns.

A quiz attempt's deadline is the earliest of:
  * ``started_at + time_limit_seconds`` (per-attempt clock), and
  * ``available_until`` (the quiz close instant), and
  * ``due_at`` — ONLY when ``hard_due`` is set (soft due is advisory).
An unbounded attempt (no limit, no close) has no deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from abridgeai.core.security import utcnow


@dataclass(frozen=True)
class EffectiveTiming:
    """The resolved per-attempt timing after any Phase 5 overrides."""

    time_limit_seconds: int | None
    available_until: datetime | None


def compute_deadline(
    started_at: datetime,
    timing: EffectiveTiming,
    *,
    due_at: datetime | None = None,
    hard_due: bool = False,
) -> datetime | None:
    """Return the moment the attempt must be finished, or None if unbounded."""
    candidates: list[datetime] = []
    if timing.time_limit_seconds is not None and timing.time_limit_seconds > 0:
        candidates.append(started_at + timedelta(seconds=timing.time_limit_seconds))
    if timing.available_until is not None:
        candidates.append(timing.available_until)
    if hard_due and due_at is not None:
        candidates.append(due_at)
    if not candidates:
        return None
    return min(candidates)


def is_overdue(
    started_at: datetime,
    timing: EffectiveTiming,
    *,
    grace_period_seconds: int | None = None,
    now: datetime | None = None,
    due_at: datetime | None = None,
    hard_due: bool = False,
) -> bool:
    """True when the attempt is past its (grace-extended) deadline."""
    now = now or utcnow()
    deadline = compute_deadline(started_at, timing, due_at=due_at, hard_due=hard_due)
    if deadline is None:
        return False
    if grace_period_seconds:
        deadline = deadline + timedelta(seconds=grace_period_seconds)
    return now > deadline


def resolve_effective_timing(quiz: object, effective: object | None = None) -> EffectiveTiming:
    """Build EffectiveTiming from a quiz + an optional Phase 5 EffectivePolicy.

    When ``effective`` (an ``override_policy.EffectivePolicy``) is provided its
    resolved ``time_limit_seconds`` / ``available_until`` win; otherwise the
    quiz's base columns are used.
    """
    time_limit = getattr(quiz, "time_limit_seconds", None)
    available_until = getattr(quiz, "available_until", None)
    if effective is not None:
        eff_limit = getattr(effective, "time_limit_seconds", None)
        eff_until = getattr(effective, "available_until", None)
        if eff_limit is not None:
            time_limit = eff_limit
        if eff_until is not None:
            available_until = eff_until
    return EffectiveTiming(time_limit_seconds=time_limit, available_until=available_until)


__all__ = [
    "EffectiveTiming",
    "compute_deadline",
    "is_overdue",
    "resolve_effective_timing",
]
