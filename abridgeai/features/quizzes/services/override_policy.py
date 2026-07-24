"""Pure precedence resolver for quiz user/group overrides (Phase 5).

No DB access here — this module is deliberately side-effect-free so the
precedence rules can be tested exhaustively. See Phase 5 Task 5.0 decisions:

* user override beats group override beats base quiz value;
* a ``None`` override column means "no exception — fall through";
* among several group overrides the MOST GENEROUS value wins: earliest open,
  latest close/due, largest time-limit/attempts, smallest cooldown, and
  ``allow_retakes=True`` beats ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime


@dataclass(frozen=True)
class EffectivePolicy:
    """The fully-resolved timing/retake policy for one (student, quiz)."""

    available_from: datetime | None
    available_until: datetime | None
    due_at: datetime | None
    time_limit_seconds: int | None
    max_attempts: int | None
    allow_retakes: bool
    cooldown_hours: int | None


@dataclass(frozen=True)
class OverrideValues:
    """A single override row's exception columns. ``None`` == 'no exception'."""

    available_from: datetime | None = None
    available_until: datetime | None = None
    due_at: datetime | None = None
    time_limit_seconds: int | None = None
    max_attempts: int | None = None
    allow_retakes: bool | None = None
    cooldown_hours: int | None = None


def _merge_groups(groups: list[OverrideValues]) -> OverrideValues:
    """Collapse several group overrides into the single most-generous set.

    Generosity rules (Task 5.0): earliest open, latest close/due, largest
    time-limit/attempts, smallest cooldown, allow_retakes True beats False.
    Fields where every group is None stay None.
    """
    if not groups:
        return OverrideValues()

    def _min_dt(field: str) -> datetime | None:
        vals = [getattr(g, field) for g in groups if getattr(g, field) is not None]
        return min(vals) if vals else None

    def _max_dt(field: str) -> datetime | None:
        vals = [getattr(g, field) for g in groups if getattr(g, field) is not None]
        return max(vals) if vals else None

    def _max_int(field: str) -> int | None:
        vals = [getattr(g, field) for g in groups if getattr(g, field) is not None]
        return max(vals) if vals else None

    def _min_int(field: str) -> int | None:
        vals = [getattr(g, field) for g in groups if getattr(g, field) is not None]
        return min(vals) if vals else None

    retake_vals = [g.allow_retakes for g in groups if g.allow_retakes is not None]
    allow_retakes = True if any(retake_vals) else (False if retake_vals else None)

    return OverrideValues(
        available_from=_min_dt("available_from"),
        available_until=_max_dt("available_until"),
        due_at=_max_dt("due_at"),
        time_limit_seconds=_max_int("time_limit_seconds"),
        max_attempts=_max_int("max_attempts"),
        cooldown_hours=_min_int("cooldown_hours"),
        allow_retakes=allow_retakes,
    )


def _coalesce(*values):
    """First non-None wins; returns None if all None."""
    for v in values:
        if v is not None:
            return v
    return None


def resolve_effective_policy(
    base: EffectivePolicy,
    *,
    user: OverrideValues | None,
    groups: list[OverrideValues],
) -> EffectivePolicy:
    """Resolve base <- group-merge <- user, field-by-field.

    Precedence: user override beats group override beats base. A None on an
    override column means 'no exception, fall through to the next level'.
    """
    user = user or OverrideValues()
    group = _merge_groups(groups)

    return replace(
        base,
        available_from=_coalesce(user.available_from, group.available_from, base.available_from),
        available_until=_coalesce(
            user.available_until, group.available_until, base.available_until
        ),
        due_at=_coalesce(user.due_at, group.due_at, base.due_at),
        time_limit_seconds=_coalesce(
            user.time_limit_seconds, group.time_limit_seconds, base.time_limit_seconds
        ),
        max_attempts=_coalesce(user.max_attempts, group.max_attempts, base.max_attempts),
        allow_retakes=_coalesce(user.allow_retakes, group.allow_retakes, base.allow_retakes),
        cooldown_hours=_coalesce(user.cooldown_hours, group.cooldown_hours, base.cooldown_hours),
    )


__all__ = ["EffectivePolicy", "OverrideValues", "resolve_effective_policy"]
