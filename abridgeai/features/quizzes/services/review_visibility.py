"""Review-visibility window resolver (Phase 2).

Pure functions (no DB) that decide, for a given quiz + attempt + ``now``, which
review time-window is active and therefore which :class:`ReviewWindowFlags`
govern what the student may see.

Windows (mirroring Moodle's review-options matrix):
* ``after_close``       — the quiz's ``available_until`` has passed.
* ``immediately_after`` — within IMMEDIATE_WINDOW_MINUTES of submitting.
* ``later_while_open``  — everything else (submitted, quiz still open).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from abridgeai.features.quizzes.schemas.review_options import (
    ReviewOptions,
    ReviewWindowFlags,
)

IMMEDIATE_WINDOW_MINUTES = 2


def _coerce_options(raw: object) -> ReviewOptions:
    if not raw:
        return ReviewOptions()
    if isinstance(raw, ReviewOptions):
        return raw
    return ReviewOptions.model_validate(raw)


def _as_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _submitted_at(attempt: object) -> datetime | None:
    return getattr(attempt, "submitted_at", None) or getattr(attempt, "completed_at", None)


def resolve_review_window(quiz: object, attempt: object, now: datetime) -> str:
    """Pick the active review window for this attempt at ``now``."""
    close_at = getattr(quiz, "available_until", None)
    submitted_at = _submitted_at(attempt)
    if close_at is not None and now >= _as_aware(close_at):
        return "after_close"
    if submitted_at is not None and (now - _as_aware(submitted_at)) <= timedelta(
        minutes=IMMEDIATE_WINDOW_MINUTES
    ):
        return "immediately_after"
    return "later_while_open"


def resolve_review_visibility(
    quiz: object, attempt: object, now: datetime
) -> ReviewWindowFlags:
    """Return the effective visibility flags for this attempt's active window."""
    opts = _coerce_options(getattr(quiz, "review_options", None))
    window = resolve_review_window(quiz, attempt, now)
    return getattr(opts, window)


__all__ = [
    "IMMEDIATE_WINDOW_MINUTES",
    "resolve_review_visibility",
    "resolve_review_window",
]
