"""Service: resolve the effective timing/retake policy for one student (Phase 5).

Layer: services -> queries. Uses the pure resolver in ``override_policy.py``.
Bridges DB rows (base quiz + override rows) into an :class:`EffectivePolicy`.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.quizzes.models import Quiz, QuizOverride
from abridgeai.features.quizzes.queries import overrides as ov_q
from abridgeai.features.quizzes.services.override_policy import (
    EffectivePolicy,
    OverrideValues,
    resolve_effective_policy,
)

_FIELDS = (
    "available_from",
    "available_until",
    "due_at",
    "time_limit_seconds",
    "max_attempts",
    "allow_retakes",
    "cooldown_hours",
)


def _row_to_values(row: QuizOverride) -> OverrideValues:
    return OverrideValues(**{f: getattr(row, f) for f in _FIELDS})


def _base_from_quiz(quiz: Quiz) -> EffectivePolicy:
    return EffectivePolicy(
        available_from=quiz.available_from,
        available_until=quiz.available_until,
        due_at=quiz.due_at,
        time_limit_seconds=quiz.time_limit_seconds,
        max_attempts=quiz.max_attempts,
        allow_retakes=quiz.allow_retakes,
        cooldown_hours=quiz.cooldown_hours,
    )


async def resolve_policy_for_student(
    db: AsyncSession, quiz: Quiz, student_id: uuid.UUID
) -> EffectivePolicy:
    """base <- most-generous applicable group overrides <- the user override."""
    all_rows = await ov_q.list_overrides(db, quiz.id)

    user_row: QuizOverride | None = next(
        (r for r in all_rows if r.scope == "user" and r.user_id == student_id), None
    )

    group_rows = [r for r in all_rows if r.scope == "group" and r.group_id is not None]
    student_groups: set[uuid.UUID] = set()
    if group_rows:
        student_groups = await ov_q.get_group_ids_for_user(db, student_id)

    applicable_groups = [
        _row_to_values(r) for r in group_rows if r.group_id in student_groups
    ]

    return resolve_effective_policy(
        _base_from_quiz(quiz),
        user=_row_to_values(user_row) if user_row else None,
        groups=applicable_groups,
    )


__all__ = ["resolve_policy_for_student"]
