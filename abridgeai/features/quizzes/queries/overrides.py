"""DB access for quiz user/group overrides. Layer: queries (models only).

Phase 5. User overrides are fully functional. Group overrides are stored but
currently inert: this app has no first-class group-membership table yet (only
``course_enrollments``), so :func:`get_group_ids_for_user` returns an empty set
until a groups feature exists. When that table lands, swap in the raw-SQL read
(mirroring ``services/taking.py::_load_cooldown_map``) and group overrides
resolve automatically — no other code changes needed.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.quizzes.models import QuizOverride

_OVERRIDE_FIELDS = (
    "scope",
    "user_id",
    "group_id",
    "available_from",
    "available_until",
    "due_at",
    "time_limit_seconds",
    "max_attempts",
    "allow_retakes",
    "cooldown_hours",
)


async def list_overrides(db: AsyncSession, quiz_id: uuid.UUID) -> list[QuizOverride]:
    result = await db.execute(
        select(QuizOverride)
        .where(QuizOverride.quiz_id == quiz_id)
        .order_by(QuizOverride.scope, QuizOverride.created_at)
    )
    return list(result.scalars().all())


async def get_override(db: AsyncSession, override_id: uuid.UUID) -> QuizOverride | None:
    return await db.get(QuizOverride, override_id)


async def create_override(db: AsyncSession, quiz_id: uuid.UUID, data: dict) -> QuizOverride:
    row = QuizOverride(quiz_id=quiz_id, **{k: data.get(k) for k in _OVERRIDE_FIELDS})
    db.add(row)
    return row


async def update_override(
    db: AsyncSession, override_id: uuid.UUID, data: dict
) -> QuizOverride | None:
    row = await db.get(QuizOverride, override_id)
    if row is None:
        return None
    for k in _OVERRIDE_FIELDS:
        if k in data:
            setattr(row, k, data[k])
    return row


async def delete_override(db: AsyncSession, override_id: uuid.UUID) -> bool:
    row = await db.get(QuizOverride, override_id)
    if row is None:
        return False
    await db.delete(row)  # app-code delete, ondelete=NO ACTION convention
    return True


async def get_group_ids_for_user(db: AsyncSession, user_id: uuid.UUID) -> set[uuid.UUID]:
    """Groups the user belongs to. Cross-feature read (courses owns groups) — raw
    SQL only, never import courses models.

    This app currently has no group-membership table, so this returns an empty
    set: group overrides are stored but do not resolve yet. Replace the body with
    a raw-SQL read once a membership table exists.
    """
    del db, user_id
    return set()


__all__ = [
    "create_override",
    "delete_override",
    "get_group_ids_for_user",
    "get_override",
    "list_overrides",
    "update_override",
]
