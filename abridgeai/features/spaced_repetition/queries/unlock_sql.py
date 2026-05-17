"""Raw-SQL helpers for spaced-repetition lesson-unlock aggregation."""

from __future__ import annotations

from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_LESSON_UNLOCK_SQL = text(
    resources.files("abridgeai.features.spaced_repetition.queries.sql")
    .joinpath("lesson_unlock.sql")
    .read_text(encoding="utf-8")
)

DEFAULT_BLOCKING_LIMIT = 50


async def aggregate_lesson_card_ef(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
    ef_min: float,
    blocking_limit: int = DEFAULT_BLOCKING_LIMIT,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Return (passing, total, blocking) for a lesson's quiz cards.

    ``passing`` and ``total`` are non-negative ints. ``blocking`` is a
    list of card descriptors (question_id / current_ef / quiz_id /
    source_chunk_ids) for every card whose stored EF is below
    ``ef_min``, capped at ``blocking_limit`` rows.
    """
    result = await db.execute(
        _LESSON_UNLOCK_SQL,
        {
            "student_id": str(student_id),
            "lesson_id": str(lesson_id),
            "ef_min": float(ef_min),
            "blocking_limit": int(blocking_limit),
        },
    )
    row = result.first()
    if row is None:
        return 0, 0, []
    passing_raw, total_raw, blocking_raw = row[0], row[1], row[2]
    passing = int(passing_raw or 0)
    total = int(total_raw or 0)
    if isinstance(blocking_raw, list):
        blocking_list: list[dict[str, Any]] = list(blocking_raw)
    elif blocking_raw is None:
        blocking_list = []
    else:
        import json

        blocking_list = list(json.loads(blocking_raw))
    return passing, total, blocking_list


async def fetch_lesson_unlock_config(
    db: AsyncSession,
    *,
    lesson_id: UUID,
) -> tuple[float, float, bool] | None:
    """Read the lesson unlock-config trio via raw SQL.

    Returns ``(ef_min_unlock, tau_unlock, requires_interview_pass)`` or
    ``None`` when the lesson does not exist (or is soft-deleted).
    """
    result = await db.execute(
        text(
            "SELECT ef_min_unlock::float8, tau_unlock::float8, "
            "requires_interview_pass "
            "FROM lessons WHERE id = :lesson_id AND deleted_at IS NULL"
        ),
        {"lesson_id": str(lesson_id)},
    )
    row = result.first()
    if row is None:
        return None
    return float(row[0]), float(row[1]), bool(row[2])


async def fetch_prerequisite_lesson_ids(
    db: AsyncSession,
    *,
    lesson_id: UUID,
) -> list[UUID]:
    """Return direct prerequisite lesson UUIDs for ``lesson_id``."""
    result = await db.execute(
        text("SELECT prereq_lesson_id FROM lesson_prerequisites WHERE lesson_id = :lesson_id"),
        {"lesson_id": str(lesson_id)},
    )
    rows = result.all()
    out: list[UUID] = []
    for r in rows:
        val = r[0]
        out.append(val if isinstance(val, UUID) else UUID(str(val)))
    return out


async def fetch_lesson_module_id(
    db: AsyncSession,
    *,
    lesson_id: UUID,
) -> UUID | None:
    """Return the parent module UUID for ``lesson_id`` (or None)."""
    result = await db.execute(
        text("SELECT module_id FROM lessons WHERE id = :lesson_id AND deleted_at IS NULL"),
        {"lesson_id": str(lesson_id)},
    )
    row = result.first()
    if row is None:
        return None
    val = row[0]
    return val if isinstance(val, UUID) else UUID(str(val))


async def has_passing_interview_for_module(
    db: AsyncSession,
    *,
    student_id: UUID,
    module_id: UUID,
) -> bool:
    """True iff the student has a completed+passed interview for module."""
    result = await db.execute(
        text(
            "SELECT 1 FROM interview_sessions s "
            "JOIN interview_configs c ON c.id = s.interview_config_id "
            "WHERE s.student_id = :student_id "
            "AND c.module_id = :module_id "
            "AND s.status = 'completed' "
            "AND s.pass_verdict = TRUE "
            "AND c.deleted_at IS NULL "
            "LIMIT 1"
        ),
        {"student_id": str(student_id), "module_id": str(module_id)},
    )
    return result.first() is not None


__all__ = [
    "DEFAULT_BLOCKING_LIMIT",
    "aggregate_lesson_card_ef",
    "fetch_lesson_module_id",
    "fetch_lesson_unlock_config",
    "fetch_prerequisite_lesson_ids",
    "has_passing_interview_for_module",
]
