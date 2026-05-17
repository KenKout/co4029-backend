from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from abridgeai.features.progress.schemas.authoring import StudentProgressRow

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_ROSTER_PROGRESS_SQL = text(
    resources.files("abridgeai.features.progress.queries.sql")
    .joinpath("roster_progress.sql")
    .read_text(encoding="utf-8")
)


async def list_course_roster_progress(
    db: AsyncSession, course_id: UUID
) -> list[StudentProgressRow]:
    rows = await db.execute(_ROSTER_PROGRESS_SQL, {"course_id": course_id})
    return [
        StudentProgressRow(
            user_id=row.user_id,
            total_lessons=int(row.total_lessons),
            completed_lessons=int(row.completed_lessons),
            in_progress_lessons=int(row.in_progress_lessons),
            not_started_lessons=int(row.not_started_lessons),
            completion_percent=row.completion_percent,
            total_time_seconds=int(row.total_time_seconds),
        )
        for row in rows.all()
    ]


__all__ = ["list_course_roster_progress"]
