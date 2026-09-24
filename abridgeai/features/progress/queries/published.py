from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, text

from abridgeai.features.progress.models import LessonProgress, MaterialEngagement

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_my_lesson_progress(
    db: AsyncSession, *, user_id: UUID, lesson_id: UUID
) -> LessonProgress | None:
    stmt = select(LessonProgress).where(
        LessonProgress.user_id == user_id,
        LessonProgress.lesson_id == lesson_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_my_lesson_progress_for_course(
    db: AsyncSession, *, user_id: UUID, lesson_ids: list[UUID]
) -> list[LessonProgress]:
    if not lesson_ids:
        return []
    stmt = select(LessonProgress).where(
        LessonProgress.user_id == user_id,
        LessonProgress.lesson_id.in_(lesson_ids),
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_lesson_ids_for_course(db: AsyncSession, course_id: UUID) -> list[UUID]:
    rows = await db.execute(
        text(
            """
            SELECT l.id
            FROM lessons l
            JOIN modules m ON m.id = l.module_id
            WHERE m.course_id = :course_id
              AND l.deleted_at IS NULL
              AND l.status = 'published'
              AND m.deleted_at IS NULL
              AND m.status = 'published'
            """
        ),
        {"course_id": course_id},
    )
    return [row[0] for row in rows.all()]


async def list_lesson_ids_for_courses(
    db: AsyncSession, course_ids: list[UUID]
) -> dict[UUID, list[UUID]]:
    """Load lesson ids grouped by course in one query."""
    if not course_ids:
        return {}
    rows = await db.execute(
        text(
            """
            SELECT m.course_id, l.id
            FROM lessons l
            JOIN modules m ON m.id = l.module_id
            WHERE m.course_id = ANY(CAST(:course_ids AS uuid[]))
              AND l.deleted_at IS NULL
              AND l.status = 'published'
              AND m.deleted_at IS NULL
              AND m.status = 'published'
            ORDER BY m.course_id, l.id
            """
        ),
        {"course_ids": [str(course_id) for course_id in course_ids]},
    )
    grouped: dict[UUID, list[UUID]] = {course_id: [] for course_id in course_ids}
    for course_id, lesson_id in rows.all():
        grouped[course_id].append(lesson_id)
    return grouped


async def list_lesson_progress_for_users(
    db: AsyncSession, *, user_ids: list[UUID], lesson_ids: list[UUID]
) -> list[LessonProgress]:
    """Load progress for many users and lessons in one query."""
    if not user_ids or not lesson_ids:
        return []
    stmt = select(LessonProgress).where(
        LessonProgress.user_id.in_(user_ids),
        LessonProgress.lesson_id.in_(lesson_ids),
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_my_engagement_for_lesson(
    db: AsyncSession, *, user_id: UUID, lesson_id: UUID
) -> list[MaterialEngagement]:
    rows = await db.execute(
        text(
            """
            SELECT me.id, me.user_id, me.material_version_id,
                   me.engagement_seconds, me.scroll_position_percent,
                   me.started_at, me.ended_at, me.created_at
            FROM material_engagement me
            JOIN learning_material_versions lmv
                ON lmv.id = me.material_version_id
            JOIN learning_materials lm
                ON lm.id = lmv.material_id
            WHERE lm.lesson_id = :lesson_id
              AND me.user_id = :user_id
              AND lmv.deleted_at IS NULL
              AND lm.deleted_at IS NULL
            ORDER BY me.created_at ASC
            """
        ),
        {"lesson_id": lesson_id, "user_id": user_id},
    )
    items: list[MaterialEngagement] = []
    for row in rows.all():
        engagement = MaterialEngagement(
            user_id=row.user_id,
            material_version_id=row.material_version_id,
            engagement_seconds=row.engagement_seconds,
            scroll_position_percent=row.scroll_position_percent,
            started_at=row.started_at,
            ended_at=row.ended_at,
        )
        engagement.id = row.id
        engagement.created_at = row.created_at
        items.append(engagement)
    return items


async def get_lesson_id_for_material_version(
    db: AsyncSession, material_version_id: UUID
) -> UUID | None:
    row = (
        await db.execute(
            text(
                """
                SELECT lm.lesson_id
                FROM learning_material_versions lmv
                JOIN learning_materials lm ON lm.id = lmv.material_id
                WHERE lmv.id = :material_version_id
                  AND lmv.deleted_at IS NULL
                  AND lm.deleted_at IS NULL
                """
            ),
            {"material_version_id": material_version_id},
        )
    ).one_or_none()
    return row[0] if row is not None else None


async def get_lesson_estimated_seconds(
    db: AsyncSession, lesson_id: UUID
) -> int | None:
    row = (
        await db.execute(
            text(
                """
                SELECT estimated_minutes
                FROM lessons
                WHERE id = :lesson_id
                  AND deleted_at IS NULL
                """
            ),
            {"lesson_id": lesson_id},
        )
    ).one_or_none()
    if row is None or row[0] is None:
        return None
    return int(row[0]) * 60


__all__ = [
    "get_lesson_estimated_seconds",
    "get_lesson_id_for_material_version",
    "get_my_lesson_progress",
    "list_lesson_ids_for_course",
    "list_my_engagement_for_lesson",
    "list_my_lesson_progress_for_course",
]
