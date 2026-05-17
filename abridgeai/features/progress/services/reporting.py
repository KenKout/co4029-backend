from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.progress.models import LessonProgress
from abridgeai.features.progress.queries import (
    get_my_lesson_progress,
    list_lesson_ids_for_course,
    list_my_lesson_progress_for_course,
)
from abridgeai.features.progress.schemas.public import (
    LessonProgressPublic,
    LessonProgressSummary,
    MyCourseProgressSummary,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_my_lesson_progress_view(
    db: AsyncSession, *, user_id: UUID, lesson_id: UUID
) -> LessonProgressPublic | None:
    progress = await get_my_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)
    if progress is None:
        return None
    return LessonProgressPublic.model_validate(progress)


async def get_my_course_progress_summary(
    db: AsyncSession, *, user_id: UUID, course_id: UUID
) -> MyCourseProgressSummary:
    lesson_ids = await list_lesson_ids_for_course(db, course_id)
    progresses = await list_my_lesson_progress_for_course(
        db, user_id=user_id, lesson_ids=lesson_ids
    )
    by_lesson = {p.lesson_id: p for p in progresses}

    completed = sum(1 for p in progresses if p.status == "completed")
    in_progress = sum(1 for p in progresses if p.status == "in_progress")
    total = len(lesson_ids)
    not_started = total - completed - in_progress
    if not_started < 0:
        not_started = 0

    completion_percent = (
        (Decimal(completed) / Decimal(total) * Decimal("100"))
        if total > 0
        else Decimal("0")
    )
    total_time_seconds = sum(p.total_time_seconds for p in progresses)
    last_activity_at = _max_last_activity(progresses)

    summaries = [
        _to_summary(by_lesson.get(lesson_id), lesson_id) for lesson_id in lesson_ids
    ]

    return MyCourseProgressSummary(
        course_id=course_id,
        total_lessons=total,
        completed_lessons=completed,
        in_progress_lessons=in_progress,
        not_started_lessons=not_started,
        completion_percent=completion_percent,
        total_time_seconds=total_time_seconds,
        last_activity_at=last_activity_at,
        lessons=summaries,
    )


def _to_summary(
    progress: LessonProgress | None, lesson_id: UUID
) -> LessonProgressSummary:
    if progress is None:
        return LessonProgressSummary(
            lesson_id=lesson_id,
            status="not_started",
            completion_percent=Decimal("0"),
            last_activity_at=None,
            total_time_seconds=0,
        )
    return LessonProgressSummary(
        lesson_id=progress.lesson_id,
        status=progress.status,
        completion_percent=progress.completion_percent,
        last_activity_at=progress.last_activity_at,
        total_time_seconds=progress.total_time_seconds,
    )


def _max_last_activity(progresses: list[LessonProgress]) -> datetime | None:
    timestamps: list[datetime] = [
        p.last_activity_at for p in progresses if p.last_activity_at is not None
    ]
    if not timestamps:
        return None
    return max(timestamps)


__all__ = ["get_my_course_progress_summary", "get_my_lesson_progress_view"]
