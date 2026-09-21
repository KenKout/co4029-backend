from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.progress.models import LessonProgress
from abridgeai.features.progress.queries import (
    get_my_lesson_progress,
    list_lesson_ids_for_course,
    list_lesson_ids_for_courses,
    list_lesson_progress_for_users,
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


async def get_course_progress_for_users(
    db: AsyncSession, *, user_ids: list[UUID], course_ids: list[UUID]
) -> dict[tuple[UUID, UUID], MyCourseProgressSummary]:
    """Build course summaries for many users/courses without per-pair queries."""
    lessons_by_course = await list_lesson_ids_for_courses(db, course_ids)
    lesson_ids = [lesson_id for ids in lessons_by_course.values() for lesson_id in ids]
    progress_rows = await list_lesson_progress_for_users(
        db, user_ids=user_ids, lesson_ids=lesson_ids
    )
    lesson_to_course = {
        lesson_id: course_id
        for course_id, ids in lessons_by_course.items()
        for lesson_id in ids
    }
    progress_by_pair: dict[tuple[UUID, UUID], list[LessonProgress]] = {}
    for progress in progress_rows:
        course_id = lesson_to_course.get(progress.lesson_id)
        if course_id is not None:
            progress_by_pair.setdefault((progress.user_id, course_id), []).append(progress)

    summaries: dict[tuple[UUID, UUID], MyCourseProgressSummary] = {}
    for user_id in user_ids:
        for course_id in course_ids:
            lesson_ids_for_course = lessons_by_course.get(course_id, [])
            progresses = progress_by_pair.get((user_id, course_id), [])
            by_lesson = {progress.lesson_id: progress for progress in progresses}
            completed = sum(1 for progress in progresses if progress.status == "completed")
            in_progress = sum(1 for progress in progresses if progress.status == "in_progress")
            total = len(lesson_ids_for_course)
            summaries[(user_id, course_id)] = MyCourseProgressSummary(
                course_id=course_id,
                total_lessons=total,
                completed_lessons=completed,
                in_progress_lessons=in_progress,
                not_started_lessons=max(0, total - completed - in_progress),
                completion_percent=(
                    Decimal(completed) / Decimal(total) * Decimal("100")
                    if total
                    else Decimal("0")
                ),
                total_time_seconds=sum(progress.total_time_seconds for progress in progresses),
                last_activity_at=_max_last_activity(progresses),
                lessons=[
                    _to_summary(by_lesson.get(lesson_id), lesson_id)
                    for lesson_id in lesson_ids_for_course
                ],
            )
    return summaries


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


__all__ = [
    "get_course_progress_for_users",
    "get_my_course_progress_summary",
    "get_my_lesson_progress_view",
]
