from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.progress.models import LessonProgress, MaterialEngagement
from abridgeai.features.progress.queries import (
    get_lesson_estimated_seconds,
    get_lesson_id_for_material_version,
    get_my_lesson_progress,
    list_my_engagement_for_lesson,
)
from abridgeai.features.progress.schemas.public import (
    LessonProgressPublic,
    MaterialEngagementCreate,
    MaterialEngagementPublic,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_AUTO_COMPLETION_THRESHOLD = Decimal("80")
_DEFAULT_ESTIMATED_SECONDS = 600


async def record_material_engagement(
    db: AsyncSession,
    *,
    user_id: UUID,
    payload: MaterialEngagementCreate,
) -> MaterialEngagementPublic:
    if payload.ended_at < payload.started_at:
        raise ValueError("ended_at must be on or after started_at")

    engagement = MaterialEngagement(
        user_id=user_id,
        material_version_id=payload.material_version_id,
        engagement_seconds=payload.engagement_seconds,
        scroll_position_percent=payload.scroll_position_percent,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
    )
    db.add(engagement)
    await db.flush()

    lesson_id = await get_lesson_id_for_material_version(
        db, payload.material_version_id
    )
    if lesson_id is not None:
        await update_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)

    await db.refresh(engagement)
    return MaterialEngagementPublic.model_validate(engagement)


async def update_lesson_progress(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
) -> LessonProgressPublic:
    engagements = await list_my_engagement_for_lesson(
        db, user_id=user_id, lesson_id=lesson_id
    )
    total_seconds = sum(e.engagement_seconds for e in engagements)
    max_scroll = _max_scroll_percent(engagements)

    estimated_seconds = await get_lesson_estimated_seconds(db, lesson_id)
    target_seconds = estimated_seconds or _DEFAULT_ESTIMATED_SECONDS
    completion_percent_decimal = _completion_percent(
        total_seconds, target_seconds, max_scroll
    )
    auto_complete = completion_percent_decimal >= _AUTO_COMPLETION_THRESHOLD
    if auto_complete:
        completion_percent_decimal = Decimal("100")
    last_activity_at = max((e.ended_at for e in engagements), default=None)

    progress = await get_my_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)
    if progress is None:
        progress = LessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            status="not_started",
            completion_percent=Decimal("0"),
            total_time_seconds=0,
        )
        db.add(progress)

    progress.total_time_seconds = total_seconds
    progress.completion_percent = completion_percent_decimal
    progress.last_activity_at = last_activity_at or _utcnow()

    if auto_complete:
        progress.status = "completed"
    elif total_seconds > 0 or (max_scroll is not None and max_scroll > 0):
        progress.status = "in_progress"
    else:
        progress.status = "not_started"

    await db.flush()
    await db.refresh(progress)
    return LessonProgressPublic.model_validate(progress)


def _max_scroll_percent(engagements: list[MaterialEngagement]) -> Decimal | None:
    values = [
        e.scroll_position_percent
        for e in engagements
        if e.scroll_position_percent is not None
    ]
    if not values:
        return None
    return max(values)


def _completion_percent(
    total_seconds: int,
    target_seconds: int,
    max_scroll: Decimal | None,
) -> Decimal:
    time_ratio = (
        Decimal(total_seconds) / Decimal(target_seconds) * Decimal("100")
        if target_seconds > 0
        else Decimal("0")
    )
    if max_scroll is not None:
        return max(time_ratio, max_scroll)
    return time_ratio


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


__all__ = ["record_material_engagement", "update_lesson_progress"]
