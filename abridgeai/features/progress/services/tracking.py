from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
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


register_conflict_mappings(
    {
        "uq_lesson_progress_user_lesson": "lesson_progress_already_recorded: progress for this lesson already exists for the user",  # noqa: E501
    }
)


async def mark_lesson_complete(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
) -> LessonProgressPublic:
    """Manually flip a lesson to ``completed`` (Coursera-style mark-as-complete).

    Idempotent — calling on an already-completed row keeps it complete and
    refreshes ``last_activity_at``. Always sets ``completion_percent=100``
    even if engagement-based completion is below the auto threshold; the
    student is asserting they're done.

    Creates the row if missing (e.g. a learner who never opened the lesson
    but skips to mark complete from the curriculum view).
    """
    progress = await get_my_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)
    if progress is None:
        progress = LessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            status="completed",
            completion_percent=Decimal("100"),
            total_time_seconds=0,
        )
        db.add(progress)
    else:
        progress.status = "completed"
        progress.completion_percent = Decimal("100")
    progress.last_activity_at = _utcnow()
    await flush_or_conflict(db)
    await db.refresh(progress)
    return LessonProgressPublic.model_validate(progress)


async def unmark_lesson_complete(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
) -> LessonProgressPublic | None:
    """Undo a manual completion — recompute status from engagement.

    The manual toggle WINS over the engagement auto-complete threshold:
    unlike a plain engagement recompute, this call does NOT let
    ``completion_percent >= _AUTO_COMPLETION_THRESHOLD`` re-assert
    ``completed`` (that was a silent no-op — a student who had watched
    enough could never un-tick the lesson). Status reverts to
    ``in_progress`` (any engagement at all) or ``not_started``, with
    ``completion_percent`` kept at the raw engagement aggregate.

    A later engagement heartbeat (``record_material_engagement``) WILL
    re-apply auto-completion — if the student keeps watching/reading past
    the threshold, the lesson legitimately becomes complete again.

    Returns ``None`` when no row exists — there's nothing to undo.
    """
    progress = await get_my_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)
    if progress is None:
        return None
    return await update_lesson_progress(
        db,
        user_id=user_id,
        lesson_id=lesson_id,
        apply_auto_complete=False,
    )


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
    await flush_or_conflict(db)

    lesson_id = await get_lesson_id_for_material_version(db, payload.material_version_id)
    if lesson_id is not None:
        await update_lesson_progress(db, user_id=user_id, lesson_id=lesson_id)

    await db.refresh(engagement)
    return MaterialEngagementPublic.model_validate(engagement)


async def update_lesson_progress(
    db: AsyncSession,
    *,
    user_id: UUID,
    lesson_id: UUID,
    apply_auto_complete: bool = True,
) -> LessonProgressPublic:
    engagements = await list_my_engagement_for_lesson(db, user_id=user_id, lesson_id=lesson_id)
    total_seconds = sum(e.engagement_seconds for e in engagements)
    max_scroll = _max_scroll_percent(engagements)

    estimated_seconds = await get_lesson_estimated_seconds(db, lesson_id)
    target_seconds = estimated_seconds or _DEFAULT_ESTIMATED_SECONDS
    completion_percent_decimal = _completion_percent(total_seconds, target_seconds, max_scroll)
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

    if apply_auto_complete and auto_complete:
        progress.status = "completed"
    elif total_seconds > 0 or (max_scroll is not None and max_scroll > 0):
        progress.status = "in_progress"
    else:
        progress.status = "not_started"

    await flush_or_conflict(db)
    await db.refresh(progress)
    return LessonProgressPublic.model_validate(progress)


def _max_scroll_percent(engagements: list[MaterialEngagement]) -> Decimal | None:
    values = [
        e.scroll_position_percent for e in engagements if e.scroll_position_percent is not None
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


__all__ = [
    "mark_lesson_complete",
    "record_material_engagement",
    "unmark_lesson_complete",
    "update_lesson_progress",
]
