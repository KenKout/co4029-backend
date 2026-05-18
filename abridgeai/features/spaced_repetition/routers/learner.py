"""Learner-facing SR dashboard endpoints (T7.5.12).

Three endpoints, all rooted under ``/me/...`` so the student id is *always*
derived from the bearer token (never a path or query parameter). The routes
compose:

* T7.5.6 :func:`check_lesson_unlock` for unlock-gate state.
* T7.5.7 :func:`student_lesson_summary` /
  :func:`knowledge_retention_estimate` for the three thesis metrics.
* Raw cross-feature SQL (joining ``student_card_state`` to ``lessons`` /
  ``quizzes`` / ``quiz_source_lessons``) for cards-due — the existing
  T7.5.7 queries don't expose per-card metadata, only aggregated per-lesson
  metrics, so this router materialises one minimal cross-feature SELECT
  inline (same pattern as the access-control sub-resource resolvers in
  :mod:`features.courses.routers._deps`).
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.cache.client import RedisFallbackError, get_cache
from abridgeai.core.cache.keys import CARDS_DUE
from abridgeai.core.db import get_db
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.core.pagination.cursor import (
    decode_cursor,
    encode_cursor,
)
from abridgeai.features.spaced_repetition.queries import (
    knowledge_retention_estimate,
    student_lesson_summary,
)
from abridgeai.features.spaced_repetition.schemas.dashboards import (
    CardsDueItem,
    CardsDuePage,
    LessonOverviewItem,
    LessonStatus,
    StudentLessonSummaryRead,
)
from abridgeai.features.spaced_repetition.sm2.lesson_unlock import (
    check_lesson_unlock,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["spaced-repetition-learner"])

_MATURE_KR_THRESHOLD = 0.85
_LEARNING_KR_THRESHOLD = 0.1


_CARDS_DUE_SQL = text(
    """
    SELECT
        scs.question_id,
        qq.quiz_id,
        qsl.lesson_id,
        l.title AS lesson_title,
        scs.due_at,
        scs.last_q,
        scs.ef
    FROM student_card_state scs
    JOIN quiz_questions qq ON qq.id = scs.question_id
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    JOIN lessons l ON l.id = qsl.lesson_id
    WHERE scs.student_id = CAST(:student_id AS uuid)
      AND scs.due_at IS NOT NULL
      AND scs.due_at <= NOW()
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
      AND l.deleted_at IS NULL
      AND (CAST(:lesson_id AS uuid) IS NULL OR qsl.lesson_id = CAST(:lesson_id AS uuid))
      AND (CAST(:after_qid AS uuid) IS NULL OR scs.question_id > CAST(:after_qid AS uuid))
    ORDER BY scs.question_id
    LIMIT :limit
    """
)

_LESSONS_IN_COURSE_SQL = text(
    """
    SELECT l.id AS lesson_id, l.title AS lesson_title
    FROM lessons l
    JOIN modules m ON m.id = l.module_id
    WHERE m.course_id = CAST(:course_id AS uuid)
      AND l.deleted_at IS NULL
      AND m.deleted_at IS NULL
      AND l.status = 'published'
      AND m.status = 'published'
    ORDER BY m.position, l.title
    """
)

_LESSON_DUE_COUNT_SQL = text(
    """
    SELECT COUNT(*)
    FROM student_card_state scs
    JOIN quiz_questions qq ON qq.id = scs.question_id
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    WHERE scs.student_id = CAST(:student_id AS uuid)
      AND qsl.lesson_id = CAST(:lesson_id AS uuid)
      AND scs.due_at IS NOT NULL
      AND scs.due_at <= NOW()
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
    """
)


def _not_found(resource: str, ident: str | UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(ident)},
    )


def _classify_status(*, eligible: bool, kr_estimate: float) -> LessonStatus:
    if not eligible:
        return "locked"
    if kr_estimate >= _MATURE_KR_THRESHOLD:
        return "mature"
    if kr_estimate >= _LEARNING_KR_THRESHOLD:
        return "learning"
    return "locked"


def _build_cards_due_cache_key(
    *,
    user_id: UUID,
    lesson_id: UUID | None,
    cursor: str | None,
    limit: int,
) -> str:
    base = CARDS_DUE.format(user_id=user_id)
    return f"{base}:{lesson_id or 'all'}:{cursor or 'first'}:{limit}"


async def _load_cards_due(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID | None,
    cursor: str | None,
    limit: int,
) -> CardsDuePage:
    after_qid = decode_cursor(cursor) if cursor else None
    rows = (
        await db.execute(
            _CARDS_DUE_SQL,
            {
                "student_id": str(student_id),
                "lesson_id": str(lesson_id) if lesson_id else None,
                "after_qid": str(after_qid) if after_qid else None,
                "limit": limit,
            },
        )
    ).all()
    items = [
        CardsDueItem(
            question_id=row[0] if isinstance(row[0], UUID) else UUID(str(row[0])),
            quiz_id=row[1] if isinstance(row[1], UUID) else UUID(str(row[1])),
            lesson_id=row[2] if isinstance(row[2], UUID) else UUID(str(row[2])),
            lesson_title=str(row[3]),
            due_at=row[4],
            last_q=int(row[5]) if row[5] is not None else None,
            ef=float(row[6]),
        )
        for row in rows
    ]
    next_cursor = encode_cursor(items[-1].question_id) if len(items) == limit else None
    return CardsDuePage(items=items, next_cursor=next_cursor)


@router.get("/me/cards-due", response_model=CardsDuePage)
async def list_my_cards_due(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    lesson_id: Annotated[UUID | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> CardsDuePage:
    """Cards due across all enrolled lessons for the bearer-token student.

    Cross-user isolation: the path has NO ``user_id`` parameter. The
    student id is always derived from the JWT, so user A can never query
    user B's queue.

    Cached under the ``CARDS_DUE`` namespace (``cards_due:{user_id}:...``)
    so T7.5.13's cache invalidator can pattern-delete ``cards_due:{user_id}*``
    on any write to ``student_card_state`` / ``card_reviews``.
    """
    cache = get_cache()
    cache_key = _build_cards_due_cache_key(
        user_id=current_user.user_id,
        lesson_id=lesson_id,
        cursor=cursor,
        limit=limit,
    )
    try:
        raw = await cache.get(cache_key)
    except (RedisError, RedisFallbackError, OSError) as exc:
        logger.warning(
            "cache.get_failed",
            extra={"event": "cache_get_failed", "key": cache_key, "err": repr(exc)},
        )
        raw = None
    if raw is not None:
        try:
            return CardsDuePage.model_validate_json(raw)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "cache.decode_failed",
                extra={"event": "cache_decode_failed", "key": cache_key, "err": repr(exc)},
            )

    page = await _load_cards_due(
        db,
        student_id=current_user.user_id,
        lesson_id=lesson_id,
        cursor=cursor,
        limit=limit,
    )
    try:
        await cache.set(cache_key, page.model_dump_json(), ex=CARDS_DUE.ttl_seconds)
    except (RedisError, RedisFallbackError, OSError, TypeError) as exc:
        logger.warning(
            "cache.set_failed",
            extra={"event": "cache_set_failed", "key": cache_key, "err": repr(exc)},
        )
    return page


@router.get(
    "/me/lessons/{lesson_id}/sr-summary",
    response_model=StudentLessonSummaryRead,
)
async def get_my_lesson_sr_summary(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StudentLessonSummaryRead:
    summary = await student_lesson_summary(db, student_id=current_user.user_id, lesson_id=lesson_id)
    return StudentLessonSummaryRead(
        kr_estimate=summary.kr_estimate,
        progression_ready=summary.progression_ready,
        compliance_rate=summary.compliance_rate,
        cards_total=summary.cards_total,
        cards_due_now=summary.cards_due_now,
    )


@router.get(
    "/me/courses/{course_id}/sr-overview",
    response_model=list[LessonOverviewItem],
)
async def get_my_course_sr_overview(
    course_id: UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[LessonOverviewItem]:
    """All lessons in a course annotated with the student's SR status.

    Status precedence (per T7.5.12 spec):

    * ``locked``  — :func:`check_lesson_unlock` returns ``eligible=False``.
    * ``mature``  — eligible AND ``kr_estimate >= 0.85``.
    * ``learning`` — eligible AND ``0.1 <= kr_estimate < 0.85``.
    * ``locked``  — eligible AND ``kr_estimate < 0.1`` (treated as
      not-yet-engaged; mirrors the "no progress yet" UX).
    """
    rows = (await db.execute(_LESSONS_IN_COURSE_SQL, {"course_id": str(course_id)})).all()
    if not rows:
        raise _not_found("course", course_id)

    student_id = current_user.user_id
    items: list[LessonOverviewItem] = []
    for row in rows:
        raw_lesson_id = row[0]
        lesson_uuid = raw_lesson_id if isinstance(raw_lesson_id, UUID) else UUID(str(raw_lesson_id))
        unlock = await check_lesson_unlock(db, student_id=student_id, lesson_id=lesson_uuid)
        kr = await knowledge_retention_estimate(db, user_id=student_id, lesson_id=lesson_uuid)
        due_count = (
            await db.execute(
                _LESSON_DUE_COUNT_SQL,
                {"student_id": str(student_id), "lesson_id": str(lesson_uuid)},
            )
        ).scalar_one()
        items.append(
            LessonOverviewItem(
                lesson_id=lesson_uuid,
                lesson_title=str(row[1]),
                status=_classify_status(eligible=unlock.eligible, kr_estimate=kr),
                kr_estimate=kr,
                due_count=int(due_count or 0),
            )
        )
    return items


__all__ = ["router"]
