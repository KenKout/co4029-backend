"""Learner-facing SR dashboard endpoints (T7.5.12).

Three endpoints, all rooted under ``/me/...`` so the student id is *always*
derived from the bearer token (never a path or query parameter). The routes
compose:

* T7.5.6 :func:`check_lesson_unlock` for unlock-gate state.
* T7.5.7 :func:`student_lesson_summary` /
  :func:`knowledge_retention_estimate` for the three thesis metrics.
* :mod:`features.courses.api.public` /
  :mod:`features.quizzes.api.public` for cross-feature reads (Wave 5
  T30a/b).
* Raw cross-feature SQL for ``cards-due`` — the published-tree DTOs
  don't expose per-card ``student_card_state`` joins, and lifting that
  join into the quizzes public API would either re-implement SR
  scheduler state on the quizzes side or require a one-off helper that
  violates the per-lesson abstraction. Keep raw + allowlist.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis.exceptions import RedisError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.cache.client import RedisFallbackError, get_cache
from abridgeai.core.cache.keys import CARDS_DUE
from abridgeai.core.db import get_db
from abridgeai.core.pagination.cursor import (
    decode_cursor,
    encode_cursor,
)
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.api.public import get_published_lessons_for_course
from abridgeai.features.quizzes.api.public import get_quiz_question_id_set_by_lesson
from abridgeai.features.spaced_repetition.models import StudentCardState
from abridgeai.features.spaced_repetition.queries import (
    knowledge_retention_estimate,
    student_lesson_summary,
)
from abridgeai.features.spaced_repetition.schemas.dashboards import (
    CardsDueItem,
    CardsDuePage,
    LessonOverviewItem,
    LessonStatus,
    StudentDashboardSummaryRead,
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
        c.slug AS course_slug,
        scs.due_at,
        scs.last_q,
        scs.ef
    FROM student_card_state scs
    JOIN quiz_questions qq ON qq.id = scs.question_id
    JOIN quizzes q ON q.id = qq.quiz_id
    JOIN quiz_source_lessons qsl ON qsl.quiz_id = q.id
    JOIN lessons l ON l.id = qsl.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
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
            course_slug=str(row[4]),
            due_at=row[5],
            last_q=int(row[6]) if row[6] is not None else None,
            ef=float(row[7]),
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
    tree = await get_published_lessons_for_course(db, course_id)
    if not tree:
        raise _not_found("course", course_id)

    published_lessons: list[tuple[UUID, str]] = [(lesson.id, lesson.title) for lesson in tree]

    student_id = current_user.user_id
    items: list[LessonOverviewItem] = []
    for lesson_uuid, lesson_title in published_lessons:
        unlock = await check_lesson_unlock(db, student_id=student_id, lesson_id=lesson_uuid)
        kr = await knowledge_retention_estimate(db, user_id=student_id, lesson_id=lesson_uuid)
        question_ids = await get_quiz_question_id_set_by_lesson(db, lesson_uuid)
        if question_ids:
            due_stmt = select(func.count(StudentCardState.question_id)).where(
                StudentCardState.student_id == student_id,
                StudentCardState.question_id.in_(question_ids),
                StudentCardState.due_at.is_not(None),
                StudentCardState.due_at <= func.now(),
            )
            due_count = (await db.execute(due_stmt)).scalar_one()
        else:
            due_count = 0
        items.append(
            LessonOverviewItem(
                lesson_id=lesson_uuid,
                lesson_title=lesson_title,
                status=_classify_status(eligible=unlock.eligible, kr_estimate=kr),
                kr_estimate=kr,
                due_count=int(due_count or 0),
                eligible=unlock.eligible,
            )
        )
    return items


@router.get(
    "/me/sr-dashboard-summary",
    response_model=StudentDashboardSummaryRead,
)
async def get_my_sr_dashboard_summary(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StudentDashboardSummaryRead:
    """Cross-course SR rollup for the student dashboard landing tiles.

    Aggregates the per-lesson metrics that already existed (``R-hat`` retention
    and the unlock gate) across every course the caller is actively enrolled in.
    The dashboard previously had no such aggregate, so its headline tiles were
    hardcoded to "—" — the thesis metrics were computed but never surfaced where
    a student would see them.

    Reuses :func:`knowledge_retention_estimate` and :func:`check_lesson_unlock`
    rather than reimplementing the maths, so these numbers agree with the
    per-lesson and per-course endpoints by construction. ``cards_due_now`` uses
    the same predicate as ``GET /me/cards-due``.

    ``next_unlock_*`` reports the locked lesson closest to opening, using the
    EF gate's own ``current_ratio`` / ``required_ratio`` (i.e. progress toward
    ``tau_unlock``) — no invented formula.
    """
    student_id = current_user.user_id

    # Active enrolments only. A dropped or completed enrolment shouldn't drag the
    # student's headline retention around.
    course_rows = (
        await db.execute(
            text(
                """
                SELECT ce.course_id
                FROM course_enrollments ce
                JOIN courses c ON c.id = ce.course_id AND c.deleted_at IS NULL
                WHERE ce.student_id = CAST(:student_id AS uuid)
                  AND ce.status = 'active'
                """
            ),
            {"student_id": str(student_id)},
        )
    ).all()
    course_ids = [row[0] for row in course_rows]
    if not course_ids:
        return StudentDashboardSummaryRead()

    kr_values: list[float] = []
    mature = learning = locked = 0
    cards_due_now = 0
    cards_total = 0
    best_locked: tuple[float, UUID, str] | None = None

    for course_id in course_ids:
        tree = await get_published_lessons_for_course(db, course_id)
        for lesson in tree or []:
            lesson_uuid = lesson.id
            unlock = await check_lesson_unlock(db, student_id=student_id, lesson_id=lesson_uuid)
            kr = await knowledge_retention_estimate(db, user_id=student_id, lesson_id=lesson_uuid)

            # Count this lesson's tracked cards FIRST — the retention gate below
            # depends on it.
            lesson_cards = 0
            question_ids = await get_quiz_question_id_set_by_lesson(db, lesson_uuid)
            if question_ids:
                counts = (
                    await db.execute(
                        select(
                            func.count(StudentCardState.question_id),
                            func.count(StudentCardState.question_id).filter(
                                StudentCardState.due_at.is_not(None),
                                StudentCardState.due_at <= func.now(),
                            ),
                        ).where(
                            StudentCardState.student_id == student_id,
                            StudentCardState.question_id.in_(question_ids),
                        )
                    )
                ).one()
                lesson_cards = int(counts[0] or 0)
                cards_total += lesson_cards
                cards_due_now += int(counts[1] or 0)

            # Retention is averaged over lessons where the student actually has
            # tracked cards, regardless of lock state. Two traps here, both found
            # against real data:
            #   * Averaging only *unlocked* lessons picked up empty lessons (0
            #     cards bypass the EF gate, so they unlock trivially) and reported
            #     a confident 0.0, while excluding the locked lessons that held
            #     the real retention signal.
            #   * ``unlock.total_cards`` is NOT the lesson's card count — it is
            #     the EF gate's own working set, and reads 0 for an already
            #     unlocked lesson. Gating on it hid retention for every unlocked
            #     lesson. Use the card count computed above instead.
            # A locked lesson with reviewed cards has a perfectly real R-hat, and
            # it is the most informative number on the dashboard.
            if lesson_cards > 0:
                kr_values.append(kr)

            if not unlock.eligible:
                locked += 1
                # Progress toward the EF gate. required_ratio is tau_unlock; a
                # lesson with no gate cards can't be ranked this way, so skip it.
                if unlock.required_ratio > 0:
                    pct = min(100.0, 100.0 * unlock.current_ratio / unlock.required_ratio)
                    if best_locked is None or pct > best_locked[0]:
                        best_locked = (pct, lesson_uuid, lesson.title)
            elif kr >= _MATURE_KR_THRESHOLD:
                mature += 1
            elif kr >= _LEARNING_KR_THRESHOLD:
                learning += 1

    avg_kr = round(sum(kr_values) / len(kr_values), 3) if kr_values else 0.0

    return StudentDashboardSummaryRead(
        avg_kr_estimate=avg_kr,
        has_retention_data=bool(kr_values),
        lessons_mature=mature,
        lessons_learning=learning,
        lessons_locked=locked,
        lessons_total=mature + learning + locked,
        cards_due_now=cards_due_now,
        cards_total=cards_total,
        next_unlock_lesson_id=best_locked[1] if best_locked else None,
        next_unlock_lesson_title=best_locked[2] if best_locked else None,
        next_unlock_progress_pct=round(best_locked[0], 1) if best_locked else 0.0,
    )


__all__ = ["router"]
