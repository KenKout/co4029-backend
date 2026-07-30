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
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis.exceptions import RedisError
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.cache.client import RedisFallbackError, get_cache
from abridgeai.core.cache.keys import CARDS_DUE
from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.pagination.cursor import (
    decode_composite_cursor,
    encode_composite_cursor,
)
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.courses.api.public import get_published_lessons_for_course
from abridgeai.features.quizzes.api.public import (
    get_quiz_question_id_set_by_lesson,
    get_review_question_payloads,
    grade_review_answer,
)
from abridgeai.features.spaced_repetition.api.public import (
    dispatch_remediation_for_card_failure,
    get_due_card_count,
    record_card_review,
)
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
from abridgeai.features.spaced_repetition.schemas.review import (
    ReviewCard,
    ReviewQueue,
    ReviewSubmitRequest,
    ReviewSubmitResult,
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
        c.title AS course_title,
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
      AND (CAST(:course_slug AS text) IS NULL OR c.slug = CAST(:course_slug AS text))
      AND (
            CAST(:after_due AS timestamptz) IS NULL
            OR scs.due_at > CAST(:after_due AS timestamptz)
            OR (
                scs.due_at = CAST(:after_due AS timestamptz)
                AND scs.question_id > CAST(:after_qid AS uuid)
            )
      )
    ORDER BY scs.due_at ASC, scs.question_id ASC
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
    course_slug: str | None,
    cursor: str | None,
    limit: int,
) -> str:
    base = CARDS_DUE.format(user_id=user_id)
    return f"{base}:{lesson_id or 'all'}:{course_slug or 'all'}:{cursor or 'first'}:{limit}"


async def _load_cards_due(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID | None,
    course_slug: str | None,
    cursor: str | None,
    limit: int,
) -> CardsDuePage:
    # Keyset cursor over the (due_at, question_id) sort. due_at alone is not
    # unique (many cards can share a due instant), so question_id is the
    # tie-breaker in both the ORDER BY and the cursor — otherwise a page
    # boundary that lands mid-tie would skip or repeat cards.
    after_due: Any = None
    after_qid: UUID | None = None
    if cursor:
        after_due, after_qid = decode_composite_cursor(cursor)
    rows = (
        await db.execute(
            _CARDS_DUE_SQL,
            {
                "student_id": str(student_id),
                "lesson_id": str(lesson_id) if lesson_id else None,
                "course_slug": course_slug,
                "after_due": after_due.isoformat() if after_due is not None else None,
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
            course_title=str(row[5]),
            due_at=row[6],
            last_q=int(row[7]) if row[7] is not None else None,
            ef=float(row[8]),
        )
        for row in rows
    ]
    next_cursor = (
        encode_composite_cursor(items[-1].due_at, items[-1].question_id)
        if len(items) == limit
        else None
    )
    return CardsDuePage(items=items, next_cursor=next_cursor)


@router.get("/me/cards-due", response_model=CardsDuePage)
async def list_my_cards_due(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    lesson_id: Annotated[UUID | None, Query()] = None,
    course_slug: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> CardsDuePage:
    """Cards due across all enrolled lessons for the bearer-token student.

    Cross-user isolation: the path has NO ``user_id`` parameter. The
    student id is always derived from the JWT, so user A can never query
    user B's queue.

    ``lesson_id`` narrows to one lesson; ``course_slug`` narrows to one course
    (the two can combine). Both are optional — omit for the full backlog.

    Cached under the ``CARDS_DUE`` namespace (``cards_due:{user_id}:...``)
    so T7.5.13's cache invalidator can pattern-delete ``cards_due:{user_id}*``
    on any write to ``student_card_state`` / ``card_reviews``.
    """
    cache = get_cache()
    cache_key = _build_cards_due_cache_key(
        user_id=current_user.user_id,
        lesson_id=lesson_id,
        course_slug=course_slug,
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
        course_slug=course_slug,
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


@router.get("/me/review/queue", response_model=ReviewQueue)
async def get_review_queue(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    lesson_id: Annotated[UUID | None, Query()] = None,
    course_slug: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> ReviewQueue:
    """Due cards + their (no-leak) question payloads, ready to answer.

    This is the *resolve* surface: previously a due card had no direct way to be
    cleared (SM-2 only fired on the first answer of a fresh quiz attempt), so a
    backlog was permanently stuck. Here the student gets the same
    ``QuizQuestionPublic`` payload the quiz-taking screen uses — fetched via the
    quizzes public API so ``is_correct`` never leaks — and answers each card via
    ``POST /me/review/{question_id}``.

    ``lesson_id`` / ``course_slug`` scope the queue the same way they scope
    ``/me/cards-due`` — so a "Review" action next to one course pulls only that
    course's due cards, not the whole backlog.

    Cards are drawn from the exact same query as ``/me/cards-due`` (so the queue
    and the dashboard count can never disagree), joined to their question
    payloads. ``total_due`` is the student's FULL due backlog (unscoped), so the
    review screen can tell the student how many cards remain beyond this
    session. Questions that no longer resolve to an approved payload (edited to
    draft after becoming due) are dropped from the queue.
    """
    page = await _load_cards_due(
        db,
        student_id=current_user.user_id,
        lesson_id=lesson_id,
        course_slug=course_slug,
        cursor=None,
        limit=limit,
    )
    payloads = await get_review_question_payloads(db, [c.question_id for c in page.items])
    payload_by_qid = {p.id: p for p in payloads}
    cards: list[ReviewCard] = []
    for item in page.items:
        question = payload_by_qid.get(item.question_id)
        if question is None:  # question no longer approved/served — skip
            continue
        cards.append(
            ReviewCard(
                question_id=item.question_id,
                quiz_id=item.quiz_id,
                lesson_id=item.lesson_id,
                lesson_title=item.lesson_title,
                course_slug=item.course_slug,
                course_title=item.course_title,
                due_at=item.due_at,
                ef=item.ef,
                last_q=item.last_q,
                question=question,
            )
        )
    total_due = await get_due_card_count(db, current_user.user_id)
    return ReviewQueue(items=cards, total_due=total_due)


@router.post("/me/review/{question_id}", response_model=ReviewSubmitResult)
async def submit_review(
    question_id: UUID,
    payload: ReviewSubmitRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ReviewSubmitResult:
    """Grade one review answer and reschedule the card via SM-2.

    The write path the old system lacked. Grading reuses the canonical quiz
    grader (``grade_review_answer``); the SM-2 update reuses
    ``record_card_review`` — the exact same engine as answering inside a quiz —
    so a passing answer (q>=3) advances the card and a failing one resets it to
    a 1-day interval + cooldown. ``quiz_attempt_id`` is ``None`` because a review
    is not tied to a quiz attempt. A q==0 failure still fires remediation
    (after commit), matching the quiz flow.

    404 if the question does not resolve (deleted / never existed).
    """
    grade = await grade_review_answer(
        db,
        question_id=question_id,
        selected_option_id=payload.selected_option_id,
        answer_text=payload.answer_text,
    )
    if grade is None:
        raise _not_found("quiz_question", question_id)

    try:
        review = await record_card_review(
            db,
            student_id=current_user.user_id,
            question_id=question_id,
            quiz_attempt_id=None,
            t_actual_ms=payload.t_actual_ms,
            correct=grade.is_correct,
            hint_used=payload.hint_used,
        )
    except (NotFoundError, ValueError) as exc:
        # No T_exp (draft question) or missing question — cannot schedule.
        raise _not_found("quiz_question", question_id) from exc
    await db.commit()

    # Fire remediation for a hard failure (q==0), after commit — same
    # caller-dispatches-after-commit contract as the quiz answer flow.
    for event in review.pending_events:
        try:
            await dispatch_remediation_for_card_failure(
                db,
                student_id=event.student_id,
                question_id=event.question_id,
                quiz_attempt_id=event.quiz_attempt_id,
            )
        except Exception:  # noqa: BLE001 — side-effect must not fail the review
            logger.exception(
                "review_remediation_dispatch_failed",
                extra={"question_id": str(question_id)},
            )

    remaining_due = await get_due_card_count(db, current_user.user_id)
    return ReviewSubmitResult(
        question_id=question_id,
        correct=grade.is_correct,
        q=review.q,
        passing=review.passing,
        due_at=review.due_at,
        interval_days=review.interval_after,
        remaining_due=remaining_due,
        correct_option_ids=grade.correct_option_ids,
        correct_answer_text=grade.correct_answer_text,
        explanation=grade.explanation,
    )


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
