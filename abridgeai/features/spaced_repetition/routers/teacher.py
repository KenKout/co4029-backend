"""Teacher-facing SR dashboard endpoints (T7.5.12).

Each endpoint is course-scoped via
:func:`features.access_control.policies.require_course_permission`
(``course.read.draft``) — the same gate authoring + cohort-progress
endpoints already use, so a teacher who can read a course's draft tree
also gets its SR analytics.

The data layer is T7.5.7's analytics queries:

* :func:`class_kr_distribution` — histogram + mean/median R̂.
* :func:`class_card_difficulty` — top-N hardest cards by mean EF.
* :func:`at_risk_students` — composite UC-COURSE-04 signal.

Cross-feature reads
-------------------
Per-lesson card counts use
:func:`features.quizzes.api.public.get_quiz_question_id_set_by_lesson`
combined with a local ``StudentCardState`` aggregation (Wave 5 T30b).

Three remaining ``text(...)`` blocks are intentionally kept raw and
remain on the ``PATTERN3_ALLOWLIST``:

* enrollment + display-name lookup (``users``, ``user_profiles``,
  ``course_enrollments``) — needs ``enrollments.api.public`` /
  ``identity.api.public`` (T30d).
* draft course lesson tree — needs
  ``courses.api.public.get_course_lessons_for_authoring`` (T30c).
* recent reviews course-scoped scan — needs
  ``quizzes.api.public.get_quiz_question_id_set_by_course`` (T30c).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.features.access_control.policies import require_course_permission
from abridgeai.features.quizzes.api.public import get_quiz_question_id_set_by_lesson
from abridgeai.features.spaced_repetition.models import StudentCardState
from abridgeai.features.spaced_repetition.queries import (
    at_risk_students,
    card_student_results,
    class_card_difficulty,
    class_kr_distribution,
    knowledge_retention_estimate,
    student_lesson_summary,
)
from abridgeai.features.spaced_repetition.schemas.dashboards import (
    AtRiskStudentRead,
    CardStudentResultRead,
    ClassKRDistributionRead,
    DifficultCardRead,
    HistogramBucket,
    LessonStatus,
    StudentSrDetailLessonRead,
    StudentSrDetailRead,
    StudentSrDetailReviewRead,
)
from abridgeai.features.spaced_repetition.sm2.lesson_unlock import (
    check_lesson_unlock,
)

router = APIRouter(prefix="/teacher", tags=["spaced-repetition-teacher"])

_REQUIRE_COURSE_READ_DRAFT = require_course_permission("course_id", "course.read.draft")

_MATURE_KR_THRESHOLD = 0.85
_LEARNING_KR_THRESHOLD = 0.1


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


_STUDENT_IN_COURSE_SQL = text(
    """
    SELECT
        u.id AS student_id,
        COALESCE(up.display_name, u.primary_email) AS name
    FROM course_enrollments ce
    JOIN users u ON u.id = ce.student_id
    LEFT JOIN user_profiles up ON up.user_id = u.id
    WHERE ce.course_id = CAST(:course_id AS uuid)
      AND ce.student_id = CAST(:student_id AS uuid)
      AND ce.status = 'active'
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
    ORDER BY m.position, l.title
    """
)

_RECENT_REVIEWS_SQL = text(
    """
    SELECT
        cr.question_id,
        cr.created_at,
        cr.q_derived,
        cr.ef_after,
        cr.correct,
        qq.prompt_text
    FROM card_reviews cr
    JOIN quiz_questions qq ON qq.id = cr.question_id
    JOIN quizzes q ON q.id = qq.quiz_id
    WHERE cr.student_id = CAST(:student_id AS uuid)
      AND q.course_id = CAST(:course_id AS uuid)
      AND qq.deleted_at IS NULL
      AND q.deleted_at IS NULL
    ORDER BY cr.created_at DESC
    LIMIT :limit
    """
)


@router.get(
    "/courses/{course_id}/lessons/{lesson_id}/cohort-kr",
    response_model=ClassKRDistributionRead,
    dependencies=[Depends(_REQUIRE_COURSE_READ_DRAFT)],
)
async def get_cohort_kr_distribution(
    course_id: UUID,
    lesson_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ClassKRDistributionRead:
    distribution = await class_kr_distribution(db, course_id=course_id, lesson_id=lesson_id)
    return ClassKRDistributionRead(
        lesson_id=distribution.lesson_id,
        student_count=distribution.student_count,
        histogram=[
            HistogramBucket(bucket_lower=lower, count=count)
            for lower, count in distribution.histogram
        ],
        mean_kr=distribution.mean_kr,
        median_kr=distribution.median_kr,
    )


@router.get(
    "/courses/{course_id}/lessons/{lesson_id}/difficult-cards",
    response_model=list[DifficultCardRead],
    dependencies=[Depends(_REQUIRE_COURSE_READ_DRAFT)],
)
async def get_difficult_cards(
    course_id: UUID,  # noqa: ARG001 -- bound by require_course_permission
    lesson_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    top_n: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[DifficultCardRead]:
    cards = await class_card_difficulty(db, lesson_id=lesson_id, top_n=top_n)
    return [
        DifficultCardRead(
            question_id=card.question_id,
            quiz_id=card.quiz_id,
            prompt_text=card.prompt_text,
            mean_ef=card.mean_ef,
            student_count=card.student_count,
        )
        for card in cards
    ]


@router.get(
    "/courses/{course_id}/questions/{question_id}/student-results",
    response_model=list[CardStudentResultRead],
    dependencies=[Depends(_REQUIRE_COURSE_READ_DRAFT)],
)
async def get_card_student_results(
    course_id: UUID,  # noqa: ARG001 -- bound by require_course_permission
    question_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[CardStudentResultRead]:
    """Per-student results for one question (weakest EF first)."""
    results = await card_student_results(db, question_id=question_id)
    return [
        CardStudentResultRead(
            student_id=r.student_id,
            name=r.name,
            ef=r.ef,
            total_reviews=r.total_reviews,
            last_reviewed_at=r.last_reviewed_at,
            last_correct=r.last_correct,
            correct_count=r.correct_count,
            review_count=r.review_count,
        )
        for r in results
    ]


@router.get(
    "/courses/{course_id}/at-risk",
    response_model=list[AtRiskStudentRead],
    dependencies=[Depends(_REQUIRE_COURSE_READ_DRAFT)],
)
async def get_at_risk_students(
    course_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[AtRiskStudentRead]:
    students = await at_risk_students(db, course_id=course_id)
    return [
        AtRiskStudentRead(
            student_id=s.student_id,
            name=s.name,
            low_compliance=s.low_compliance,
            frozen_kr=s.frozen_kr,
            high_theory_practice_gap=s.high_theory_practice_gap,
            last_active_at=s.last_active_at,
        )
        for s in students
    ]


@router.get(
    "/courses/{course_id}/students/{student_id}/sr-detail",
    response_model=StudentSrDetailRead,
    dependencies=[Depends(_REQUIRE_COURSE_READ_DRAFT)],
)
async def get_student_sr_detail(
    course_id: UUID,
    student_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    recent_reviews_limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> StudentSrDetailRead:
    """Per-student lesson-by-lesson SR breakdown for one course.

    The course-scoped permission gate runs first; this query layer assumes
    the caller has already cleared ``course.read.draft`` and only enforces
    that the target student is enrolled in the course (404 otherwise so
    enrollment state isn't leaked across courses).
    """
    enrollment = (
        await db.execute(
            _STUDENT_IN_COURSE_SQL,
            {"course_id": str(course_id), "student_id": str(student_id)},
        )
    ).one_or_none()
    if enrollment is None:
        raise _not_found("student_enrollment", student_id)

    lesson_rows = (await db.execute(_LESSONS_IN_COURSE_SQL, {"course_id": str(course_id)})).all()

    lesson_breakdown: list[StudentSrDetailLessonRead] = []
    for row in lesson_rows:
        raw_lesson_id = row[0]
        lesson_uuid = raw_lesson_id if isinstance(raw_lesson_id, UUID) else UUID(str(raw_lesson_id))
        summary = await student_lesson_summary(db, student_id=student_id, lesson_id=lesson_uuid)
        unlock = await check_lesson_unlock(db, student_id=student_id, lesson_id=lesson_uuid)
        kr = await knowledge_retention_estimate(db, user_id=student_id, lesson_id=lesson_uuid)
        question_ids = await get_quiz_question_id_set_by_lesson(db, lesson_uuid)
        if question_ids:
            counts_stmt = select(
                func.count(StudentCardState.question_id),
                func.count(StudentCardState.question_id).filter(
                    StudentCardState.due_at.is_not(None),
                    StudentCardState.due_at <= func.now(),
                ),
            ).where(
                StudentCardState.student_id == student_id,
                StudentCardState.question_id.in_(question_ids),
            )
            counts = (await db.execute(counts_stmt)).one()
            cards_total = int(counts[0] or 0)
            cards_due_now = int(counts[1] or 0)
        else:
            cards_total = 0
            cards_due_now = 0
        lesson_breakdown.append(
            StudentSrDetailLessonRead(
                lesson_id=lesson_uuid,
                lesson_title=str(row[1]),
                kr_estimate=summary.kr_estimate,
                cards_total=cards_total,
                cards_due_now=cards_due_now,
                status=_classify_status(eligible=unlock.eligible, kr_estimate=kr),
            )
        )

    review_rows = (
        await db.execute(
            _RECENT_REVIEWS_SQL,
            {
                "student_id": str(student_id),
                "course_id": str(course_id),
                "limit": recent_reviews_limit,
            },
        )
    ).all()
    recent_reviews = [
        StudentSrDetailReviewRead(
            question_id=r[0] if isinstance(r[0], UUID) else UUID(str(r[0])),
            created_at=r[1],
            q_derived=int(r[2] or 0),
            ef_after=float(r[3] or 0.0),
            correct=bool(r[4]),
            prompt_text=str(r[5] or ""),
        )
        for r in review_rows
    ]

    return StudentSrDetailRead(
        student_id=student_id,
        name=str(enrollment[1]),
        lessons=lesson_breakdown,
        recent_reviews=recent_reviews,
    )


__all__ = ["router"]
