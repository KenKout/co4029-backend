"""Visibility-filtered quiz queries (learner / take surface).

Plan §5495-5498. Returns ORM models; the schema layer enforces the
``is_correct`` redaction boundary (per §5513), so these queries
deliberately do NOT strip option data — they hand back the ORM.

Soft-delete is filtered automatically by the T0.7 ``with_loader_criteria``
listener — every ``select(Quiz)`` / ``select(QuizQuestion)`` /
``select(QuizQuestionOption)`` emits ``WHERE deleted_at IS NULL`` for
free.

Cooldown / max-attempts gating
-------------------------------
:func:`get_quiz_for_taking` checks two policy gates *before* returning
the quiz to the learner:

1. **Cooldown** — when ``Quiz.cooldown_hours`` is set and the latest
   ``QuizAttempt.submitted_at`` is within ``cooldown_hours`` of *now*,
   :class:`CooldownActive` is raised. Service maps to HTTP 429.
2. **Max attempts** — when ``Quiz.max_attempts`` is set and the count
   of submitted/graded attempts reaches that ceiling,
   :class:`MaxAttemptsReached` is raised. Service maps to HTTP 409.

Both gates are skipped when the corresponding column is ``NULL``
(unbounded retakes / no cooldown). ``Quiz.allow_retakes=FALSE``
clamps the effective attempt ceiling to 1 inside the gate itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizQuestion,
    QuizQuestionOption,
)


class CooldownActive(Exception):  # noqa: N818  # spec-mandated name (T5.3 §5498)
    """Student attempted the quiz inside its ``cooldown_hours`` window.

    Service layer maps to HTTP 429. Carries the ``retry_after`` datetime
    so the router can populate the ``Retry-After`` header.
    """

    def __init__(self, *, quiz_id: UUID, retry_after: datetime) -> None:
        super().__init__(f"Quiz {quiz_id} is in cooldown until {retry_after.isoformat()}")
        self.quiz_id = quiz_id
        self.retry_after = retry_after


class MaxAttemptsReached(Exception):  # noqa: N818  # spec-mandated name (T5.3 §5498)
    """Student already used every allowed attempt.

    Service layer maps to HTTP 409.
    """

    def __init__(self, *, quiz_id: UUID, max_attempts: int) -> None:
        super().__init__(f"Quiz {quiz_id} max_attempts={max_attempts} reached")
        self.quiz_id = quiz_id
        self.max_attempts = max_attempts


class QuizPasswordRequired(Exception):  # noqa: N818  # matches sibling gate exceptions
    """A quiz password is configured and none (or an empty one) was submitted.

    Service layer maps to HTTP 403 (Phase 12).
    """

    def __init__(self, *, quiz_id: UUID) -> None:
        super().__init__(f"Quiz {quiz_id} requires a password")
        self.quiz_id = quiz_id


class QuizPasswordIncorrect(Exception):  # noqa: N818  # matches sibling gate exceptions
    """A quiz password was submitted but did not match. Service maps to HTTP 403."""

    def __init__(self, *, quiz_id: UUID) -> None:
        super().__init__(f"Quiz {quiz_id} password incorrect")
        self.quiz_id = quiz_id


class QuizSubnetBlocked(Exception):  # noqa: N818  # matches sibling gate exceptions
    """Client IP is not within the quiz's allowed subnet(s). Service maps to HTTP 403."""

    def __init__(self, *, quiz_id: UUID) -> None:
        super().__init__(f"Quiz {quiz_id} blocked for client subnet")
        self.quiz_id = quiz_id


class QuizNotYetOpen(Exception):  # noqa: N818  # matches CooldownActive/MaxAttemptsReached naming
    """Student tried to start before ``Quiz.available_from``.

    Service layer maps to HTTP 409. Carries ``available_from`` so the
    router can tell the student when the quiz opens.
    """

    def __init__(self, *, quiz_id: UUID, available_from: datetime) -> None:
        super().__init__(f"Quiz {quiz_id} not open until {available_from.isoformat()}")
        self.quiz_id = quiz_id
        self.available_from = available_from


class QuizClosed(Exception):  # noqa: N818  # matches CooldownActive/MaxAttemptsReached naming
    """Student tried to start after ``Quiz.available_until``.

    Service layer maps to HTTP 409. Carries ``available_until`` so the
    router can tell the student when the quiz closed.
    """

    def __init__(self, *, quiz_id: UUID, available_until: datetime) -> None:
        super().__init__(f"Quiz {quiz_id} closed at {available_until.isoformat()}")
        self.quiz_id = quiz_id
        self.available_until = available_until


def _assert_within_schedule_window(quiz: Quiz) -> None:
    """Enforce the migration-0032 scheduling window on attempt start.

    NULL columns disable the respective bound, so pre-existing quizzes
    (all NULL) are unaffected.

    * ``available_from`` → hard "not open yet" floor (raises
      :class:`QuizNotYetOpen`, router → 409).
    * ``available_until`` → hard "closed" ceiling (raises
      :class:`QuizClosed`, router → 409).

    ``due_at`` is a SOFT deadline and is deliberately NOT enforced here —
    the UI flags late attempts, but a student can still start after it.
    Only the two hard bounds gate attempt creation.
    """
    now = datetime.now(UTC)
    available_from = quiz.available_from
    if available_from is not None:
        if available_from.tzinfo is None:
            available_from = available_from.replace(tzinfo=UTC)
        if now < available_from:
            raise QuizNotYetOpen(quiz_id=quiz.id, available_from=available_from)
    available_until = quiz.available_until
    if available_until is not None:
        if available_until.tzinfo is None:
            available_until = available_until.replace(tzinfo=UTC)
        if now > available_until:
            raise QuizClosed(quiz_id=quiz.id, available_until=available_until)


def _assert_within_schedule_window_values(
    quiz_id: UUID,
    available_from: datetime | None,
    available_until: datetime | None,
) -> None:
    """Value-based schedule gate (Phase 5).

    Same hard open/close semantics as :func:`_assert_within_schedule_window`,
    but takes resolved (possibly override-adjusted) values rather than reading
    the quiz columns, so per-student overrides shift the window.
    """
    now = datetime.now(UTC)
    if available_from is not None:
        if available_from.tzinfo is None:
            available_from = available_from.replace(tzinfo=UTC)
        if now < available_from:
            raise QuizNotYetOpen(quiz_id=quiz_id, available_from=available_from)
    if available_until is not None:
        if available_until.tzinfo is None:
            available_until = available_until.replace(tzinfo=UTC)
        if now > available_until:
            raise QuizClosed(quiz_id=quiz_id, available_until=available_until)


def _published_clause() -> tuple[Any, ...]:
    return (Quiz.status == "published",)


def _looks_like_uuid(value: str) -> bool:
    try:
        UUID(value)
    except (TypeError, ValueError):
        return False
    return True


async def get_published_quiz(db: AsyncSession, quiz_id: UUID | str) -> Quiz | None:
    """Single published quiz by id OR slug, or ``None`` (router maps to 404).

    Excludes drafts, archived, and soft-deleted quizzes. Existence is
    not leaked — service treats ``None`` uniformly as 404.

    Slug addressing supports the breadcrumb student URLs
    (``/courses/<course-slug>/learn/<item-slug>``): a non-UUID path
    segment is matched against ``Quiz.slug``. Slugs are only unique per
    module, so the caller MUST resolve the course first and pass its id —
    the slug lookup joins through ``modules.course_id`` and takes the
    first published match inside that course.
    """
    if isinstance(quiz_id, str) and not _looks_like_uuid(quiz_id):
        # Lazy import to avoid breaking import-linter's cross-feature contract.
        from abridgeai.features.courses.models import Module  # noqa: PLC0415

        stmt = (
            select(Quiz)
            .join(Module, Module.id == Quiz.module_id)
            .where(
                Quiz.slug == quiz_id,
                *_published_clause(),
                Quiz.deleted_at.is_(None),
                Module.deleted_at.is_(None),
            )
            .order_by(Quiz.created_at)
            .limit(2)
        )
        rows = list((await db.execute(stmt)).scalars().all())
        if len(rows) != 1:
            # 0 = unknown slug; >1 = ambiguous across courses → both 404.
            return None
        return rows[0]
    stmt = select(Quiz).where(Quiz.id == quiz_id, *_published_clause())
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_published_quizzes_for_module(db: AsyncSession, module_id: UUID) -> list[Quiz]:
    """Published quizzes attached to ``module_id``.

    A quiz is "attached" to a module when EITHER:

    * ``Quiz.module_id`` matches (the canonical owning column on the
      ``quizzes`` table), OR
    * a ``module_items`` row with ``item_type='quiz'`` points at the
      quiz from that module (the catalog-level link table).

    The cross-feature ``module_items`` arm is queried via raw SQL so
    the import-linter contract (features are independent) stays clean
    — quizzes does not import the courses ORM.
    """
    rows = (
        await db.execute(
            text(
                "SELECT q.id FROM quizzes q "
                "WHERE q.deleted_at IS NULL "
                "  AND q.status = 'published' "
                "  AND ( "
                "    q.module_id = :module_id "
                "    OR EXISTS ( "
                "      SELECT 1 FROM module_items mi "
                "      WHERE mi.module_id = :module_id "
                "        AND mi.quiz_id = q.id "
                "        AND mi.deleted_at IS NULL "
                "    ) "
                "  ) "
                "ORDER BY q.created_at"
            ),
            {"module_id": module_id},
        )
    ).all()
    if not rows:
        return []
    quiz_ids = [row.id for row in rows]
    stmt = select(Quiz).where(Quiz.id.in_(quiz_ids), *_published_clause()).order_by(Quiz.created_at)
    return list((await db.execute(stmt)).scalars().all())


async def get_quiz_for_taking(
    db: AsyncSession,
    quiz_id: UUID,
    user_id: UUID,
    effective: object | None = None,
    *,
    password: str | None = None,
    client_ip: str | None = None,
) -> Quiz | None:
    """Validate and return a published quiz for a student to take.

    Returns ``None`` (→ 404) when the quiz is missing / unpublished /
    soft-deleted. Raises:

    * :class:`CooldownActive` when the student's most recent submitted
      attempt sits inside ``cooldown_hours``.
    * :class:`MaxAttemptsReached` when the student has used every
      allowed attempt.

    NULL columns disable the corresponding gate, except
    ``allow_retakes=FALSE`` which clamps the effective ceiling to a
    single attempt regardless of ``max_attempts`` (enforced here —
    creation does NOT normalise the columns, and enforcing at read
    time covers pre-existing rows without a data migration).
    """
    quiz = await get_published_quiz(db, quiz_id)
    if quiz is None:
        return None

    # Phase 5: an EffectivePolicy (from override resolution) may override the
    # quiz's base timing/retake columns for THIS student. When absent, the base
    # quiz columns are used (unchanged cohort-wide behaviour).
    eff_available_from = getattr(effective, "available_from", None) if effective else None
    eff_available_until = getattr(effective, "available_until", None) if effective else None
    eff_cooldown_hours = (
        getattr(effective, "cooldown_hours", None) if effective else None
    )
    eff_max_attempts = getattr(effective, "max_attempts", None) if effective else None
    eff_allow_retakes = getattr(effective, "allow_retakes", None) if effective else None
    if eff_available_from is None:
        eff_available_from = quiz.available_from
    if eff_available_until is None:
        eff_available_until = quiz.available_until
    if eff_cooldown_hours is None:
        eff_cooldown_hours = quiz.cooldown_hours
    if eff_max_attempts is None:
        eff_max_attempts = quiz.max_attempts
    if eff_allow_retakes is None:
        eff_allow_retakes = quiz.allow_retakes

    _assert_within_schedule_window_values(
        quiz.id, eff_available_from, eff_available_until
    )

    # Phase 12: access rules. Subnet allowlist first (fail closed), then password.
    # These are quiz-level (not overridable). Both raise gate exceptions the
    # service maps to HTTP 403.
    require_subnet = getattr(quiz, "require_subnet", None)
    if require_subnet:
        from abridgeai.features.quizzes.queries.access_rules import (  # noqa: PLC0415
            ip_in_allowlist,
            parse_subnet_allowlist,
        )

        allowlist = parse_subnet_allowlist(require_subnet)
        if allowlist and not ip_in_allowlist(client_ip, allowlist):
            raise QuizSubnetBlocked(quiz_id=quiz.id)

    require_password = getattr(quiz, "require_password", None)
    if require_password:
        from abridgeai.features.quizzes.queries.access_rules import (  # noqa: PLC0415
            password_matches,
        )

        if not password:
            raise QuizPasswordRequired(quiz_id=quiz.id)
        if not password_matches(require_password, password):
            raise QuizPasswordIncorrect(quiz_id=quiz.id)

    # ------------------------------------------------------------------
    # Cooldown gate — most-recent submitted_at + cooldown_hours > now ?
    # ------------------------------------------------------------------
    if eff_cooldown_hours is not None and eff_cooldown_hours > 0:
        last_submit_stmt = select(func.max(QuizAttempt.submitted_at)).where(
            QuizAttempt.quiz_id == quiz_id,
            QuizAttempt.student_id == user_id,
            QuizAttempt.submitted_at.is_not(None),
        )
        last_submit = (await db.execute(last_submit_stmt)).scalar_one_or_none()
        if last_submit is not None:
            now = datetime.now(UTC)
            retry_after = last_submit + timedelta(hours=quiz.cooldown_hours)
            # ``submitted_at`` is stored TZ-aware (DateTime(timezone=True)),
            # but defensive — coerce naive → UTC for the comparison.
            if retry_after.tzinfo is None:
                retry_after = retry_after.replace(tzinfo=UTC)
            if retry_after > now:
                raise CooldownActive(quiz_id=quiz_id, retry_after=retry_after)

    # ------------------------------------------------------------------
    # Max-attempts gate — count student attempts (any status counts;
    # in_progress + abandoned consume an attempt slot just like submitted).
    # ------------------------------------------------------------------
    effective_max = 1 if not eff_allow_retakes else eff_max_attempts
    if effective_max is not None and effective_max > 0:
        attempts_stmt = select(func.count(QuizAttempt.id)).where(
            QuizAttempt.quiz_id == quiz_id,
            QuizAttempt.student_id == user_id,
        )
        used = (await db.execute(attempts_stmt)).scalar_one()
        if used >= effective_max:
            raise MaxAttemptsReached(quiz_id=quiz_id, max_attempts=effective_max)

    return quiz


async def get_attempt_for_review(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    user_id: UUID,
) -> QuizAttempt | None:
    """Load an attempt + its answers for the calling student.

    Returns ``None`` (router → 404) when the attempt doesn't exist,
    belongs to a different student, or is still in flight (review only
    surfaces after submission). Eagerly loads ``answers`` so the service
    layer can project them without further round-trips.
    """
    stmt = (
        select(QuizAttempt)
        .where(
            QuizAttempt.id == attempt_id,
            QuizAttempt.student_id == user_id,
            QuizAttempt.status.in_(("submitted", "graded")),
        )
        .options(selectinload(QuizAttempt.answers))
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_in_progress_attempt(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    user_id: UUID,
) -> QuizAttempt | None:
    """Load an in-progress attempt + its saved answers for resume.

    Returns ``None`` (router → 404) when the attempt doesn't exist,
    belongs to a different student, or is no longer in flight (a
    submitted/graded/abandoned attempt has nothing to resume). Eagerly
    loads ``answers`` so the service can project the resume payload
    without extra round-trips. The projection layer is responsible for
    dropping correctness fields (no-leak: the attempt is still open).
    """
    stmt = (
        select(QuizAttempt)
        .where(
            QuizAttempt.id == attempt_id,
            QuizAttempt.student_id == user_id,
            QuizAttempt.status == "in_progress",
        )
        .options(selectinload(QuizAttempt.answers))
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_quiz_questions_with_options(
    db: AsyncSession, quiz_id: UUID
) -> list[tuple[QuizQuestion, list[QuizQuestionOption]]]:
    """Load questions + options for a quiz, ordered by position.

    Returns pairs ``(question, [options])`` so the projection layer doesn't
    need another round-trip and we don't need to monkey-patch a relationship
    that isn't declared on ``QuizQuestion``.
    """
    stmt = (
        select(QuizQuestion)
        .where(
            QuizQuestion.quiz_id == quiz_id,
            # Approved-only: the published/preview view mirrors what a student
            # sees, so pending/rejected drafts must not surface here either.
            QuizQuestion.review_status == "approved",
        )
        .order_by(QuizQuestion.position)
    )
    questions = list((await db.execute(stmt)).scalars().all())
    if not questions:
        return []
    options_stmt = (
        select(QuizQuestionOption)
        .where(QuizQuestionOption.question_id.in_([q.id for q in questions]))
        .order_by(QuizQuestionOption.question_id, QuizQuestionOption.position)
    )
    options_by_question: dict[UUID, list[QuizQuestionOption]] = {}
    for opt in (await db.execute(options_stmt)).scalars().all():
        options_by_question.setdefault(opt.question_id, []).append(opt)
    return [(q, options_by_question.get(q.id, [])) for q in questions]


__all__ = [
    "CooldownActive",
    "MaxAttemptsReached",
    "QuizClosed",
    "QuizNotYetOpen",
    "get_attempt_for_review",
    "get_published_quiz",
    "get_quiz_for_taking",
    "list_published_quizzes_for_module",
    "list_quiz_questions_with_options",
]
