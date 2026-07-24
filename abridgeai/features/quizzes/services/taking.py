"""Student-side quiz taking service (T5.13).

Ports the attempt lifecycle from
``backend/app/routes/quizzes/service.py`` (legacy 465 LOC god-file):
``create_attempt`` → ``start_attempt``, ``answer_attempt``,
``submit_attempt``, plus ``get_attempt_history`` for the learner
dashboard.

Security invariant (plan §5398): :func:`start_attempt` returns
questions through the :class:`QuizQuestionPublic` schema, which
intentionally drops :attr:`QuizQuestionOption.is_correct` (and the
question's ``correct_option_id``-style hints) so the learner client
cannot peek at answer correctness during the take. The grading
service is the only consumer of the authoring projection.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)
from abridgeai.features.quizzes.queries import authoring as authoring_queries
from abridgeai.features.quizzes.queries import published as published_queries
from abridgeai.features.quizzes.queries.published import (
    CooldownActive,
    MaxAttemptsReached,
    QuizClosed,  # noqa: F401  -- re-exported for the learner router (see __all__)
    QuizNotYetOpen,  # noqa: F401  -- re-exported for the learner router (see __all__)
    QuizPasswordIncorrect,  # noqa: F401  -- re-exported for the learner router
    QuizPasswordRequired,  # noqa: F401  -- re-exported for the learner router
    QuizSubnetBlocked,  # noqa: F401  -- re-exported for the learner router
)
from abridgeai.features.quizzes.schemas.attempt import (
    QuizAttemptProgressAnswer,
    QuizAttemptProgressRead,
    QuizAttemptRead,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptReviewRead,
)
from abridgeai.features.quizzes.schemas.public import (
    QuizForTakingPublic,
    QuizPublic,
    QuizQuestionPublic,
)
from abridgeai.features.quizzes.services.grader import grade_answer, needs_manual_grade
from abridgeai.features.spaced_repetition.api.public import (
    CardReviewResult,
    record_card_review,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_logger = get_logger(__name__)

_DEFAULT_FAILURE_COOLDOWN_SECONDS = 86400


class AllCardsInCooldownError(AppError):
    """Every question in the quiz has ``student_card_state.due_at > now``.

    Per thesis UC-LEARN-01 Alt 1a: a student who fails a card cannot
    retry it until the SR scheduler's failure cooldown elapses (default
    24 h, configurable via ``settings.sr_failure_cooldown_seconds``).
    When the *whole* quiz is in cooldown the router must reply HTTP 429
    with a ``Retry-After`` header — :class:`AllCardsInCooldownError`
    carries the timing payload needed to build that response.
    """

    def __init__(
        self,
        retry_available_at: datetime,
        cards_due_at: list[tuple[UUID, datetime]],
    ) -> None:
        super().__init__("All cards in cooldown")
        self.retry_available_at = retry_available_at
        self.cards_due_at = cards_due_at


async def get_published_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz | None:
    """Pass-through to :func:`published_queries.get_published_quiz`.

    Routers cannot import queries directly (T0.4 contract); learner
    callers reach the published-quiz fetcher through this thin service
    indirection.
    """
    return await published_queries.get_published_quiz(db, quiz_id)


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = await authoring_queries.get_quiz_for_authoring(db, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


async def _require_attempt(db: AsyncSession, attempt_id: UUID) -> QuizAttempt:
    attempt = await db.get(QuizAttempt, attempt_id)
    if attempt is None:
        raise NotFoundError(f"Quiz attempt {attempt_id} not found")
    return attempt


async def _next_attempt_number(db: AsyncSession, quiz_id: UUID, student_id: UUID) -> int:
    from sqlalchemy import func, select  # noqa: PLC0415

    stmt = select(func.coalesce(func.max(QuizAttempt.attempt_number), 0) + 1).where(
        QuizAttempt.quiz_id == quiz_id,
        QuizAttempt.student_id == student_id,
    )
    return int((await db.execute(stmt)).scalar_one())


async def _load_quiz_questions_for_taking(db: AsyncSession, quiz_id: UUID) -> list[QuizQuestion]:
    from sqlalchemy import select  # noqa: PLC0415

    questions = (
        (
            await db.execute(
                select(QuizQuestion)
                .where(
                    QuizQuestion.quiz_id == quiz_id,
                    # Students only ever see approved questions — pending /
                    # rejected drafts are never served to the taking surface.
                    QuizQuestion.review_status == "approved",
                )
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    question_ids = [q.id for q in questions]
    options_by_qid: dict[UUID, list[QuizQuestionOption]] = {qid: [] for qid in question_ids}
    if question_ids:
        option_rows = (
            (
                await db.execute(
                    select(QuizQuestionOption)
                    .where(QuizQuestionOption.question_id.in_(question_ids))
                    .order_by(QuizQuestionOption.position)
                )
            )
            .scalars()
            .all()
        )
        for option in option_rows:
            options_by_qid.setdefault(option.question_id, []).append(option)

    for question in questions:
        question.options = options_by_qid.get(question.id, [])  # type: ignore[attr-defined]

    # Stamp the derived (L.O.x) position so the take payload can render the
    # prefix student-side. Raw SQL against course_learning_outcomes keeps the
    # quizzes feature from importing the courses ORM (T0.4 contract), same
    # precedent as the cross-feature cooldown read below. Soft-deleted /
    # missing outcomes resolve to None → no prefix.
    outcome_ids = [q.learning_outcome_id for q in questions if q.learning_outcome_id]
    positions: dict[UUID, int] = {}
    if outcome_ids:
        from sqlalchemy import text as _text  # noqa: PLC0415

        rows = (
            await db.execute(
                _text(
                    "SELECT id, position FROM course_learning_outcomes "
                    "WHERE id = ANY(:ids) AND deleted_at IS NULL"
                ),
                {"ids": list(set(outcome_ids))},
            )
        ).all()
        positions = {row[0]: row[1] for row in rows}
    for question in questions:
        lo_id = question.learning_outcome_id
        question.outcome_position = (  # type: ignore[attr-defined]
            positions.get(lo_id) if lo_id is not None else None
        )
    return list(questions)


async def _load_cooldown_map(
    db: AsyncSession,
    student_id: UUID,
    question_ids: list[UUID],
) -> dict[UUID, datetime]:
    """Return ``{question_id: due_at}`` for cards still in cooldown.

    A row is included only when ``student_card_state.due_at > NOW()``;
    cards the student has never attempted (no row at all) are absent
    from the map and therefore treated as available — per thesis
    UC-LEARN-01 Alt 1a, cooldown applies to *failed* cards only, not to
    cards the learner has not touched yet.

    The query reaches across the spaced-repetition feature boundary on
    purpose: services cannot import ``features/spaced_repetition/models``
    directly (Features-independent contract), so this uses raw SQL
    against the ``student_card_state`` table — mirroring the precedent
    established by T7.5.5's ``record_card_review``.
    """
    from sqlalchemy import text  # noqa: PLC0415

    if not question_ids:
        return {}
    result = await db.execute(
        text(
            "SELECT question_id, due_at FROM student_card_state "
            "WHERE student_id = :sid "
            "  AND question_id = ANY(CAST(:qids AS uuid[])) "
            "  AND due_at > NOW()"
        ),
        {"sid": str(student_id), "qids": [str(q) for q in question_ids]},
    )
    return {row[0]: row[1] for row in result.all()}


async def start_attempt(
    db: AsyncSession,
    quiz_id: UUID,
    actor: CurrentUser,
    *,
    idempotency_key: UUID | None = None,
    password: str | None = None,
    client_ip: str | None = None,
) -> tuple[QuizAttempt, QuizAttemptProgressRead]:
    """Create a :class:`QuizAttempt` and return the no-leak take payload.

    Returns the same :class:`QuizAttemptProgressRead` shape as
    :func:`get_attempt_progress` (``answers=[]`` since nothing is saved
    yet) so the client learns the new ``attempt_id`` immediately instead
    of having to guess it from the attempts list, and so "start" and
    "resume" share one hydration code path on the frontend.

    The serialized response goes through :class:`QuizQuestionPublic`
    which drops ``is_correct`` (and any other answer-correctness fields)
    before the bytes leave the service boundary. Routers must serialize
    the returned :class:`QuizForTakingPublic` directly — never re-hydrate
    via the authoring schema.

    **T7.5.11 — cooldown enforcement.** Before persisting the attempt,
    the service consults ``student_card_state.due_at`` for every quiz
    question (cross-feature read against the SR table — see
    :func:`_load_cooldown_map`). Three outcomes:

    * **All questions in cooldown** — raise
      :class:`AllCardsInCooldownError` carrying the earliest
      ``retry_available_at`` plus the per-question ``due_at`` list. The
      router maps this to HTTP 429 with a ``Retry-After`` header.
    * **Some questions in cooldown** — drop them from the take payload
      and stash the ``(question_id, due_at)`` pairs onto
      ``attempt.cards_in_cooldown`` so the router can echo them on the
      response (no schema change required; the field is dynamic on the
      ORM instance).
    * **None in cooldown** — proceed with the existing happy path.

    Cards the student has never attempted (no ``student_card_state``
    row) are *not* in cooldown — they are new material per thesis
    UC-LEARN-01 Alt 1a.

    **FR-4.3 — retake policy.** The quiz loads through
    :func:`published_queries.get_quiz_for_taking`, which (a) returns
    ``None`` for draft/archived/soft-deleted quizzes (→ 404; students
    can no longer attempt unpublished quizzes) and (b) raises
    :class:`CooldownActive` (→ 429) / :class:`MaxAttemptsReached`
    (→ 409) per the quiz's ``cooldown_hours`` / ``max_attempts`` /
    ``allow_retakes`` columns. Both exceptions propagate to the router.
    """
    # Phase 5: resolve this student's effective timing/retake policy (base quiz
    # columns overridden by any user/group override) and feed it into the gate
    # so an accommodation (extra attempts, extended window) is honoured. We load
    # the published quiz first for resolution, then re-run the gate with the
    # effective values.
    from abridgeai.features.quizzes.services.overrides import (  # noqa: PLC0415
        resolve_policy_for_student,
    )

    _quiz_for_policy = await published_queries.get_published_quiz(db, quiz_id)
    effective = (
        await resolve_policy_for_student(db, _quiz_for_policy, actor.user_id)
        if _quiz_for_policy is not None
        else None
    )
    quiz = await published_queries.get_quiz_for_taking(
        db, quiz_id, actor.user_id, effective=effective,
        password=password, client_ip=client_ip,
    )
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    questions = await _load_quiz_questions_for_taking(db, quiz_id)

    cooldown_map = await _load_cooldown_map(
        db, actor.user_id, [question.id for question in questions]
    )
    if questions and len(cooldown_map) == len(questions):
        cards_due_at = sorted(cooldown_map.items(), key=lambda item: item[1])
        retry_available_at = cards_due_at[0][1]
        raise AllCardsInCooldownError(retry_available_at, cards_due_at)

    available_questions = [q for q in questions if q.id not in cooldown_map]

    next_number = await _next_attempt_number(db, quiz_id, actor.user_id)
    attempt = QuizAttempt(
        quiz_id=quiz_id,
        student_id=actor.user_id,
        attempt_number=next_number,
        idempotency_key=idempotency_key,
    )
    db.add(attempt)
    await flush_or_conflict(db)
    await db.refresh(attempt)

    attempt.cards_in_cooldown = [  # type: ignore[attr-defined]
        {"question_id": qid, "due_at": due_at} for qid, due_at in cooldown_map.items()
    ]

    # Phase 6: when the quiz enables shuffling, deterministically realize a
    # per-attempt question/option order seeded by the attempt id, persist it to
    # attempt.layout, and reorder the take payload. Resume/review re-read layout
    # verbatim so the student always sees the same order.
    if quiz.shuffle_questions or quiz.shuffle_options:
        from abridgeai.features.quizzes.services.shuffle import (  # noqa: PLC0415
            apply_layout,
            build_layout,
        )

        options_by_question = {
            q.id: [o.id for o in getattr(q, "options", []) or []] for q in available_questions
        }
        layout = build_layout(
            attempt.id,
            [q.id for q in available_questions],
            options_by_question,
            shuffle_questions=quiz.shuffle_questions,
            shuffle_options=quiz.shuffle_options,
        )
        attempt.layout = layout
        await flush_or_conflict(db)
        available_questions = apply_layout(available_questions, layout)

    public_quiz = QuizPublic.model_validate(quiz)
    public_questions = [
        QuizQuestionPublic.model_validate(question) for question in available_questions
    ]
    take_payload = QuizForTakingPublic(quiz=public_quiz, questions=public_questions)
    progress = QuizAttemptProgressRead.model_validate(
        {
            "attempt_id": attempt.id,
            "quiz_id": attempt.quiz_id,
            "status": attempt.status,
            "started_at": attempt.started_at,
            "take": take_payload,
            "answers": [],
        }
    )
    return attempt, progress


async def answer_attempt(
    db: AsyncSession,
    attempt_id: UUID,
    payload: object,
    actor: CurrentUser,
) -> tuple[QuizAttemptAnswer, CardReviewResult | None]:
    """Record one answer for an in-flight attempt.

    Computes ``is_correct`` server-side via the type-aware grader so a
    malicious client cannot self-grade. Multiple-choice and true_false
    answers grade by option lookup; short_answer and fill_blank grade
    by comparing the submitted text against the canonical answer
    stored on ``QuizQuestion.original_generated_payload``.

    **FR-4.4 learning loop.** After the answer persists, the SM-2
    review fires via :func:`record_card_review` (SR public API): Q is
    derived from correctness + hint + ρ, EF updates, and the card is
    rescheduled — all inside this transaction. SR failure never blocks
    the answer write: a question without T_exp (draft quiz) or a
    missing question logs a warning and skips the review. The returned
    :class:`CardReviewResult` (or ``None`` when skipped) carries
    ``pending_events`` the router must dispatch **after commit**.

    Duplicate reviews are impossible per attempt: the
    ``uq_quiz_attempt_answers_question`` constraint rejects a second
    answer for the same question before the review would fire.
    """
    del actor
    from sqlalchemy import select  # noqa: PLC0415

    attempt = await _require_attempt(db, attempt_id)

    question_id = payload.question_id  # type: ignore[attr-defined]
    selected_option_id = getattr(payload, "selected_option_id", None)
    answer_text = getattr(payload, "answer_text", None)
    grade = await grade_answer(
        db,
        question_id=question_id,
        selected_option_id=selected_option_id,
        answer_text=answer_text,
    )

    t_actual_ms = getattr(payload, "t_actual_ms", None)
    if t_actual_ms is None:
        t_actual_ms = getattr(payload, "response_time_ms", None)
    hint_used = bool(getattr(payload, "hint_used", False))

    # Phase 4: flag open-response answers that need a human grader (code always;
    # short_answer/fill_blank only when the exact-match auto-grade missed).
    question_type_row = (
        await db.execute(select(QuizQuestion.question_type).where(QuizQuestion.id == question_id))
    ).scalar_one_or_none()
    needs_manual = (
        needs_manual_grade(question_type_row, grade) if question_type_row is not None else False
    )

    # Phase 1: pin the answer to the question's CURRENT revision (its highest
    # revision_no) so a later regrade can judge it against the exact snapshot
    # the student saw. NULL when the question has no revision rows yet.
    graded_revision_id = (
        await db.execute(
            select(QuizQuestionRevision.id)
            .where(QuizQuestionRevision.question_id == question_id)
            .order_by(QuizQuestionRevision.revision_no.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    # Idempotent upsert: a student may re-save a question after changing
    # their answer (or after resuming an interrupted attempt), so key on
    # (attempt_id, question_id) rather than blindly INSERTing — a second
    # INSERT would hit uq_quiz_attempt_answers_question and 409. The
    # unique constraint stays as a backstop against races.
    existing = (
        await db.execute(
            select(QuizAttemptAnswer).where(
                QuizAttemptAnswer.attempt_id == attempt.id,
                QuizAttemptAnswer.question_id == question_id,
            )
        )
    ).scalar_one_or_none()

    is_first_answer = existing is None

    if existing is not None:
        # Edit in place. Latest answer determines the final score (submit
        # sums points_awarded), but see below: SM-2 does NOT re-fire.
        existing.selected_option_id = selected_option_id
        existing.answer_text = answer_text
        existing.is_correct = grade.is_correct
        existing.hint_used = hint_used
        existing.t_actual_ms = t_actual_ms
        existing.points_awarded = grade.points_awarded
        existing.graded_revision_id = graded_revision_id
        existing.needs_manual_grade = needs_manual
        answer = existing
    else:
        answer = QuizAttemptAnswer(
            attempt_id=attempt.id,
            question_id=question_id,
            selected_option_id=selected_option_id,
            answer_text=answer_text,
            is_correct=grade.is_correct,
            hint_used=hint_used,
            t_actual_ms=t_actual_ms,
            points_awarded=grade.points_awarded,
            graded_revision_id=graded_revision_id,
            needs_manual_grade=needs_manual,
        )
        db.add(answer)

    await flush_or_conflict(db)
    await db.refresh(answer)

    # SM-2 grade-once policy (agreed): the spaced-repetition review fires
    # only on the FIRST answer for a (attempt, question). Re-firing on every
    # edit would let a student game their SR schedule by toggling answers and
    # would double-count the card. The stored answer is still updated above,
    # so the student's latest answer counts toward their quiz score — only
    # the SR scheduler is left untouched on edits.
    review_result: CardReviewResult | None = None
    if is_first_answer:
        try:
            review_result = await record_card_review(
                db,
                student_id=attempt.student_id,
                question_id=answer.question_id,
                quiz_attempt_id=attempt.id,
                t_actual_ms=answer.t_actual_ms,
                correct=answer.is_correct,
                hint_used=answer.hint_used,
            )
        except (NotFoundError, ValueError) as exc:
            _logger.warning(
                "sm2_review_skipped",
                attempt_id=str(attempt.id),
                question_id=str(answer.question_id),
                reason=str(exc),
            )
    return answer, review_result


async def _recompute_attempt_score(
    db: AsyncSession,
    attempt: QuizAttempt,
    quiz: Quiz,
) -> tuple[Decimal, Decimal, int, int]:
    """Recompute an attempt's headline numbers from its stored answers.

    Shared by :func:`submit_attempt` (initial grading) and the regrade commit
    path (Phase 1) so both use the identical denominator rule. Returns
    ``(score_points, score_percent, correct_count, question_count)``. Does NOT
    mutate the attempt — the caller assigns the fields it needs.

    Denominator MUST match what the student was actually served: only approved
    questions reach the taking surface, so counting all questions would divide
    by a larger set than the student saw and silently deflate every score.
    """
    from sqlalchemy import func, select  # noqa: PLC0415

    answers = (
        (
            await db.execute(
                select(QuizAttemptAnswer).where(QuizAttemptAnswer.attempt_id == attempt.id)
            )
        )
        .scalars()
        .all()
    )
    question_count_row = await db.execute(
        select(func.count(QuizQuestion.id)).where(
            QuizQuestion.quiz_id == quiz.id,
            QuizQuestion.review_status == "approved",
        )
    )
    question_count = int(question_count_row.scalar_one()) or len(answers) or 1
    score_points = sum((answer.points_awarded for answer in answers), Decimal("0"))
    score_percent = (score_points / Decimal(question_count)) * Decimal("100")
    correct_count = sum(1 for answer in answers if answer.is_correct)
    return score_points, score_percent, correct_count, question_count


async def submit_attempt(
    db: AsyncSession,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttempt:
    """Grade and finalize an attempt.

    Computes ``score_percent = score_points / question_count * 100`` and
    flips ``passed`` based on :attr:`Quiz.passing_score_percent`. The
    division uses the quiz's question count (not the answer count) so
    skipped questions count against the score — matching legacy parity.
    """
    del actor
    attempt = await _require_attempt(db, attempt_id)
    quiz = await _require_quiz(db, attempt.quiz_id)

    # Phase 6: deadline enforcement at submit time. If the attempt is past its
    # (grace-extended) deadline and the quiz is 'autoabandon', expire it with no
    # grade; otherwise grade normally (autosubmit / graceperiod both grade what
    # was answered). This makes a late submit correct even if the sweep never ran.
    from abridgeai.features.quizzes.services import timing as _timing  # noqa: PLC0415

    now = utcnow()
    eff = _timing.resolve_effective_timing(quiz)
    overdue = _timing.is_overdue(
        attempt.started_at,
        eff,
        grace_period_seconds=(
            quiz.grace_period_seconds if quiz.overdue_handling == "graceperiod" else None
        ),
        now=now,
        due_at=quiz.due_at,
        hard_due=False,
    )
    if overdue and quiz.overdue_handling == "autoabandon":
        return await _expire_attempt(db, attempt, now=now)

    return await _finalize_attempt(db, attempt, quiz, now=now)


async def _finalize_attempt(
    db: AsyncSession,
    attempt: QuizAttempt,
    quiz: Quiz,
    *,
    now: datetime | None = None,
) -> QuizAttempt:
    """Grade + close an in_progress attempt as 'submitted'.

    Idempotent: returns unchanged if the attempt is not in_progress. Shared by
    :func:`submit_attempt` and the Phase 6 overdue sweep so both use identical
    scoring. Attaches response-only correct_count/total_questions.
    """
    now = now or utcnow()
    if attempt.status != "in_progress":
        return attempt

    score_points, score_percent, correct_count, question_count = await _recompute_attempt_score(
        db, attempt, quiz
    )
    attempt.status = "submitted"
    attempt.submitted_at = now
    attempt.time_taken_seconds = int((now - attempt.started_at).total_seconds())
    attempt.score_points = score_points
    attempt.score_percent = score_percent
    attempt.passed = score_percent >= quiz.passing_score_percent
    await flush_or_conflict(db)
    await db.refresh(attempt)
    setattr(attempt, "correct_count", correct_count)  # noqa: B010 -- dynamic, not column
    setattr(attempt, "total_questions", question_count)  # noqa: B010 -- dynamic, not column
    return attempt


async def _expire_attempt(
    db: AsyncSession,
    attempt: QuizAttempt,
    *,
    now: datetime | None = None,
) -> QuizAttempt:
    """Close an in_progress attempt as 'expired' with no grade (autoabandon)."""
    now = now or utcnow()
    if attempt.status != "in_progress":
        return attempt
    attempt.status = "expired"
    attempt.submitted_at = now
    attempt.time_taken_seconds = int((now - attempt.started_at).total_seconds())
    await flush_or_conflict(db)
    await db.refresh(attempt)
    return attempt


async def get_attempt_history(
    db: AsyncSession,
    quiz_id: UUID,
    actor: CurrentUser,
) -> list[QuizAttempt]:
    """Return every attempt the calling student has against ``quiz_id``."""
    from sqlalchemy import select  # noqa: PLC0415

    attempts = (
        (
            await db.execute(
                select(QuizAttempt)
                .where(
                    QuizAttempt.quiz_id == quiz_id,
                    QuizAttempt.student_id == actor.user_id,
                )
                .order_by(QuizAttempt.attempt_number)
            )
        )
        .scalars()
        .all()
    )
    return list(attempts)


async def get_attempt_review(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttemptReviewRead | None:
    """Project a submitted attempt + its answers + correct options.

    Returns ``None`` (router → 404) when the attempt doesn't belong to the
    caller or hasn't been submitted yet — review surface only opens after
    the student locks in their answers.
    """
    attempt = await published_queries.get_attempt_for_review(
        db, attempt_id=attempt_id, user_id=actor.user_id
    )
    if attempt is None:
        return None

    answers_by_question: dict[UUID, QuizAttemptAnswer] = {a.question_id: a for a in attempt.answers}

    questions_with_options = await published_queries.list_quiz_questions_with_options(
        db, attempt.quiz_id
    )

    # Phase 2: resolve the teacher-configured review-visibility flags for the
    # active time-window (immediately_after / later_while_open / after_close) and
    # mask the payload server-side so a hidden field never leaves the service.
    from abridgeai.features.quizzes.schemas.attempt import (  # noqa: PLC0415
        ReviewVisibilityFlags,
    )
    from abridgeai.features.quizzes.services.review_visibility import (  # noqa: PLC0415
        resolve_review_visibility,
    )

    quiz = await _require_quiz(db, attempt.quiz_id)
    vis = resolve_review_visibility(quiz, attempt, utcnow())

    review_questions: list[QuizAttemptReviewQuestion] = []
    for question, options in questions_with_options:
        ans = answers_by_question.get(question.id)
        # Options carry is_correct; strip the correct-answer signal when hidden.
        review_options = [QuizAttemptReviewOption.model_validate(opt) for opt in options]
        if not vis.show_correct_answers:
            for opt in review_options:
                opt.is_correct = False
        review_questions.append(
            QuizAttemptReviewQuestion(
                question_id=question.id,
                position=question.position,
                question_type=question.question_type,
                prompt_text=question.prompt_text,
                explanation=question.explanation if vis.show_explanation else None,
                hint_text=question.hint_text,
                options=review_options,
                selected_option_id=ans.selected_option_id if ans else None,
                answer_text=ans.answer_text if ans else None,
                is_correct=(ans.is_correct if ans else False) if vis.show_correctness else False,
                points_awarded=(ans.points_awarded if ans else Decimal("0"))
                if vis.show_points
                else Decimal("0"),
                hint_used=ans.hint_used if ans else False,
                t_actual_ms=ans.t_actual_ms if ans else None,
            )
        )

    attempt_read = QuizAttemptRead.model_validate(attempt)
    if not vis.show_score:
        attempt_read.score_points = None
        attempt_read.score_percent = None
        attempt_read.passed = None
        attempt_read.correct_count = None

    # Phase 8: attach the matched overall grade-band feedback, but only when the
    # score is visible (feedback is a review-time disclosure like the score).
    overall_text: str | None = None
    overall_format: str | None = None
    if vis.show_score:
        from abridgeai.features.quizzes.services import feedback as _fb  # noqa: PLC0415

        band = await _fb.select_overall_feedback(
            db, quiz_id=attempt.quiz_id, score_percent=attempt.score_percent
        )
        if band is not None:
            overall_text = band.feedback_text
            overall_format = band.feedback_format

    return QuizAttemptReviewRead(
        attempt=attempt_read,
        questions=review_questions,
        visibility=ReviewVisibilityFlags(
            show_score=vis.show_score,
            show_correctness=vis.show_correctness,
            show_correct_answers=vis.show_correct_answers,
            show_explanation=vis.show_explanation,
            show_points=vis.show_points,
        ),
        overall_feedback_text=overall_text,
        overall_feedback_format=overall_format,
    )


async def get_attempt_progress(
    db: AsyncSession,
    *,
    attempt_id: UUID,
    actor: CurrentUser,
) -> QuizAttemptProgressRead | None:
    """Project an in-progress attempt into a no-leak resume payload.

    Returns ``None`` (router → 404) when the attempt doesn't belong to the
    caller or is no longer in flight. Only the student's own inputs are
    echoed — correctness / points are dropped by
    :class:`QuizAttemptProgressAnswer` because the attempt is still open
    and a leak here would let a student probe answers by save-and-read.
    """
    attempt = await published_queries.get_in_progress_attempt(
        db, attempt_id=attempt_id, user_id=actor.user_id
    )
    if attempt is None:
        return None

    # Rebuild the no-leak take payload so the client can re-render the quiz
    # without POSTing a fresh attempt (which would create a duplicate). We
    # load the quiz row directly (NOT via get_quiz_for_taking, which would
    # re-run the retake/cooldown gate and could 409/429 a legitimate resume)
    # and project through the same QuizQuestionPublic schema that drops
    # is_correct. All published questions are shown; the saved answers are a
    # subset keyed by question_id.
    quiz = await _require_quiz(db, attempt.quiz_id)
    questions = await _load_quiz_questions_for_taking(db, attempt.quiz_id)
    take_payload = QuizForTakingPublic(
        quiz=QuizPublic.model_validate(quiz),
        questions=[QuizQuestionPublic.model_validate(q) for q in questions],
    )

    return QuizAttemptProgressRead.model_validate(
        {
            "attempt_id": attempt.id,
            "quiz_id": attempt.quiz_id,
            "status": attempt.status,
            "started_at": attempt.started_at,
            "take": take_payload,
            "answers": [
                QuizAttemptProgressAnswer(
                    question_id=ans.question_id,
                    selected_option_id=ans.selected_option_id,
                    answer_text=ans.answer_text,
                    hint_used=ans.hint_used,
                    t_actual_ms=ans.t_actual_ms,
                )
                for ans in attempt.answers
            ],
        }
    )


__all__ = [
    "AllCardsInCooldownError",
    "CooldownActive",
    "MaxAttemptsReached",
    "QuizClosed",
    "QuizNotYetOpen",
    "QuizPasswordIncorrect",
    "QuizPasswordRequired",
    "QuizSubnetBlocked",
    "answer_attempt",
    "get_attempt_history",
    "get_attempt_progress",
    "get_attempt_review",
    "get_published_quiz",
    "start_attempt",
    "submit_attempt",
]
