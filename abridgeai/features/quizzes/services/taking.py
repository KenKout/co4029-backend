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
    QuizAttemptProgressRead,
)
from abridgeai.features.quizzes.schemas.public import (
    QuizForTakingPublic,
    QuizPublic,
    QuizQuestionPublic,
)
from abridgeai.features.quizzes.services.attempt_reading import (  # noqa: F401
    get_attempt_history,
    get_attempt_progress,
    get_attempt_review,
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


class InterviewPassRequiredError(AppError):
    """FR-5.3: the quiz's module gates its spaced-repetition unlock behind a
    passed interview the student has not yet passed.

    Raised by :func:`_ensure_interview_pass_lock` when the quiz belongs to a
    module carrying a published interview config with
    ``lock_quiz_ef_until_pass = TRUE`` and the student has no completed+passed
    interview session for that module. The router maps this to HTTP 403 with an
    ``interview_pass_required`` payload so the client can deep-link the pending
    interview. Carries the blocking module + interview config ids for that.
    """

    def __init__(self, module_id: UUID, interview_config_id: UUID) -> None:
        super().__init__(
            f"Interview pass required for module {module_id} "
            f"(interview config {interview_config_id}) before this quiz"
        )
        self.module_id = module_id
        self.interview_config_id = interview_config_id


async def _ensure_interview_pass_lock(db: AsyncSession, *, quiz_id: UUID, student_id: UUID) -> None:
    """FR-5.3 server-side gate: block a quiz attempt when the quiz's module has a
    published interview with ``lock_quiz_ef_until_pass`` the student hasn't passed.

    Resolution:

    1. Emergency off-switch — when ``settings.lesson_gating_enforced`` is False
       the gate is a no-op and never touches the DB.
    2. Look up a *locking* interview config for the quiz's module: published,
       ``lock_quiz_ef_until_pass = TRUE``, not soft-deleted. None found → no gate.
    3. Otherwise the student must have a completed+passed interview session for
       that module (:func:`has_passing_interview_for_module`); if not, raise
       :class:`InterviewPassRequiredError`.

    ``get_settings`` and the SR public helper are imported lazily so the unit
    test's ``patch`` on the source module attributes takes effect at call time
    (a module-top ``from ... import`` would bind an unpatched reference).
    """
    from abridgeai.core.config import get_settings  # noqa: PLC0415

    if not get_settings().lesson_gating_enforced:
        return

    from sqlalchemy import text  # noqa: PLC0415

    result = await db.execute(
        text(
            "SELECT c.id, c.module_id "
            "FROM interview_configs c "
            "JOIN quizzes q ON q.module_id = c.module_id "
            "WHERE q.id = :quiz_id "
            "AND c.status = 'published' "
            "AND c.lock_quiz_ef_until_pass = TRUE "
            "AND c.deleted_at IS NULL "
            "AND q.deleted_at IS NULL "
            "LIMIT 1"
        ),
        {"quiz_id": str(quiz_id)},
    )
    row = result.first()
    if row is None:
        return  # no locking interview config for this module → no gate

    interview_config_id, module_id = row[0], row[1]

    from abridgeai.features.spaced_repetition.api import public as sr_public  # noqa: PLC0415

    passed = await sr_public.has_passing_interview_for_module(
        db, student_id=student_id, module_id=module_id
    )
    if not passed:
        raise InterviewPassRequiredError(module_id, interview_config_id)


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
    positions: dict[UUID, tuple[int, str]] = {}
    if outcome_ids:
        from sqlalchemy import text as _text  # noqa: PLC0415

        # Recursive CTE rebuilds the dotted code (L.O.1.2.1) by walking each
        # outcome's parent chain, so the student-facing label matches the
        # hierarchical teacher codes. position is the leaf's own sibling
        # position (back-compat).
        rows = (
            await db.execute(
                _text(
                    """
                    WITH RECURSIVE coded AS (
                        SELECT id, parent_id, position, position::text AS code
                        FROM course_learning_outcomes
                        WHERE parent_id IS NULL AND deleted_at IS NULL
                        UNION ALL
                        SELECT c.id, c.parent_id, c.position,
                               coded.code || '.' || c.position::text
                        FROM course_learning_outcomes c
                        JOIN coded ON c.parent_id = coded.id
                        WHERE c.deleted_at IS NULL
                    )
                    SELECT id, position, code FROM coded WHERE id = ANY(:ids)
                    """
                ),
                {"ids": list(set(outcome_ids))},
            )
        ).all()
        positions = {row[0]: (row[1], row[2]) for row in rows}
    for question in questions:
        lo_id = question.learning_outcome_id
        resolved = positions.get(lo_id) if lo_id is not None else None
        question.outcome_position = resolved[0] if resolved else None  # type: ignore[attr-defined]
        question.outcome_code = resolved[1] if resolved else None  # type: ignore[attr-defined]
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


def _renumber_display_positions(questions: list[QuizQuestionPublic]) -> None:
    """Rewrite ``position`` on the take-payload DTOs to their display slot.

    The student-facing ``position`` must encode the order the student should
    SEE the questions in (1..N), not the authored order. This matters for
    shuffled attempts: :func:`apply_layout` reorders the list but leaves the
    original authored ``position`` on each row, and the SPA defensively
    re-sorts questions AND options by ``position`` before rendering
    (course-quiz.tsx, QuestionRenderer.tsx). Without this renumber those sorts
    would restore the authored order and silently undo the shuffle — leaking
    the canonical sequence and defeating the anti-cheat feature.

    Mutates the passed DTOs in place. Safe because these are detached Pydantic
    projections built for this one response, never the session-tracked ORM
    rows (rewriting the ORM ``position`` would corrupt the stored order on
    flush). Options are renumbered within each question in their already-
    ordered (post-layout) sequence. Grading is keyed by question_id/option_id,
    so it is unaffected by any of this.
    """
    for q_index, question in enumerate(questions, start=1):
        question.position = q_index
        for o_index, option in enumerate(question.options, start=1):
            option.position = o_index


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
    # FR-5.3 interview-pass gate: fail fast before creating any attempt state
    # when the quiz's module gates its unlock behind a passed interview the
    # student hasn't passed. No-op when the emergency switch is off or the module
    # carries no locking interview config. Raises InterviewPassRequiredError
    # (router → HTTP 403) so the client can deep-link the pending interview.
    await _ensure_interview_pass_lock(db, quiz_id=quiz_id, student_id=actor.user_id)

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
        db,
        quiz_id,
        actor.user_id,
        effective=effective,
        password=password,
        client_ip=client_ip,
    )
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")

    # Tenancy gate: organizations do not share quizzes. A student may only
    # attempt a quiz whose course belongs to an organization they actively
    # belong to. Resolve the quiz's course -> organization via the courses
    # public API and the caller's membership via the access-control public API
    # (both cross-feature reads through the sanctioned surfaces). Raise the same
    # NotFoundError as a missing quiz so the endpoint cannot be used to probe
    # which quiz ids exist in another tenant.
    from abridgeai.features.access_control.api.public import (  # noqa: PLC0415
        is_user_member_of_org,
    )
    from abridgeai.features.courses.api import public as courses_api  # noqa: PLC0415

    course = await courses_api.get_course_by_id(db, quiz.course_id)
    if course is None or not await is_user_member_of_org(
        db, user_id=actor.user_id, org_id=course.organization_id
    ):
        raise NotFoundError(f"Quiz {quiz_id} not found")

    questions = await _load_quiz_questions_for_taking(db, quiz_id)

    # Card-level (SR) cooldown only applies when the quiz itself sets a
    # cooldown. A quiz with no ``cooldown_hours`` must not inherit the SR
    # failure cooldown as a hidden default — the student can retry immediately.
    cooldown_map: dict[UUID, datetime] = {}
    if quiz.cooldown_hours:
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
    # Encode the display order (1..N) into position so the SPA's defensive
    # position-sorts render the shuffled order rather than undoing it.
    _renumber_display_positions(public_questions)
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
    # Phase 9: refresh the student's materialised grade-of-record from all their
    # completed attempts (grading_method-aware). Participates in this transaction.
    from abridgeai.features.quizzes.services.gradebook import (  # noqa: PLC0415
        recompute_final_grade,
    )

    await recompute_final_grade(db, quiz, attempt.student_id)
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
