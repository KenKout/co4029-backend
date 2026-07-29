"""Cross-feature public API for the quizzes feature.

This module is the ONLY surface other features may import from
``abridgeai.features.quizzes`` (per the import-linter
``features-are-independent`` contract). All functions are
module-level ``async`` callables; there is no ``QuizzesPublicAPI``
god-class. Returns are Pydantic v2 DTOs, never ORM rows, so
consumers cannot accidentally mutate quiz state.

Surface
-------
``get_question_with_quiz_context``
    SR remediation seed — joins ``quiz_questions`` to its parent
    ``quizzes`` and returns the prompt + chunk source_refs together
    with the surrounding ``course_id`` / ``module_id``.

``get_t_exp_for_question``
    Reads ``quiz_questions.expected_response_time_ms`` for the SM-2
    ρ-derivation in :mod:`spaced_repetition.services.review`.

``get_attempt_score``
    Reads a single ``quiz_attempts`` row's score percentile, used by
    the interviews Gap Report.

``get_quiz_question_id_set_by_lesson``
    SR analytics + published surfaces ask "which quiz_question.ids
    belong to lesson L?" once per request — return a frozen set so
    callers can do membership tests without re-querying.

``create_generation_run``
    THE blessed factory. Replaces every direct
    ``GenerationRun(generation_type=..., ...)`` call site outside the
    quizzes feature. Wave 5 will lock this in by adding an
    import-linter forbid rule on
    ``abridgeai.features.interviews -> abridgeai.features.quizzes.models.GenerationRun``;
    until then the factory is the migration target.

``get_generation_run``
    Read-only fetch by id — used by the interviews AI pipeline
    dispatcher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.ai.models import GenerationRun
from abridgeai.features.quizzes.api._dto import (
    AttemptScoreDTO,
    GenerationRunDTO,
    GenerationRunKind,
    GenerationRunSourceScopeKind,
    QuestionWithQuizDTO,
)
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizQuestion,
    QuizQuestionOption,
    QuizSourceLesson,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_question_with_quiz_context(
    db: AsyncSession,
    question_id: UUID,
) -> QuestionWithQuizDTO | None:
    """Return question + parent-quiz context, or ``None`` if not found.

    Soft-deleted ``quiz_questions`` and ``quizzes`` rows are filtered
    by the ORM loader-criteria listener (``core.db.soft_delete``), so
    this returns ``None`` for them as well as for genuine misses.
    """

    stmt = (
        select(
            QuizQuestion.id,
            QuizQuestion.quiz_id,
            QuizQuestion.prompt_text,
            QuizQuestion.source_refs,
            Quiz.course_id,
            Quiz.module_id,
            Quiz.initial_ef,
        )
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .where(QuizQuestion.id == question_id)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return None
    return QuestionWithQuizDTO(
        question_id=row.id,
        quiz_id=row.quiz_id,
        prompt_text=row.prompt_text,
        source_refs=list(row.source_refs or []),
        course_id=row.course_id,
        module_id=row.module_id,
        initial_ef=row.initial_ef,
    )


async def get_t_exp_for_question(
    db: AsyncSession,
    question_id: UUID,
) -> int | None:
    """Return ``expected_response_time_ms`` for a quiz question.

    The SR review service derives the SM-2 ρ multiplier from this
    value. ``None`` covers both not-found and unpublished questions
    (T7.5.9 publish gate enforces NOT NULL at publish time, but draft
    questions may legitimately have it unset).
    """

    stmt = select(QuizQuestion.expected_response_time_ms).where(QuizQuestion.id == question_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_attempt_score(
    db: AsyncSession,
    attempt_id: UUID,
) -> AttemptScoreDTO | None:
    """Return the minimal score-shape DTO for one quiz attempt."""

    attempt = await db.get(QuizAttempt, attempt_id)
    if attempt is None:
        return None
    return AttemptScoreDTO.model_validate(
        {
            "attempt_id": attempt.id,
            "quiz_id": attempt.quiz_id,
            "student_id": attempt.student_id,
            "status": attempt.status,
            "score_percent": attempt.score_percent,
            "passed": attempt.passed,
        }
    )


async def get_quiz_question_id_set_by_lesson(
    db: AsyncSession,
    lesson_id: UUID,
) -> frozenset[UUID]:
    """All ``quiz_question.id`` values whose parent quiz sources ``lesson_id``.

    Joins ``quiz_source_lessons`` -> ``quizzes`` -> ``quiz_questions``.
    Soft-deleted rows are filtered by the ORM loader-criteria.
    Returns ``frozenset`` so callers can do safe membership lookups
    and pass it through cache layers without worrying about mutation.
    """

    stmt = (
        select(QuizQuestion.id)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(QuizSourceLesson, QuizSourceLesson.quiz_id == Quiz.id)
        .where(QuizSourceLesson.lesson_id == lesson_id)
    )
    result = await db.execute(stmt)
    return frozenset(result.scalars().all())


async def create_generation_run(
    db: AsyncSession,
    *,
    kind: GenerationRunKind,
    source_scope_kind: GenerationRunSourceScopeKind,
    course_id: UUID | None = None,
    module_id: UUID | None = None,
    lesson_id: UUID | None = None,
    requested_by: UUID | None = None,
    config_json: dict[str, Any] | None = None,
    dedup_key: str | None = None,
) -> GenerationRunDTO:
    """Create a ``generation_runs`` row of the requested ``kind``.

    The single blessed cross-feature instantiation path. Replaces
    direct ``from features.quizzes.models import GenerationRun;
    GenerationRun(...)`` calls in the interviews feature
    (``services/authoring.py``, ``ai/pipelines/generation.py``).

    Status is fixed at ``"pending"`` — the caller's pipeline takes it
    from there. ``started_at`` / ``finished_at`` are left ``NULL``;
    the dispatcher stamps them. ``created_by`` / ``updated_by`` are
    populated by the audit listener from the current actor contextvar
    (T3) — the caller does NOT pass them.

    The caller is responsible for committing the surrounding unit of
    work; this function only ``flush()``-es so the row gets an id
    that subsequent statements (e.g. ``module_items.quiz_id`` FK
    binds) can reference inside the same transaction.
    """

    run = GenerationRun(
        generation_type=kind,
        source_scope_kind=source_scope_kind,
        course_id=course_id,
        module_id=module_id,
        lesson_id=lesson_id,
        requested_by=requested_by,
        status="pending",
        config_json=dict(config_json) if config_json is not None else {},
        dedup_key=dedup_key,
    )
    db.add(run)
    await db.flush()
    await db.refresh(run)
    return GenerationRunDTO.model_validate(run)


async def get_generation_run(
    db: AsyncSession,
    run_id: UUID,
) -> GenerationRunDTO | None:
    """Fetch one ``generation_runs`` row by id."""

    run = await db.get(GenerationRun, run_id)
    if run is None:
        return None
    return GenerationRunDTO.model_validate(run)


async def deep_clone_quiz(
    db: AsyncSession,
    *,
    source_quiz_id: UUID,
    target_module_id: UUID,
    actor_id: UUID,
    title_suffix: str = "",
) -> UUID:
    """Deep-clone a quiz (and every authoring child) into ``target_module_id``.

    Cross-feature entry point for the courses module/item duplicate flow. The
    caller (courses authoring) cannot import quizzes models directly under the
    feature-independence contract, so the whole quiz subtree is cloned here.

    What is copied:

    * the :class:`Quiz` row itself, always forced to ``status='draft'`` with
      ``published_at`` cleared — a duplicate must never inherit a published
      state (matches the user requirement: duplicated data is always
      unpublished).
    * every non-deleted :class:`QuizQuestion`, each forced to
      ``review_status='pending'`` (reviewed/published fields cleared) with its
      full option set (:class:`QuizQuestionOption`).
    * ``quiz_source_lessons`` link rows (pure attribution, no runtime state).

    What is intentionally NOT copied: attempts, answers, grades, overrides,
    regrade runs, statistics — all runtime rows keyed on the source quiz.

    Returns the new quiz id. Flushes but does not commit; the caller owns the
    surrounding transaction.
    """
    source = (
        await db.execute(select(Quiz).where(Quiz.id == source_quiz_id))
    ).scalar_one_or_none()
    if source is None:
        raise ValueError(f"Quiz {source_quiz_id} not found")

    clone = Quiz(
        course_id=source.course_id,
        module_id=target_module_id,
        title=f"{source.title}{title_suffix}",
        description=source.description,
        status="draft",
        time_limit_seconds=source.time_limit_seconds,
        passing_score_percent=source.passing_score_percent,
        grading_method=source.grading_method,
        allow_retakes=source.allow_retakes,
        max_attempts=source.max_attempts,
        cooldown_hours=source.cooldown_hours,
        shuffle_questions=source.shuffle_questions,
        shuffle_options=source.shuffle_options,
        show_hints=source.show_hints,
        initial_ef=source.initial_ef,
        min_ef_for_unlock=source.min_ef_for_unlock,
        coverage_threshold=source.coverage_threshold,
        reminders_enabled=source.reminders_enabled,
        generation_instructions=source.generation_instructions,
        published_at=None,
        available_from=source.available_from,
        available_until=source.available_until,
        due_at=source.due_at,
        review_options=dict(source.review_options or {}),
        overdue_handling=source.overdue_handling,
        grace_period_seconds=source.grace_period_seconds,
        require_password=source.require_password,
        require_subnet=source.require_subnet,
        browser_security=source.browser_security,
        delay1_seconds=source.delay1_seconds,
        delay2_seconds=source.delay2_seconds,
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(clone)
    await db.flush()

    questions = (
        (
            await db.execute(
                select(QuizQuestion)
                .where(QuizQuestion.quiz_id == source_quiz_id)
                .where(QuizQuestion.deleted_at.is_(None))
                .order_by(QuizQuestion.position)
            )
        )
        .scalars()
        .all()
    )
    q_ids = [q.id for q in questions]
    options_by_q: dict[UUID, list[QuizQuestionOption]] = {}
    if q_ids:
        opt_rows = (
            (
                await db.execute(
                    select(QuizQuestionOption)
                    .where(QuizQuestionOption.question_id.in_(q_ids))
                    .where(QuizQuestionOption.deleted_at.is_(None))
                    .order_by(QuizQuestionOption.position)
                )
            )
            .scalars()
            .all()
        )
        for opt in opt_rows:
            options_by_q.setdefault(opt.question_id, []).append(opt)

    for src_q in questions:
        q_clone = QuizQuestion(
            quiz_id=clone.id,
            learning_outcome_id=src_q.learning_outcome_id,
            position=src_q.position,
            question_type=src_q.question_type,
            prompt_text=src_q.prompt_text,
            hint_text=src_q.hint_text,
            explanation=src_q.explanation,
            difficulty=src_q.difficulty,
            bloom_level=src_q.bloom_level,
            review_status="pending",
            expected_response_time_ms=src_q.expected_response_time_ms,
            expected_ef_ceiling=src_q.expected_ef_ceiling,
            source_refs=list(src_q.source_refs or []),
            original_generated_payload=(
                dict(src_q.original_generated_payload)
                if src_q.original_generated_payload
                else None
            ),
            imported_from_question_id=src_q.id,
            prompt_format=src_q.prompt_format,
            hint_format=src_q.hint_format,
            explanation_format=src_q.explanation_format,
            single_answer=src_q.single_answer,
            answer_numbering=src_q.answer_numbering,
            numeric_answer=src_q.numeric_answer,
            numeric_tolerance=src_q.numeric_tolerance,
            match_pairs=(list(src_q.match_pairs) if src_q.match_pairs is not None else None),
            ordering_sequence=(
                list(src_q.ordering_sequence) if src_q.ordering_sequence is not None else None
            ),
            category_id=src_q.category_id,
            created_by=actor_id,
            updated_by=actor_id,
        )
        db.add(q_clone)
        await db.flush()
        for opt in options_by_q.get(src_q.id, []):
            db.add(
                QuizQuestionOption(
                    question_id=q_clone.id,
                    option_key=opt.option_key,
                    option_text=opt.option_text,
                    is_correct=opt.is_correct,
                    position=opt.position,
                    option_format=opt.option_format,
                    grade_fraction=opt.grade_fraction,
                    feedback_text=opt.feedback_text,
                    feedback_format=opt.feedback_format,
                    created_by=actor_id,
                    updated_by=actor_id,
                )
            )

    src_links = (
        (
            await db.execute(
                select(QuizSourceLesson).where(QuizSourceLesson.quiz_id == source_quiz_id)
            )
        )
        .scalars()
        .all()
    )
    for link in src_links:
        db.add(QuizSourceLesson(quiz_id=clone.id, lesson_id=link.lesson_id))

    await db.flush()
    return clone.id


__all__ = [
    "AttemptScoreDTO",
    "GenerationRunDTO",
    "GenerationRunKind",
    "GenerationRunSourceScopeKind",
    "QuestionWithQuizDTO",
    "create_generation_run",
    "deep_clone_quiz",
    "get_attempt_score",
    "get_generation_run",
    "get_question_with_quiz_context",
    "get_quiz_question_id_set_by_lesson",
    "get_t_exp_for_question",
]
