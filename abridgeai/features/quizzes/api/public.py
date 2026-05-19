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


__all__ = [
    "AttemptScoreDTO",
    "GenerationRunDTO",
    "GenerationRunKind",
    "GenerationRunSourceScopeKind",
    "QuestionWithQuizDTO",
    "create_generation_run",
    "get_attempt_score",
    "get_generation_run",
    "get_question_with_quiz_context",
    "get_quiz_question_id_set_by_lesson",
    "get_t_exp_for_question",
]
