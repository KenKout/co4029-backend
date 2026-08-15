"""Quiz generation-run dispatch (create run → enqueue ARQ task).

Split out of :mod:`services.authoring` to keep the authoring service
under the feature's god-file cap. Holds the two run-dispatch entry
points (:func:`start_generation_run`, :func:`regenerate_question`) plus
the coverage-mode preflight (:func:`_require_embedded_chunks`).

Authoring-module helpers (``_resolve_module_course``, ``_require_quiz``,
``_require_question``, ``_quiz_has_in_flight_run``,
``_add_quiz_source_lessons``) stay in :mod:`services.authoring` and are
imported lazily here to avoid a module-level import cycle (authoring
re-exports these three functions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, ConflictError
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)
from abridgeai.features.quizzes.schemas import QuizGenerationRequest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _require_embedded_chunks(db: AsyncSession, lesson_ids: list[UUID]) -> None:
    """Reject coverage generation when the source lessons have no chunks.

    Coverage mode groups pre-existing ``document_chunks`` into an outline;
    it cannot generate anything if the lesson's materials were never
    embedded (never processed, or embedding failed upstream — e.g. the
    provider token lacking embedding-model access). Raises ``AppError`` so
    the router maps it to a 400 with an actionable message instead of the
    worker dying later with the opaque "no document chunks found".

    Raw SQL against ``document_chunks.lesson_id`` keeps the quizzes feature
    from importing ``features.materials`` (import-linter contract).
    """
    from sqlalchemy import text as sa_text  # noqa: PLC0415

    if not lesson_ids:
        raise AppError("Select at least one source lesson to generate from.")
    row = (
        await db.execute(
            sa_text(
                "SELECT count(*) FROM document_chunks "
                "WHERE lesson_id = ANY(CAST(:lesson_ids AS uuid[]))"
            ),
            {"lesson_ids": [str(x) for x in lesson_ids]},
        )
    ).scalar()
    if not row:
        raise AppError(
            "The selected lesson has no processed content yet. Reprocess the "
            "lesson's materials (or switch generation to topic mode) before "
            "generating a quiz in coverage mode."
        )


async def start_generation_run(
    db: AsyncSession,
    module_id: UUID,
    payload: QuizGenerationRequest,
    actor: CurrentUser,
    *,
    arq_pool: object | None,
) -> GenerationRun:
    """Create a quiz (or attach to an existing one), persist a
    :class:`GenerationRun`, and enqueue the ``run_quiz_generation_task``
    ARQ task.

    This is the only authoring helper that commits inline — the worker
    needs to read the run row out of band, so the transaction must be
    committed before :func:`enqueue_job` returns. The task name is the
    canonical Python function name registered on the ARQ worker
    (T5.15: reconciled from the legacy ``generate_quiz`` alias).

    Phase 2 of the FR-5 schema port (T5.14): the legacy ``getattr``
    defensive accessors were removed in favour of direct attribute
    access. The schema layer (``QuizGenerationRequest``) is now strict
    (``extra="forbid"``, all fields typed), so every name read here is
    guaranteed to resolve. If you find yourself reaching for
    ``getattr`` again, fix the schema first — don't smuggle in
    untyped fields through the service.
    """
    from abridgeai.features.quizzes.services.authoring import (  # noqa: PLC0415
        _add_quiz_source_lessons,
        _quiz_has_in_flight_run,
        _require_quiz,
        _resolve_module_course,
    )

    course_id = await _resolve_module_course(db, module_id)

    quiz: Quiz | None = None
    if payload.quiz_id is not None:
        quiz = await _require_quiz(db, payload.quiz_id)
        if quiz.module_id != module_id:
            raise AppError("Quiz must belong to this module")
        if await _quiz_has_in_flight_run(db, quiz.id):
            raise ConflictError("quiz_generation_in_progress")
        # Preflight (coverage mode only): coverage generation needs the source
        # lessons' materials to already be embedded into document_chunks. If a
        # lesson was never processed (or embedding failed upstream), the worker
        # dies deep in the pipeline with a cryptic "no document chunks found"
        # and — worse, pre-fix — only AFTER wiping the quiz's existing
        # questions. Reject early with a clear, actionable message and BEFORE
        # the wipe, so the teacher keeps their current questions and knows to
        # reprocess the lesson (or switch to topic mode).
        if str(payload.generation_mode or "topic").strip().lower() == "coverage":
            await _require_embedded_chunks(db, payload.source_lesson_ids)
        if not payload.append:
            from sqlalchemy import delete as sa_delete  # noqa: PLC0415
            from sqlalchemy import select as sa_select  # noqa: PLC0415
            from sqlalchemy import text as sa_text  # noqa: PLC0415

            # Regenerating with append=false wipes the quiz's existing
            # questions. A bare DELETE on quiz_questions bypasses the ORM
            # relationships (no delete cascade) and trips the child tables'
            # foreign keys, which are ondelete=NO ACTION (options, revisions,
            # attempt_answers, student_quiz_card_state). The raw IntegrityError
            # is not UNIQUE, so flush_or_conflict re-raises it and the router
            # turns it into a 500. This bit any teacher regenerating a quiz
            # that already had questions (and, once students had answered,
            # would trip on the attempt/card-state rows too). Delete children
            # first, in FK order, then the questions.
            question_id_subq = sa_select(QuizQuestion.id).where(QuizQuestion.quiz_id == quiz.id)
            await db.execute(
                sa_delete(QuizQuestionOption).where(
                    QuizQuestionOption.question_id.in_(question_id_subq)
                )
            )
            await db.execute(
                sa_delete(QuizQuestionRevision).where(
                    QuizQuestionRevision.question_id.in_(question_id_subq)
                )
            )
            await db.execute(
                sa_delete(QuizAttemptAnswer).where(
                    QuizAttemptAnswer.question_id.in_(question_id_subq)
                )
            )
            # student_quiz_card_state has no ORM model in this feature (ported
            # separately with the SR scheduler) and its question_id FK is
            # NO ACTION on the live schema, so clear it via raw SQL.
            await db.execute(
                sa_text(
                    "DELETE FROM student_quiz_card_state "
                    "WHERE question_id IN (SELECT id FROM quiz_questions "
                    "WHERE quiz_id = :quiz_id)"
                ),
                {"quiz_id": str(quiz.id)},
            )
            await db.execute(sa_delete(QuizQuestion).where(QuizQuestion.quiz_id == quiz.id))
            await flush_or_conflict(db)

    base_config = dict(payload.config_json)
    coverage_dump: dict[str, object] | None = (
        payload.coverage_options.model_dump() if payload.coverage_options is not None else None
    )
    # Structured FR-5 fields shadow ``config_json`` on conflict — the
    # schema layer documents this contract, so the merge order matters.
    generation_config: dict[str, object] = base_config | {
        "question_count": payload.question_count,
        "question_types": list(payload.question_types),
        "difficulty": payload.difficulty,
        "expected_response_time_ms": payload.expected_response_time_ms,
        "bloom_distribution": dict(payload.bloom_distribution),
        "include_prerequisites": payload.include_prerequisites,
        "model_preference": payload.model_preference,
        "source_lesson_ids": [str(x) for x in payload.source_lesson_ids],
        "generation_mode": payload.generation_mode,
        "focus_topics": list(payload.focus_topics),
        "avoid_topics": list(payload.avoid_topics),
        "extra_instructions": payload.extra_instructions,
        "append": payload.append,
        "coverage_options": coverage_dump,
        "target_outcome_ids": [str(x) for x in payload.target_outcome_ids],
    }
    run = GenerationRun(
        generation_type="quiz",
        source_scope_kind="module",
        course_id=course_id,
        module_id=module_id,
        requested_by=actor.user_id,
        status="pending",
        config_json=generation_config,
    )
    db.add(run)
    await flush_or_conflict(db)

    if quiz is None:
        if not payload.title:
            raise AppError("title is required when creating a new quiz")
        quiz = Quiz(
            course_id=course_id,
            module_id=module_id,
            title=payload.title,
            description=payload.description,
            generation_run_id=run.id,
            created_by=actor.user_id,
        )
        db.add(quiz)
        await flush_or_conflict(db)
    else:
        quiz.generation_run_id = run.id
    run.config_json = dict(run.config_json) | {"quiz_id": str(quiz.id)}
    await _add_quiz_source_lessons(db, quiz.id, list(payload.source_lesson_ids))
    await db.commit()
    await db.refresh(run)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            "run_quiz_generation_task", actor.user_id, run.id
        )
    return run


async def regenerate_question(
    db: AsyncSession,
    question_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None,
) -> GenerationRun:
    """Create a per-question regeneration run + enqueue the worker task.

    The dispatcher uses ``run.config_json["question_id"]`` to route to
    the regenerate pipeline; no other config is required for the
    legacy parity port. Task name matches the registered worker
    function (``run_quiz_generation_task``) per ARQ convention.
    """
    from abridgeai.features.quizzes.services.authoring import (  # noqa: PLC0415
        _require_question,
        _require_quiz,
    )

    question = await _require_question(db, question_id)
    quiz = await _require_quiz(db, question.quiz_id)
    run = GenerationRun(
        generation_type="quiz",
        source_scope_kind="module",
        course_id=quiz.course_id,
        module_id=quiz.module_id,
        requested_by=actor.user_id,
        status="pending",
        config_json={"question_id": str(question_id), "quiz_id": str(quiz.id)},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            "run_quiz_generation_task", actor.user_id, run.id
        )
    return run


__all__ = [
    "regenerate_question",
    "start_generation_run",
]
