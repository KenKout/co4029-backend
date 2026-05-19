"""Teacher-side quiz authoring service (T5.13).

Ports the CRUD / revision / generation-trigger surface from
``backend/app/routes/quizzes/service.py`` (legacy 465 LOC god-file).
Composes :mod:`features.quizzes.queries.authoring` for reads and
applies business rules + ORM writes for quiz / question / option
CRUD, MCQ option validation, revision history, and ARQ enqueue.

Discipline matches T1.7 / T3.5 / T4.5: services flush, the router
commits — except :func:`start_generation_run` which commits inline
because the ARQ worker reads the new ``GenerationRun`` row and must
see the row before the job dequeues.

§A5 / §C5 invariants ported as-is:

* MCQ questions require exactly four options with keys ``A,B,C,D`` and
  exactly one ``is_correct`` flag set.
* Per-question revisions append a ``QuizQuestionRevision`` row keyed
  by ``(question_id, revision_no)``; ``revision_no`` is bumped via the
  ``MAX(revision_no) + 1`` query helper.
* ``start_generation_run`` enforces FR-10a "no two in-flight runs per
  quiz" by querying ``GenerationRun.config_json['quiz_id']``; raises
  :class:`ConflictError` so the router maps to HTTP 409.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.quizzes.models import (
    GenerationRun,
    Quiz,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
    QuizSourceLesson,
)
from abridgeai.features.quizzes.queries import authoring as authoring_queries
from abridgeai.features.quizzes.services.publish_gate import (
    QuizPublishValidationError,
    assert_t_exp_set_for_all_questions,
    bulk_set_expected_response_time,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_DEFAULT_PASSING_SCORE = Decimal("70.00")


register_conflict_mappings(
    {
        "quiz_questions_quiz_id_position_key": "quiz_question_position_taken: another question already occupies this position in the quiz",  # noqa: E501
        "uq_quiz_questions_position": "quiz_question_position_taken: another question already occupies this position in the quiz",  # noqa: E501
        "quiz_question_options_question_id_option_key_key": "quiz_option_key_taken: another option with this key already exists for this question",  # noqa: E501
        "uq_quiz_question_options_key": "quiz_option_key_taken: another option with this key already exists for this question",  # noqa: E501
        "quiz_question_options_question_id_position_key": "quiz_option_position_taken: another option already occupies this position",  # noqa: E501
        "uq_quiz_question_options_position": "quiz_option_position_taken: another option already occupies this position",  # noqa: E501
        "uq_quiz_question_revisions_number": "quiz_revision_number_taken: this revision number already exists for the question",  # noqa: E501
        "uq_quiz_attempts_number": "quiz_attempt_number_taken: this attempt number already exists for the student",  # noqa: E501
        "uq_quiz_attempt_answers_question": "quiz_attempt_answer_already_recorded: this question has already been answered for this attempt",  # noqa: E501
        "quiz_attempts_idempotency_key_key": "quiz_attempt_idempotency_key_replayed: this idempotency key has already been used",  # noqa: E501
    }
)


def _apply_patch(model: object, payload: object) -> None:
    data = payload.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
    for key, value in data.items():
        setattr(model, key, value)


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = await authoring_queries.get_quiz_for_authoring(db, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


async def _require_question(db: AsyncSession, question_id: UUID) -> QuizQuestion:
    question = await db.get(QuizQuestion, question_id)
    if question is None:
        raise NotFoundError(f"Question {question_id} not found")
    return question


async def _resolve_module_course(db: AsyncSession, module_id: UUID) -> UUID:
    """Look up ``modules.course_id`` via :mod:`courses.api.public`."""
    module = await courses_api.get_module_by_id(db, module_id)
    if module is None:
        raise NotFoundError(f"Module {module_id} not found")
    return module.course_id


async def _ensure_module_item(db: AsyncSession, *, module_id: UUID, quiz_id: UUID) -> None:
    from sqlalchemy import text  # noqa: PLC0415

    existing = (
        await db.execute(
            text(
                "SELECT 1 FROM module_items WHERE quiz_id = :quiz_id AND deleted_at IS NULL LIMIT 1"
            ),
            {"quiz_id": quiz_id},
        )
    ).first()
    if existing is not None:
        return
    next_pos = await courses_api.next_module_item_position(db, module_id)
    await db.execute(
        text(
            "INSERT INTO module_items (id, module_id, item_type, quiz_id, position, "
            "created_at, updated_at) VALUES "
            "(uuid_generate_v4(), :module_id, 'quiz', :quiz_id, :pos, NOW(), NOW())"
        ),
        {"module_id": module_id, "quiz_id": quiz_id, "pos": next_pos},
    )


def _validate_mcq_options(options: list[Any]) -> None:
    if len(options) != 4:
        raise AppError("MCQ questions must have exactly four options")
    keys = [str(option.option_key).strip().upper() for option in options]
    if set(keys) != {"A", "B", "C", "D"}:
        raise AppError("MCQ option keys must be A, B, C, D")
    if any(not str(option.option_text).strip() for option in options):
        raise AppError("Question option text is required")
    if sum(1 for option in options if option.is_correct) != 1:
        raise AppError("MCQ questions must have exactly one correct option")


async def _next_question_position(db: AsyncSession, quiz_id: UUID) -> int:
    from sqlalchemy import func, select  # noqa: PLC0415

    stmt = select(func.coalesce(func.max(QuizQuestion.position), 0) + 1).where(
        QuizQuestion.quiz_id == quiz_id
    )
    return int((await db.execute(stmt)).scalar_one())


async def _next_revision_no(db: AsyncSession, question_id: UUID) -> int:
    from sqlalchemy import func, select  # noqa: PLC0415

    stmt = select(func.coalesce(func.max(QuizQuestionRevision.revision_no), 0) + 1).where(
        QuizQuestionRevision.question_id == question_id
    )
    return int((await db.execute(stmt)).scalar_one())


async def _quiz_has_in_flight_run(db: AsyncSession, quiz_id: UUID) -> bool:
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(GenerationRun.id)
        .where(
            GenerationRun.config_json["quiz_id"].astext == str(quiz_id),
            GenerationRun.status.in_(("pending", "running")),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).first() is not None


async def _add_quiz_source_lessons(db: AsyncSession, quiz_id: UUID, lesson_ids: list[UUID]) -> None:
    if not lesson_ids:
        return
    from sqlalchemy import select  # noqa: PLC0415

    existing_rows = await db.execute(
        select(QuizSourceLesson.lesson_id).where(QuizSourceLesson.quiz_id == quiz_id)
    )
    existing = {lesson_id for (lesson_id,) in existing_rows.all()}
    for lesson_id in lesson_ids:
        if lesson_id not in existing:
            db.add(QuizSourceLesson(quiz_id=quiz_id, lesson_id=lesson_id))


async def create_quiz(
    db: AsyncSession,
    module_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14 (router slice).
    actor: CurrentUser,
) -> Quiz:
    """Create a new draft quiz under ``module_id`` and link a ``ModuleItem``."""
    course_id = await _resolve_module_course(db, module_id)
    data = payload.model_dump(exclude_unset=True)
    quiz = Quiz(
        course_id=course_id,
        module_id=module_id,
        title=data["title"],
        description=data.get("description"),
        time_limit_seconds=data.get("time_limit_seconds"),
        passing_score_percent=data.get("passing_score_percent") or _DEFAULT_PASSING_SCORE,
        allow_retakes=data.get("allow_retakes", True),
        max_attempts=data.get("max_attempts"),
        cooldown_hours=data.get("cooldown_hours"),
        shuffle_questions=data.get("shuffle_questions", False),
        shuffle_options=data.get("shuffle_options", False),
        show_hints=data.get("show_hints", True),
        initial_ef=data.get("initial_ef"),
        min_ef_for_unlock=data.get("min_ef_for_unlock"),
        coverage_threshold=data.get("coverage_threshold"),
        reminders_enabled=data.get("reminders_enabled", False),
        generation_instructions=data.get("generation_instructions"),
        created_by=actor.user_id,
    )
    db.add(quiz)
    await flush_or_conflict(db)
    await _ensure_module_item(db, module_id=module_id, quiz_id=quiz.id)
    await flush_or_conflict(db)
    await db.refresh(quiz)
    return quiz


async def update_quiz(
    db: AsyncSession,
    quiz_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14.
    actor: CurrentUser,
) -> Quiz:
    del actor
    quiz = await _require_quiz(db, quiz_id)
    _apply_patch(quiz, payload)
    await flush_or_conflict(db)
    await db.refresh(quiz)
    return quiz


async def publish_quiz(db: AsyncSession, quiz_id: UUID, actor: CurrentUser) -> Quiz:
    del actor
    quiz = await _require_quiz(db, quiz_id)
    if quiz.status == "archived":
        raise AppError(f"Cannot publish archived quiz {quiz_id}")
    await assert_t_exp_set_for_all_questions(db, quiz_id)
    quiz.status = "published"
    quiz.published_at = utcnow()
    await _ensure_module_item(db, module_id=quiz.module_id, quiz_id=quiz.id)
    await flush_or_conflict(db)
    await db.refresh(quiz)
    return quiz


async def archive_quiz(db: AsyncSession, quiz_id: UUID, actor: CurrentUser) -> Quiz:
    del actor
    quiz = await _require_quiz(db, quiz_id)
    quiz.status = "archived"
    await flush_or_conflict(db)
    await db.refresh(quiz)
    return quiz


async def delete_quiz(db: AsyncSession, quiz_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete the quiz + cascade to questions / options / revisions."""
    quiz = await _require_quiz(db, quiz_id)
    await soft_delete_cascade(db, quiz, actor_id=actor.user_id)


async def create_question(
    db: AsyncSession,
    quiz_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14.
    actor: CurrentUser,
) -> QuizQuestion:
    """Create a single question; MCQ flavours are validated up front."""
    await _require_quiz(db, quiz_id)
    if not payload.prompt_text.strip():
        raise AppError("Question text is required")
    options_payload = list(payload.options or [])
    question_type = payload.question_type
    if question_type in {"mcq", "multiple_choice"}:
        _validate_mcq_options(options_payload)
        question_type = "multiple_choice"
    elif options_payload:
        raise AppError("Only MCQ questions support options")

    next_position = await _next_question_position(db, quiz_id)
    question = QuizQuestion(
        quiz_id=quiz_id,
        position=next_position,
        question_type=question_type,
        prompt_text=payload.prompt_text.strip(),
        hint_text=getattr(payload, "hint_text", None),
        explanation=getattr(payload, "explanation", None),
        difficulty=getattr(payload, "difficulty", None),
        bloom_level=getattr(payload, "bloom_level", None),
        review_status=getattr(payload, "review_status", "pending"),
        expected_response_time_ms=getattr(payload, "expected_response_time_ms", None),
        source_refs=getattr(payload, "source_refs", []) or [],
        original_generated_payload=None,
        reviewed_by=(
            actor.user_id if getattr(payload, "review_status", None) == "approved" else None
        ),
        reviewed_at=utcnow() if getattr(payload, "review_status", None) == "approved" else None,
    )
    db.add(question)
    await flush_or_conflict(db)

    for position, option_payload in enumerate(options_payload, start=1):
        db.add(
            QuizQuestionOption(
                question_id=question.id,
                option_key=str(option_payload.option_key).strip().upper(),
                option_text=str(option_payload.option_text).strip(),
                is_correct=bool(option_payload.is_correct),
                position=position,
            )
        )
    db.add(
        QuizQuestionRevision(
            question_id=question.id,
            revision_no=1,
            source_kind="teacher",
            payload_json=payload.model_dump(mode="json"),
            created_by=actor.user_id,
        )
    )
    await flush_or_conflict(db)
    await db.refresh(question)
    return question


async def update_question(
    db: AsyncSession,
    question_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14.
    actor: CurrentUser,
) -> QuizQuestion:
    """Patch fields + append a revision; option edits are MCQ-only."""
    question = await _require_question(db, question_id)
    revision_no = await _next_revision_no(db, question_id)
    payload_json = payload.model_dump(exclude_unset=True, mode="json")
    db.add(
        QuizQuestionRevision(
            question_id=question_id,
            revision_no=revision_no,
            source_kind="teacher",
            payload_json=payload_json,
            created_by=actor.user_id,
        )
    )
    field_updates = payload.model_dump(exclude_unset=True, exclude={"options"})
    for key, value in field_updates.items():
        setattr(question, key, value)

    options_payload = getattr(payload, "options", None)
    if options_payload is not None:
        await _update_question_options(db, question, list(options_payload))

    question.reviewed_by = actor.user_id
    question.reviewed_at = utcnow()
    await flush_or_conflict(db)
    await db.refresh(question)
    return question


async def _update_question_options(
    db: AsyncSession,
    question: QuizQuestion,
    option_payloads: list[Any],
) -> None:
    if question.question_type not in {"mcq", "multiple_choice"}:
        raise AppError("Only MCQ questions support option editing")

    from sqlalchemy import select  # noqa: PLC0415

    result = await db.execute(
        select(QuizQuestionOption)
        .where(QuizQuestionOption.question_id == question.id)
        .order_by(QuizQuestionOption.position)
    )
    options = list(result.scalars().all())
    options_by_id = {option.id: option for option in options}
    options_by_key = {option.option_key: option for option in options}

    for option_payload in option_payloads:
        option = None
        payload_id = getattr(option_payload, "id", None)
        if payload_id is not None:
            option = options_by_id.get(payload_id)
        if option is None and getattr(option_payload, "option_key", None) is not None:
            option = options_by_key.get(option_payload.option_key)
        if option is None:
            raise AppError("Question option not found")
        new_text = getattr(option_payload, "option_text", None)
        if new_text is not None:
            stripped = str(new_text).strip()
            if not stripped:
                raise AppError("Question option text is required")
            option.option_text = stripped
        new_is_correct = getattr(option_payload, "is_correct", None)
        if new_is_correct is not None:
            option.is_correct = bool(new_is_correct)

    if len(options) != 4:
        raise AppError("MCQ questions must have exactly four options")
    if sum(1 for option in options if option.is_correct) != 1:
        raise AppError("MCQ questions must have exactly one correct option")


async def delete_question(db: AsyncSession, question_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete a question + repack sibling positions to stay 1..N."""
    question = await _require_question(db, question_id)
    quiz_id = question.quiz_id
    await soft_delete_cascade(db, question, actor_id=actor.user_id)

    from sqlalchemy import select  # noqa: PLC0415

    siblings_result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.quiz_id == quiz_id).order_by(QuizQuestion.position)
    )
    siblings = list(siblings_result.scalars().all())
    for new_position, sibling in enumerate(siblings, start=1):
        if sibling.position != new_position:
            sibling.position = new_position
    await flush_or_conflict(db)


async def start_generation_run(
    db: AsyncSession,
    module_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14.
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
    """
    course_id = await _resolve_module_course(db, module_id)

    quiz: Quiz | None = None
    quiz_id_in = getattr(payload, "quiz_id", None)
    if quiz_id_in is not None:
        quiz = await _require_quiz(db, quiz_id_in)
        if quiz.module_id != module_id:
            raise AppError("Quiz must belong to this module")
        if await _quiz_has_in_flight_run(db, quiz.id):
            raise ConflictError("quiz_generation_in_progress")
        if not getattr(payload, "append", False):
            from sqlalchemy import delete as sa_delete  # noqa: PLC0415

            await db.execute(sa_delete(QuizQuestion).where(QuizQuestion.quiz_id == quiz.id))
            await flush_or_conflict(db)

    base_config = dict(getattr(payload, "config_json", None) or {})
    coverage_options = getattr(payload, "coverage_options", None)
    coverage_dump: Any = None
    if coverage_options is not None:
        coverage_dump = (
            coverage_options.model_dump()
            if hasattr(coverage_options, "model_dump")
            else coverage_options
        )
    generation_config = base_config | {
        "question_count": getattr(payload, "question_count", None),
        "question_types": getattr(payload, "question_types", None),
        "difficulty": getattr(payload, "difficulty", None),
        "bloom_distribution": getattr(payload, "bloom_distribution", None),
        "include_prerequisites": getattr(payload, "include_prerequisites", None),
        "model_preference": getattr(payload, "model_preference", None),
        "source_lesson_ids": [str(x) for x in getattr(payload, "source_lesson_ids", []) or []],
        "generation_mode": getattr(payload, "generation_mode", None),
        "focus_topics": getattr(payload, "focus_topics", None),
        "avoid_topics": getattr(payload, "avoid_topics", None),
        "extra_instructions": getattr(payload, "extra_instructions", None),
        "append": getattr(payload, "append", False),
        "coverage_options": coverage_dump,
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
        quiz = Quiz(
            course_id=course_id,
            module_id=module_id,
            title=getattr(payload, "title", None) or "Generated quiz",
            description=getattr(payload, "description", None),
            generation_run_id=run.id,
            created_by=actor.user_id,
        )
        db.add(quiz)
        await flush_or_conflict(db)
    else:
        quiz.generation_run_id = run.id
    run.config_json = dict(run.config_json) | {"quiz_id": str(quiz.id)}
    lesson_ids = list(getattr(payload, "source_lesson_ids", []) or [])
    await _add_quiz_source_lessons(db, quiz.id, lesson_ids)
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
    "QuizPublishValidationError",
    "archive_quiz",
    "bulk_set_expected_response_time",
    "create_question",
    "create_quiz",
    "delete_question",
    "delete_quiz",
    "publish_quiz",
    "regenerate_question",
    "start_generation_run",
    "update_question",
    "update_quiz",
]
