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

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
    QuizSourceLesson,
)
from abridgeai.features.quizzes.queries import authoring as authoring_queries
from abridgeai.features.quizzes.schemas import QuizGenerationRequest
from abridgeai.features.quizzes.services.publish_gate import (
    QuizPublishValidationError,
    assert_all_questions_approved,
    assert_t_exp_set_for_all_questions,
    bulk_set_expected_response_time,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_DEFAULT_PASSING_SCORE = Decimal("70.00")

# Two-phase position-repack offset (mirrors courses.reorder_module_items):
# bump surviving rows into a disjoint high range before assigning final
# 1..N so the non-deferrable (quiz_id, position) unique constraint never
# collides mid-flush. A quiz never has 100k questions.
_POSITION_REPACK_OFFSET = 100_000


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


# Quiz columns typed DateTime(timezone=True) that a loose dict PATCH body may
# deliver as an ISO-8601 string. setattr'ing a str onto a DateTime column would
# persist wrong / raise at flush, so coerce these keys str -> datetime here.
_DATETIME_PATCH_KEYS = frozenset({"available_from", "available_until", "due_at"})


def _coerce_patch_value(key: str, value: object) -> object:
    """Coerce known datetime-typed PATCH keys from ISO strings to datetime.

    NULL (clear the window) and already-datetime values pass through
    untouched. A trailing 'Z' is normalised to '+00:00' for
    ``datetime.fromisoformat`` (Python < 3.11 compatibility, and harmless
    on newer). An unparseable string raises ``AppError`` → HTTP 400.
    """
    # Phase 2: validate the review-visibility matrix through its schema so an
    # invalid shape is rejected (→ 400) rather than written raw to the JSONB
    # column. Store the normalised (defaults-filled) dict.
    if key == "review_options" and value is not None:
        from abridgeai.features.quizzes.schemas.review_options import (  # noqa: PLC0415
            ReviewOptions,
        )

        try:
            return ReviewOptions.model_validate(value).model_dump()
        except Exception as exc:  # noqa: BLE001
            raise AppError("review_options is not a valid review-visibility matrix") from exc
    if key not in _DATETIME_PATCH_KEYS or value is None:
        return value
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AppError(f"{key} must be an ISO-8601 datetime or null") from exc
    raise AppError(f"{key} must be an ISO-8601 datetime or null")


def _apply_patch(model: object, payload: object) -> None:
    data = payload.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
    for key, value in data.items():
        setattr(model, key, _coerce_patch_value(key, value))


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


_LEGACY_TYPE_ALIASES: dict[str, str] = {
    "mcq": "multiple_choice",
    "fill_in_the_blank": "fill_blank",
    "true/false": "true_false",
    "tf": "true_false",
}


def _normalize_question_type(raw: Any) -> str:  # noqa: ANN401 -- arbitrary DTO field
    """Map legacy aliases onto the DB ``question_type`` vocabulary.

    Mirrors :func:`abridgeai.features.quizzes.ai.stages.generation.parsers._normalize_question_type`
    so the manual authoring path accepts the same legacy spellings the
    AI-generated path does (no client-visible behavior change).
    """
    if not isinstance(raw, str):
        return "multiple_choice"
    cleaned = raw.strip().lower()
    return _LEGACY_TYPE_ALIASES.get(cleaned, cleaned)


def _validate_question_options(
    question_type: str, options: list[Any], *, single_answer: bool = True
) -> None:
    """Type-aware option validation for the manual authoring path.

    Mirrors the per-type shape rules enforced by the AI generation
    parser (``stages/generation/parsers.GeneratedQuestion``):

    * ``multiple_choice`` — 2..10 options; single_answer → exactly 1 correct,
      else ≥1 correct (Phase 7 multi-select).
    * ``true_false`` — exactly 2 options keyed T/F, exactly 1 correct.
    * ``short_answer`` / ``fill_blank`` / ``numerical`` — no options (the
      expected answer is carried on the question's own columns / payload).
    """
    if question_type == "multiple_choice":
        if len(options) < 2 or len(options) > 10:
            raise AppError("multiple_choice questions need between 2 and 10 options")
        if any(not str(option.option_text).strip() for option in options):
            raise AppError("Question option text is required")
        n_correct = sum(1 for option in options if option.is_correct)
        if single_answer and n_correct != 1:
            raise AppError("single-answer multiple_choice must have exactly one correct option")
        if not single_answer and n_correct < 1:
            raise AppError("multi-answer multiple_choice must have at least one correct option")
    elif question_type == "true_false":
        if len(options) != 2:
            raise AppError("true_false questions must have exactly two options")
        keys = [str(option.option_key).strip().upper() for option in options]
        if set(keys) != {"T", "F"}:
            raise AppError("true_false option keys must be T, F")
        if sum(1 for option in options if option.is_correct) != 1:
            raise AppError("true_false questions must have exactly one correct option")
    elif options:
        raise AppError(f"{question_type} questions do not support options")


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


async def get_latest_generation_run(db: AsyncSession, quiz_id: UUID) -> GenerationRun | None:
    """Return the most recent ``GenerationRun`` for ``quiz_id`` (any status).

    Powers the SPA's quiz-generation panel reattach-on-mount: instead
    of stashing the active run id in the browser, the SPA asks the
    server for the latest run on every mount. That makes the run
    handle survive cross-device sessions, tab closes, and lets a
    second teacher viewing the same quiz see the in-flight run too.
    Returns ``None`` when the quiz has never been generated.
    """
    from datetime import timedelta  # noqa: PLC0415

    from sqlalchemy import and_, not_, select  # noqa: PLC0415

    # Ignore *stale failed* runs: a run that failed more than 6 hours ago is
    # not something the SPA should reattach to on mount — otherwise the panel
    # greets the teacher with a long-dead error (e.g. a transient upstream
    # blip from days ago) as if it just happened. Pending/running/completed
    # runs of any age still reattach, and recent (<6h) failures still surface
    # so a just-failed generation is visible.
    stale_failed_cutoff = utcnow() - timedelta(hours=6)
    stmt = (
        select(GenerationRun)
        .where(
            GenerationRun.config_json["quiz_id"].astext == str(quiz_id),
            not_(
                and_(
                    GenerationRun.status == "failed",
                    GenerationRun.finished_at.is_not(None),
                    GenerationRun.finished_at < stale_failed_cutoff,
                )
            ),
        )
        .order_by(GenerationRun.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


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
    await assert_all_questions_approved(db, quiz_id)
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
    question_type = _normalize_question_type(payload.question_type)
    _validate_question_options(question_type, options_payload)

    # Phase 3: sanitize rich content on write per each field's format
    # discriminator (plain passes through; markdown/html are nh3-cleaned).
    from abridgeai.features.quizzes.services.sanitize import (  # noqa: PLC0415
        sanitize_rich_content,
    )

    prompt_format = getattr(payload, "prompt_format", "plain")
    hint_format = getattr(payload, "hint_format", "plain")
    explanation_format = getattr(payload, "explanation_format", "plain")

    next_position = await _next_question_position(db, quiz_id)
    question = QuizQuestion(
        quiz_id=quiz_id,
        position=next_position,
        question_type=question_type,
        prompt_text=sanitize_rich_content(payload.prompt_text.strip(), fmt=prompt_format),
        hint_text=sanitize_rich_content(getattr(payload, "hint_text", None), fmt=hint_format),
        explanation=sanitize_rich_content(
            getattr(payload, "explanation", None), fmt=explanation_format
        ),
        prompt_format=prompt_format,
        hint_format=hint_format,
        explanation_format=explanation_format,
        difficulty=getattr(payload, "difficulty", None),
        bloom_level=getattr(payload, "bloom_level", None),
        review_status=getattr(payload, "review_status", "pending"),
        expected_response_time_ms=getattr(payload, "expected_response_time_ms", None),
        learning_outcome_id=getattr(payload, "learning_outcome_id", None),
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


async def bulk_approve_questions(
    db: AsyncSession,
    quiz_id: UUID,
    question_ids: list[UUID],
    actor: CurrentUser,
) -> int:
    """Set ``review_status='approved'`` on each ``question_id`` in ``quiz_id``.

    The teacher's bulk sign-off for AI-generated content. Questions that
    don't belong to ``quiz_id`` (or are soft-deleted → auto-filtered by
    db.get's SELECT filter) are skipped. Stamps ``reviewed_by`` /
    ``reviewed_at`` like the single-question approve path so the audit trail
    matches.
    """
    quiz = await db.get(Quiz, quiz_id)
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    if not question_ids:
        return 0
    updated = 0
    for question_id in question_ids:
        question = await db.get(QuizQuestion, question_id)
        if question is None or question.quiz_id != quiz_id:
            continue
        question.review_status = "approved"
        question.reviewed_by = actor.user_id
        question.reviewed_at = utcnow()
        updated += 1
    await db.flush()
    return updated


async def _update_question_options(
    db: AsyncSession,
    question: QuizQuestion,
    option_payloads: list[Any],
) -> None:
    if question.question_type not in {"multiple_choice", "true_false"}:
        raise AppError("Only multiple_choice and true_false questions support option editing")

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
    await db.flush()

    from sqlalchemy import select  # noqa: PLC0415

    # Repack surviving siblings to a dense 1..N ordering. Two caveats:
    #
    # 1. Exclude the just-tombstoned row (``deleted_at IS NULL``) — the
    #    partial unique index ignores soft-deleted rows, but if we pull the
    #    dead row into the renumber we'd assign a live position to it.
    # 2. ``uq_quiz_questions_position`` (quiz_id, position) is a
    #    non-deferrable unique constraint checked per-row, and the ORM flush
    #    order is not guaranteed to be collision-free (e.g. shifting 2->1
    #    while another row still holds 1 mid-batch). Use the same two-phase
    #    offset swap as ``courses.reorder_module_items``: bump everyone into
    #    a disjoint high range first, flush, then assign final 1..N.
    siblings_result = await db.execute(
        select(QuizQuestion)
        .where(
            QuizQuestion.quiz_id == quiz_id,
            QuizQuestion.deleted_at.is_(None),
        )
        .order_by(QuizQuestion.position)
    )
    siblings = list(siblings_result.scalars().all())

    needs_repack = any(
        sibling.position != new_position for new_position, sibling in enumerate(siblings, start=1)
    )
    if not needs_repack:
        return

    for idx, sibling in enumerate(siblings):
        sibling.position = _POSITION_REPACK_OFFSET + idx
    await db.flush()

    for new_position, sibling in enumerate(siblings, start=1):
        sibling.position = new_position
    await flush_or_conflict(db)


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
    coverage_dump: dict[str, Any] | None = (
        payload.coverage_options.model_dump() if payload.coverage_options is not None else None
    )
    # Structured FR-5 fields shadow ``config_json`` on conflict — the
    # schema layer documents this contract, so the merge order matters.
    generation_config: dict[str, Any] = base_config | {
        "question_count": payload.question_count,
        "question_types": list(payload.question_types),
        "difficulty": payload.difficulty,
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
