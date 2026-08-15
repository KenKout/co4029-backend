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

from abridgeai.features.quizzes.services.generation_runs import (  # noqa: F401
    regenerate_question,
    start_generation_run,
)
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
    # ``browser_security`` is a string enum column ('none' | 'securewindow')
    # guarded by a CHECK constraint, but the client models it as a boolean
    # toggle and the loose-dict PATCH body delivers a bool. Map the toggle to
    # the enum here so a raw ``false``/``true`` can't reach the column and trip
    # ``ck_quizzes_browser_security`` (→ 500). Already-valid strings pass
    # through; anything else is a 400 rather than a DB error.
    if key == "browser_security":
        if isinstance(value, bool):
            return "securewindow" if value else "none"
        if value is None:
            return "none"
        if isinstance(value, str) and value in {"none", "securewindow"}:
            return value
        raise AppError("browser_security must be one of: none, securewindow")
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


# Settings that stay editable on a PUBLISHED quiz because changing them can't
# corrupt or interrupt a student who is taking the quiz (or who already
# finished): pure display copy, notification behaviour, and the scheduling
# window (extend a deadline, open later). Everything NOT in this set —
# scoring, timing, attempt limits, shuffle, mastery/SM-2 params, proctoring,
# and the post-attempt review matrix — is frozen once published, since it
# would change grading/presentation under live or completed attempts. A
# whitelist (not a blacklist) is deliberate: any column added later is frozen
# by default until someone explicitly vets it as student-safe.
_PUBLISHED_EDITABLE_FIELDS = frozenset(
    {
        "title",
        "description",
        "available_from",
        "available_until",
        "due_at",
        "reminders_enabled",
    }
)


def _as_plain_json(value: Any) -> Any:  # noqa: ANN401  -- mirrors arbitrary JSON payloads
    """Coerce a payload value into plain JSON-serializable data.

    Values destined for JSONB columns must be dicts/lists/scalars. Two callers
    can hand us something richer:

    * the authoring router's private ``_AttrShim``, which wraps nested dicts
      (and lists of dicts) for attribute access — psycopg cannot serialize it;
    * a real Pydantic model, once the DTO surface replaces the shim.

    Both expose ``model_dump()``, so we prefer that and recurse through
    containers. Anything already plain passes through untouched.
    """
    if isinstance(value, list):
        return [_as_plain_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _as_plain_json(item) for key, item in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _as_plain_json(dump())
    return value


def _assert_quiz_editable(quiz: Quiz) -> None:
    """Reject ALL authoring edits on a published quiz (content paths).

    Used by the question CRUD + bulk-approve paths: a published quiz's
    questions/options are fully frozen, since students can already see and
    attempt them and editing would corrupt grading. Raised as
    :class:`ConflictError` so the router maps to HTTP 409. Quiz-settings
    edits use the field-aware :func:`_assert_quiz_settings_editable` instead.
    """
    if quiz.status == "published":
        raise ConflictError(
            "quiz_published_readonly: a published quiz's questions cannot be "
            "edited; archive it first to make changes"
        )


def _assert_quiz_settings_editable(quiz: Quiz, changed_fields: set[str]) -> None:
    """Field-aware freeze for quiz-settings PATCH on a published quiz.

    Student-safe settings (see :data:`_PUBLISHED_EDITABLE_FIELDS`) stay
    editable so teachers can still rename, tweak reminders, or extend the
    schedule window on a live quiz. Touching any other setting — anything
    that would change scoring/timing/attempts/presentation under a student
    who is taking or has finished the quiz — is rejected with HTTP 409.
    Draft quizzes are unrestricted.
    """
    if quiz.status != "published":
        return
    frozen = changed_fields - _PUBLISHED_EDITABLE_FIELDS
    if frozen:
        raise ConflictError(
            "quiz_published_setting_locked: these settings are frozen on a "
            "published quiz and would affect students mid-attempt or after "
            f"finishing: {', '.join(sorted(frozen))}. Archive the quiz first "
            "to change them."
        )


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


def _validate_fill_blank_options(options: list[Any]) -> None:
    """fill_blank word bank: every entry needs text. The correct/distractor
    split is the caller's job (the AI path validates it more strictly in
    ``shape_validators.validate_fill_blank``)."""
    if any(not str(option.option_text).strip() for option in options):
        raise AppError("Question option text is required")


def _validate_question_options(
    question_type: str, options: list[Any], *, single_answer: bool = True
) -> None:
    """Type-aware option validation for the manual authoring path.

    Mirrors the per-type shape rules enforced by the AI generation
    parser (``stages/generation/parsers.GeneratedQuestion``):

    * ``multiple_choice`` — 2..10 options; single_answer → exactly 1 correct,
      else ≥1 correct (Phase 7 multi-select).
    * ``true_false`` — exactly 2 options keyed T/F, exactly 1 correct.
    * ``short_answer`` / ``numerical`` — no options (the expected answer is
      carried on the question's own columns / payload).
    * ``fill_blank`` — ``options`` is the WORD BANK: the correct entries (one
      per distinct blank) plus any teacher-declared distractors.
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
    elif question_type == "fill_blank":
        _validate_fill_blank_options(options)
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
    # Field-aware freeze: on a published quiz only student-safe settings may
    # change (title/description/schedule/reminders); everything else is locked
    # so a student mid-attempt (or one who already finished) isn't disrupted.
    changed_fields = set(payload.model_dump(exclude_unset=True).keys())
    _assert_quiz_settings_editable(quiz, changed_fields)
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
    """Soft-delete the quiz + cascade to questions / options / revisions.

    Also soft-deletes the ``module_items`` row that points at this quiz.
    ``soft_delete_cascade`` walks ONETOMANY relationships only, and
    ``module_items -> quizzes`` is MANYTOONE from the item side (``Quiz`` has no
    ``items`` relationship), so the cascade cannot reach it. Left behind, the
    orphaned item kept appearing in the course content tree with a ``quiz_id``
    pointing at a deleted quiz — the UI rendered it, and clicking it 404'd.
    """
    quiz = await _require_quiz(db, quiz_id)

    # Collect this quiz's question ids BEFORE the cascade so we can purge their
    # SM-2 card state. student_card_state is keyed on question_id and, being
    # cross-feature, is invisible to soft_delete_cascade (it walks ONETOMANY
    # SoftDelete children only, and SR's state table is neither). Left behind,
    # those rows become perpetually-"due" cards no student can review.
    from sqlalchemy import select as _select  # noqa: PLC0415

    from abridgeai.features.spaced_repetition.api.public import (  # noqa: PLC0415
        purge_card_state_for_questions,
    )

    question_ids = list(
        (await db.execute(_select(QuizQuestion.id).where(QuizQuestion.quiz_id == quiz_id)))
        .scalars()
        .all()
    )

    await soft_delete_cascade(db, quiz, actor_id=actor.user_id)
    await purge_card_state_for_questions(db, question_ids)

    from sqlalchemy import update  # noqa: PLC0415

    from abridgeai.features.courses.models import ModuleItem  # noqa: PLC0415

    await db.execute(
        update(ModuleItem)
        .where(ModuleItem.quiz_id == quiz_id, ModuleItem.deleted_at.is_(None))
        .values(deleted_at=utcnow(), deleted_by=actor.user_id)
    )
    await db.flush()


async def create_question(
    db: AsyncSession,
    quiz_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO lands in T5.14.
    actor: CurrentUser,
) -> QuizQuestion:
    """Create a single question; MCQ flavours are validated up front."""
    quiz = await _require_quiz(db, quiz_id)
    _assert_quiz_editable(quiz)
    if not payload.prompt_text.strip():
        raise AppError("Question text is required")
    options_payload = list(payload.options or [])
    question_type = _normalize_question_type(payload.question_type)
    _single_answer = getattr(payload, "single_answer", None)
    _validate_question_options(
        question_type,
        options_payload,
        single_answer=True if _single_answer is None else bool(_single_answer),
    )

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
    # Phase 7: persist type-specific answer fields when supplied. These are
    # only meaningful for the expanded types; MCQ/T-F ignore them. Set only
    # when present so MCQ creation keeps the server defaults (single_answer=True).
    #
    # match_pairs / ordering_sequence land in JSONB columns, so they MUST be
    # plain JSON data. The router hands us a shim that turns nested lists of
    # dicts into attribute-access objects (handy for ``options``, fatal here —
    # psycopg cannot json.dumps them). ``_as_plain_json`` unwraps that back to
    # dicts/lists, and also handles real Pydantic models once DTOs land.
    _single = getattr(payload, "single_answer", None)
    if _single is not None:
        question.single_answer = bool(_single)
    _num_ans = getattr(payload, "numeric_answer", None)
    if _num_ans is not None:
        question.numeric_answer = _num_ans
    _num_tol = getattr(payload, "numeric_tolerance", None)
    if _num_tol is not None:
        question.numeric_tolerance = _num_tol
    _pairs = getattr(payload, "match_pairs", None)
    if _pairs is not None:
        question.match_pairs = _as_plain_json(_pairs)
    _distractors = getattr(payload, "match_distractors", None)
    if _distractors is not None:
        question.match_distractors = _as_plain_json(_distractors)
    _seq = getattr(payload, "ordering_sequence", None)
    if _seq is not None:
        question.ordering_sequence = _as_plain_json(_seq)
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
    quiz = await _require_quiz(db, question.quiz_id)
    _assert_quiz_editable(quiz)
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

    # Phase 3 SECURITY: sanitize rich text on the UPDATE path too. Without this
    # a teacher could PATCH raw <script>/onerror markup into prompt/hint/
    # explanation, which the client renders as HTML for the ``html`` format —
    # i.e. stored XSS against every student taking the quiz. The format used is
    # the one in this payload when supplied, else the value already on the row.
    from abridgeai.features.quizzes.services.sanitize import (  # noqa: PLC0415
        sanitize_rich_content,
    )

    for _text_field, _format_field in (
        ("prompt_text", "prompt_format"),
        ("hint_text", "hint_format"),
        ("explanation", "explanation_format"),
    ):
        if _text_field in field_updates and field_updates[_text_field] is not None:
            _fmt = field_updates.get(_format_field, getattr(question, _format_field, "plain"))
            field_updates[_text_field] = sanitize_rich_content(field_updates[_text_field], fmt=_fmt)

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
    _assert_quiz_editable(quiz)
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


async def _replace_fill_blank_options(
    db: AsyncSession,
    question: QuizQuestion,
    option_payloads: list[Any],
) -> None:
    """Replace a fill_blank word bank in full (delete + insert). Option ids
    aren't referenced anywhere (the grader reads
    ``original_generated_payload.correct_answer``; the student word bank reads
    ``option_text``), so this is safe and lets the teacher add/remove
    distractors freely."""
    from sqlalchemy import delete as sa_delete  # noqa: PLC0415

    await db.execute(
        sa_delete(QuizQuestionOption).where(
            QuizQuestionOption.question_id == question.id
        )
    )
    for position, option_payload in enumerate(option_payloads, start=1):
        db.add(
            QuizQuestionOption(
                question_id=question.id,
                option_key=str(
                    getattr(option_payload, "option_key", None)
                    or f"O{position:02d}"
                ).strip().upper(),
                option_text=str(getattr(option_payload, "option_text", "")).strip(),
                is_correct=bool(getattr(option_payload, "is_correct", False)),
                position=position,
            )
        )


async def _update_question_options(  # noqa: C901 -- MCQ/TF option-sync is inherently branchy (pre-existing)
    db: AsyncSession,
    question: QuizQuestion,
    option_payloads: list[Any],
) -> None:
    if question.question_type not in {"multiple_choice", "true_false", "fill_blank"}:
        raise AppError(
            "Only multiple_choice, true_false and fill_blank questions support option editing"
        )

    if question.question_type == "fill_blank":
        await _replace_fill_blank_options(db, question, option_payloads)
        return

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

    # Phase 7: mirror the create-path rules (``_validate_question_options``).
    # This used to hardcode "exactly four options / exactly one correct", which
    # rejected both multi-select MCQ (single_answer=False → >=1 correct) and the
    # relaxed 2..10 option count. true_false stays strictly 2 options / 1 correct.
    n_correct = sum(1 for option in options if option.is_correct)
    if question.question_type == "true_false":
        if len(options) != 2:
            raise AppError("true_false questions must have exactly two options")
        if n_correct != 1:
            raise AppError("true_false questions must have exactly one correct option")
        return
    if len(options) < 2 or len(options) > 10:
        raise AppError("multiple_choice questions need between 2 and 10 options")
    if question.single_answer:
        if n_correct != 1:
            raise AppError("single-answer multiple_choice must have exactly one correct option")
    elif n_correct < 1:
        raise AppError("multi-answer multiple_choice must have at least one correct option")


async def delete_question(db: AsyncSession, question_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete a question + repack sibling positions to stay 1..N."""
    question = await _require_question(db, question_id)
    quiz_id = question.quiz_id
    quiz = await _require_quiz(db, quiz_id)
    _assert_quiz_editable(quiz)

    # Serialize concurrent deletes on the SAME quiz.
    #
    # The repack below UPDATEs every surviving sibling. Two deletes running
    # concurrently against one quiz each grab row locks on the same sibling set
    # and interleave, so each ends up waiting on a row the other already holds:
    #
    #   DeadlockDetected: Process A waits for ShareLock on transaction of B;
    #   B waits for ShareLock on transaction of A
    #   CONTEXT: while locking tuple (...) in relation "quiz_questions"
    #
    # The client fires bulk deletes concurrently (Promise.allSettled), so this
    # is a normal path, not an edge case. A transaction-scoped advisory lock
    # keyed on the quiz makes same-quiz deletes queue instead of deadlocking,
    # while deletes on DIFFERENT quizzes still run fully in parallel. It is
    # released automatically at COMMIT/ROLLBACK — no unlock bookkeeping.
    from sqlalchemy import text  # noqa: PLC0415

    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"quiz_questions_repack:{quiz_id}"},
    )

    await soft_delete_cascade(db, question, actor_id=actor.user_id)

    # Purge the question's SM-2 card state (cross-feature; the cascade cannot
    # reach it). Otherwise the deleted question's cards stay perpetually "due".
    from abridgeai.features.spaced_repetition.api.public import (  # noqa: PLC0415
        purge_card_state_for_questions,
    )

    await purge_card_state_for_questions(db, [question_id])
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
