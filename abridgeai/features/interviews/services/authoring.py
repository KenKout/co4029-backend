"""Interview authoring service (T6.11).

Ports the teacher CRUD + generation-trigger surface from
``backend/app/routes/interviews/service.py``. Composes
:mod:`features.interviews.queries.authoring` for reads and applies
business rules + ORM writes for config / outcome / question CRUD,
manual edits, and ARQ enqueue.

Discipline matches T5.13 (quiz authoring): the service flushes, the
router commits — except :func:`start_generation_run` and
:func:`regenerate_question`, which commit inline because the ARQ
worker reads the new ``GenerationRun`` row out of band and must see
the row before the job dequeues.

Locked task names (mirrored by T6.13 worker registration):

* ``run_interview_generation_task`` — fan-out for full + per-question
  regeneration runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.quizzes.api import public as quizzes_public
from abridgeai.features.quizzes.api.public import GenerationRunDTO

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_RUN_INTERVIEW_GENERATION_TASK = "run_interview_generation_task"


def _apply_patch(model: object, payload: object) -> None:
    data = payload.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
    for key, value in data.items():
        setattr(model, key, value)


async def _require_config(db: AsyncSession, config_id: UUID) -> InterviewConfig:
    config = await authoring_queries.get_interview_for_authoring(db, config_id)
    if config is None:
        raise NotFoundError(f"Interview config {config_id} not found")
    return config


register_conflict_mappings(
    {
        "interview_outcomes_interview_config_id_position_key": "interview_outcome_position_taken: another outcome already occupies this position in the config",  # noqa: E501
        "uq_interview_outcomes_position": "interview_outcome_position_taken: another outcome already occupies this position in the config",  # noqa: E501
        "interview_questions_interview_config_id_position_key": "interview_question_position_taken: another question already occupies this position in the config",  # noqa: E501
        "uq_interview_questions_position": "interview_question_position_taken: another question already occupies this position in the config",  # noqa: E501
        "uq_interview_outcome_evaluations": "interview_outcome_evaluation_already_recorded: this outcome has already been evaluated for this session",  # noqa: E501
        "uq_interview_session_questions_sequence": "interview_session_question_sequence_taken: this answer was already submitted — please refresh",  # noqa: E501
    }
)


async def _require_question(
    db: AsyncSession, config_id: UUID, question_id: UUID
) -> InterviewQuestion:
    question = await db.get(InterviewQuestion, question_id)
    if question is None or question.interview_config_id != config_id:
        raise NotFoundError(f"Interview question {question_id} not found")
    return question


async def _require_outcome(db: AsyncSession, config_id: UUID, outcome_id: UUID) -> InterviewOutcome:
    outcome = await db.get(InterviewOutcome, outcome_id)
    if outcome is None or outcome.interview_config_id != config_id:
        raise NotFoundError(f"Interview outcome {outcome_id} not found")
    return outcome


async def _ensure_module_item(
    db: AsyncSession, *, module_id: UUID, interview_config_id: UUID
) -> None:
    """Insert a ``module_items`` row pointing at the published config.

    Existence check stays raw because ``courses.api.public`` does not
    surface a "find module_items by interview_config_id" reader (the
    available ``find_module_items`` is keyed by ``lesson_id``).  Next-
    position resolution and the INSERT itself bracket that raw SELECT:
    the position fetch goes through ``courses.api.public`` and the
    INSERT remains raw because it is a cross-feature WRITE that owns
    its own UoW boundary (mirrors T5.13 quiz authoring).
    """
    from sqlalchemy import text  # noqa: PLC0415

    existing = (
        await db.execute(
            text(
                "SELECT 1 FROM module_items "
                "WHERE interview_config_id = :id AND deleted_at IS NULL LIMIT 1"
            ),
            {"id": interview_config_id},
        )
    ).first()
    if existing is not None:
        return
    next_pos = await courses_public.next_module_item_position(db, module_id)
    await db.execute(
        text(
            "INSERT INTO module_items (id, module_id, item_type, "
            "interview_config_id, position, created_at, updated_at) VALUES "
            "(uuid_generate_v4(), :module_id, 'interview', :cfg, :pos, NOW(), NOW())"
        ),
        {"module_id": module_id, "cfg": interview_config_id, "pos": next_pos},
    )


async def create_interview_config(
    db: AsyncSession,
    course_id: UUID,
    payload: Any,  # noqa: ANN401  -- DTO carried in by router (T6.12).
    actor: CurrentUser,
) -> InterviewConfig:
    """Create a new ``draft`` interview config under ``course_id``."""
    data = payload.model_dump(exclude_unset=True)
    module_id = data.get("module_id")
    if module_id is None:
        raise AppError("module_id is required to create an interview config")
    config = InterviewConfig(
        course_id=course_id,
        module_id=module_id,
        title=data["title"],
        persona=data.get("persona"),
        supported_modes=data.get("supported_modes", "hybrid"),
        time_limit_minutes=data.get("time_limit_minutes"),
        max_attempts=data.get("max_attempts"),
        cooldown_hours=data.get("cooldown_hours"),
        min_outcomes_to_pass=data.get("min_outcomes_to_pass"),
        lock_quiz_ef_until_pass=bool(data.get("lock_quiz_ef_until_pass", False)),
        supplementary_instructions=data.get("supplementary_instructions"),
        security_response_policy=data.get("security_response_policy", "warn_and_continue"),
        security_max_consecutive_attempts=data.get("security_max_consecutive_attempts", 3),
        security_custom_refusal_en=data.get("security_custom_refusal_en"),
        security_custom_refusal_vi=data.get("security_custom_refusal_vi"),
        security_incident_summary_enabled=bool(
            data.get("security_incident_summary_enabled", True)
        ),
        created_by=actor.user_id,
    )
    db.add(config)
    await flush_or_conflict(db)
    # Surface the draft on the course content tree immediately. The
    # ``/content`` reader renders one row per ``module_items`` entry, so without
    # this insert a freshly created draft is invisible until publish. The
    # frontend already styles non-published items with an amber status badge.
    await _ensure_module_item(db, module_id=config.module_id, interview_config_id=config.id)
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def update_interview_config(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewConfig:
    del actor
    config = await _require_config(db, config_id)
    _apply_patch(config, payload)
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def publish_interview_config(
    db: AsyncSession, config_id: UUID, actor: CurrentUser
) -> InterviewConfig:
    del actor
    config = await _require_config(db, config_id)
    if config.status == "archived":
        raise AppError(f"Cannot publish archived interview config {config_id}")
    approved = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
    if not approved:
        raise AppError(
            "interview_no_approved_questions: at least one approved question is required to publish"
        )
    # Thesis §4.3: the pass/fail verdict is judged per learning outcome. A
    # config with zero outcomes can never produce a pass — block publishing it.
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    if not outcomes:
        raise AppError(
            "interview_no_outcomes: at least one learning outcome is required to publish"
        )
    config.status = "published"
    config.published_at = utcnow()
    await _ensure_module_item(db, module_id=config.module_id, interview_config_id=config.id)
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def archive_interview_config(
    db: AsyncSession, config_id: UUID, actor: CurrentUser
) -> InterviewConfig:
    del actor
    config = await _require_config(db, config_id)
    config.status = "archived"
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def unarchive_interview_config(
    db: AsyncSession, config_id: UUID, actor: CurrentUser
) -> InterviewConfig:
    del actor
    config = await _require_config(db, config_id)
    if config.status != "archived":
        raise AppError(
            f"Cannot unarchive interview config {config_id} — "
            f"status is '{config.status}', expected 'archived'"
        )
    config.status = "draft"
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def unpublish_interview_config(
    db: AsyncSession, config_id: UUID, actor: CurrentUser
) -> InterviewConfig:
    del actor
    config = await _require_config(db, config_id)
    if config.status != "published":
        raise AppError(
            f"Cannot unpublish interview config {config_id} — "
            f"status is '{config.status}', expected 'published'"
        )
    config.status = "draft"
    config.published_at = None
    await flush_or_conflict(db)
    await db.refresh(config)
    return config


async def delete_interview_config(db: AsyncSession, config_id: UUID, actor: CurrentUser) -> None:
    """Soft-delete the config + cascade to outcomes / questions."""
    config = await _require_config(db, config_id)
    await soft_delete_cascade(db, config, actor_id=actor.user_id)


async def add_question(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewQuestion:
    await _require_config(db, config_id)
    next_position = await authoring_queries.next_question_position(db, config_id)
    data = payload.model_dump(exclude_unset=True)
    if not str(data.get("prompt_text", "")).strip():
        raise AppError("Question prompt is required")
    question = InterviewQuestion(
        interview_config_id=config_id,
        linked_outcome_id=data.get("linked_outcome_id"),
        position=data.get("position") or next_position,
        question_type=data["question_type"],
        prompt_text=data["prompt_text"].strip(),
        difficulty=data.get("difficulty"),
        model_answer=(data.get("model_answer") or "").strip() or None,
        review_status="approved",
        ai_generated=False,
        source_refs_json=data.get("source_refs_json", []) or [],
        reviewed_by=actor.user_id,
        reviewed_at=utcnow(),
        created_by=actor.user_id,
    )
    db.add(question)
    await flush_or_conflict(db)
    await db.refresh(question)
    return question


async def update_question(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewQuestion:
    question = await _require_question(db, config_id, question_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(question, key, value)
    question.reviewed_by = actor.user_id
    question.reviewed_at = utcnow()
    await flush_or_conflict(db)
    await db.refresh(question)
    return question


async def delete_question(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    actor: CurrentUser,
) -> None:
    question = await _require_question(db, config_id, question_id)
    await soft_delete_cascade(db, question, actor_id=actor.user_id)


async def add_outcome(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewOutcome:
    await _require_config(db, config_id)
    data = payload.model_dump(exclude_unset=True)
    next_position = await authoring_queries.next_outcome_position(db, config_id)
    outcome = InterviewOutcome(
        interview_config_id=config_id,
        position=data.get("position") or next_position,
        outcome_text=data["outcome_text"],
        outcome_type=data["outcome_type"],
        importance_weight=data.get("importance_weight", 1),
        created_by=actor.user_id,
    )
    db.add(outcome)
    await flush_or_conflict(db)
    await db.refresh(outcome)
    return outcome


async def update_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewOutcome:
    del actor
    outcome = await _require_outcome(db, config_id, outcome_id)
    _apply_patch(outcome, payload)
    await flush_or_conflict(db)
    await db.refresh(outcome)
    return outcome


async def delete_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
    actor: CurrentUser,
) -> None:
    outcome = await _require_outcome(db, config_id, outcome_id)
    await soft_delete_cascade(db, outcome, actor_id=actor.user_id)


async def start_generation_run(
    db: AsyncSession,
    config_id: UUID,
    request: Any,  # noqa: ANN401  -- InterviewGenerationRequest DTO at router edge.
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> GenerationRunDTO:
    """Create a generation run + enqueue ``run_interview_generation_task``.

    Commits inline so the worker sees the row before the ARQ job
    dequeues. Mirrors T5.13 quiz ``start_generation_run`` semantics.
    """
    config = await _require_config(db, config_id)
    request_data = request.model_dump(exclude_unset=True, mode="json") if request else {}
    config_json: dict[str, Any] = dict(request_data) | {
        "interview_config_id": str(config.id),
    }
    run = await quizzes_public.create_generation_run(
        db,
        kind="interview",
        source_scope_kind="module",
        course_id=config.course_id,
        module_id=config.module_id,
        requested_by=actor.user_id,
        config_json=config_json,
    )
    config.generation_run_id = run.id
    await db.commit()

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _RUN_INTERVIEW_GENERATION_TASK, actor.user_id, run.id
        )
    return run


async def regenerate_question(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> GenerationRunDTO:
    """Per-question regeneration run + ARQ enqueue.

    Worker dispatcher reads ``config_json["question_id"]`` to route to
    the per-question regeneration path. Task name matches the canonical
    worker function (``run_interview_generation_task``).
    """
    question = await _require_question(db, config_id, question_id)
    config = await _require_config(db, config_id)
    run = await quizzes_public.create_generation_run(
        db,
        kind="interview",
        source_scope_kind="module",
        course_id=config.course_id,
        module_id=config.module_id,
        requested_by=actor.user_id,
        config_json={
            "interview_config_id": str(config.id),
            "question_id": str(question.id),
        },
    )
    await db.commit()

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _RUN_INTERVIEW_GENERATION_TASK, actor.user_id, run.id
        )
    return run


async def get_generation_run(
    db: AsyncSession, config_id: UUID, run_id: UUID
) -> GenerationRunDTO | None:
    """Look up a run by id; ensures it belongs to ``config_id``."""
    run = await quizzes_public.get_generation_run(db, run_id)
    if run is None:
        return None
    config_ref = (run.config_json or {}).get("interview_config_id")
    if config_ref != str(config_id):
        return None
    return run


__all__ = [
    "add_outcome",
    "add_question",
    "archive_interview_config",
    "create_interview_config",
    "delete_interview_config",
    "delete_outcome",
    "delete_question",
    "get_generation_run",
    "publish_interview_config",
    "regenerate_question",
    "start_generation_run",
    "unarchive_interview_config",
    "update_interview_config",
    "update_outcome",
    "update_question",
]
