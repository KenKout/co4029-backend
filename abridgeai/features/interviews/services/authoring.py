"""Interview authoring service (T6.11).

Ports the teacher CRUD + generation-trigger surface from
``backend/app/routes/interviews/service.py``. Composes
:mod:`features.interviews.queries.authoring` for reads and applies
business rules + ORM writes for config / outcome / question CRUD,
manual edits, and ARQ enqueue.

Discipline matches T5.13 (quiz authoring): the service flushes, the
router commits — except :func:`start_generation_run`, which commits
inline because the ARQ worker reads the new ``GenerationRun`` row out of
band and must see the row before the job dequeues.

Locked task names (mirrored by T6.13 worker registration):

* ``run_interview_generation_task`` — full interview generation runs
  (per-question regeneration was retired; see the router 410).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.core.slug import slugify, unique_slug
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
    max_logical_question_count,
    resolve_question_count,
    resolve_supplementary,
    resolve_type_mix,
    resolve_variant_strategy,
)
from abridgeai.features.interviews.dedup import (
    NOT_DUPLICATE,
    check_duplicate,
    store_question_embeddings,
)
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
    InterviewQuestionBankItem,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import published_freeze
from abridgeai.features.quizzes.api import public as quizzes_public
from abridgeai.features.quizzes.api.public import GenerationRunDTO

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_RUN_INTERVIEW_GENERATION_TASK = "run_interview_generation_task"

logger = logging.getLogger(__name__)


def _apply_patch(model: object, payload: object) -> None:
    data = payload.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
    # The API exposes per-trait persona overrides as the nested ``persona_profile``
    # object, but the ORM stores them in the ``persona_profile_json`` JSONB column.
    # Translate the key so a PATCH carrying persona_profile persists correctly;
    # an explicit null clears the overrides (back to the bare preset).
    if "persona_profile" in data:
        override = data.pop("persona_profile")
        model.persona_profile_json = override or None
    for key, value in data.items():
        setattr(model, key, value)


async def _require_config(db: AsyncSession, config_id: UUID) -> InterviewConfig:
    config = await authoring_queries.get_interview_for_authoring(db, config_id)
    if config is None:
        raise NotFoundError(f"Interview config {config_id} not found")
    return config


async def _require_module_in_course(db: AsyncSession, module_id: UUID, course_id: UUID) -> None:
    module = await courses_public.get_module_by_id(db, module_id)
    if module is None or module.course_id != course_id:
        raise AppError(f"Module {module_id} is not part of course {course_id}")


async def _require_lessons_in_course(
    db: AsyncSession, lesson_ids: list[UUID], course_id: UUID
) -> None:
    for lesson_id in lesson_ids:
        lesson = await courses_public.get_lesson_by_id(db, lesson_id)
        if lesson is None:
            raise AppError(f"Lesson {lesson_id} not found")
        module = await courses_public.get_module_by_id(db, lesson.module_id)
        if module is None or module.course_id != course_id:
            raise AppError(f"Lesson {lesson_id} is not part of course {course_id}")


register_conflict_mappings(
    {
        "interview_outcomes_interview_config_id_position_key": "interview_outcome_position_taken: another outcome already occupies this position in the config",  # noqa: E501
        "uq_interview_outcomes_position": "interview_outcome_position_taken: another outcome already occupies this position in the config",  # noqa: E501
        "interview_questions_interview_config_id_position_key": "interview_question_position_taken: another question already occupies this position in the config",  # noqa: E501
        "uq_interview_questions_position": "interview_question_position_taken: another question already occupies this position in the config",  # noqa: E501
        "uq_interview_outcome_evaluations": "interview_outcome_evaluation_already_recorded: this outcome has already been evaluated for this session",  # noqa: E501
        "uq_interview_session_questions_sequence": "interview_session_question_sequence_taken: this answer was already submitted — please refresh",  # noqa: E501
        "uq_generation_runs_active_dedup": "interview_generation_in_progress",
        # Partial unique index (migration 0092): a logical bank group holds each
        # angle at most once. Last line of defence behind the group advisory
        # lock — without the mapping a lost race surfaces as IntegrityError
        # (HTTP 500) instead of a clean 409.
        "uq_iq_bank_live_group_angle": "logical_question_angle_already_present",
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


def _assert_threshold_within_outcome_count(
    min_outcomes_to_pass: int | None,
    outcome_count: int,
) -> None:
    if min_outcomes_to_pass is not None and min_outcomes_to_pass > outcome_count:
        raise AppError(
            "min_outcomes_to_pass_exceeds_active_outcomes: minimum outcomes to pass "
            f"({min_outcomes_to_pass}) cannot exceed active learning outcomes ({outcome_count})"
        )


async def _assert_config_threshold_valid(db: AsyncSession, config: InterviewConfig) -> None:
    outcomes = await authoring_queries.list_outcomes_for_config(db, config.id)
    _assert_threshold_within_outcome_count(config.min_outcomes_to_pass, len(outcomes))


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
    module_id = UUID(str(module_id))
    await _require_module_in_course(db, module_id, course_id)
    # Auto-generate the URL slug from the title (unique per module over live
    # rows; collisions get -1, -2, … incrementing from 1).
    from sqlalchemy import select  # noqa: PLC0415

    base = slugify(data.get("slug") or data["title"]) or "interview"
    taken_rows = await db.execute(
        select(InterviewConfig.slug).where(
            InterviewConfig.module_id == module_id,
            InterviewConfig.deleted_at.is_(None),
        )
    )
    taken = {row[0] for row in taken_rows.all()}
    config = InterviewConfig(
        course_id=course_id,
        module_id=module_id,
        title=data["title"],
        slug=unique_slug(base, taken),
        persona=data.get("persona"),
        persona_profile_json=(data.get("persona_profile") or None),
        tts_voice=data.get("tts_voice"),
        time_limit_minutes=data.get("time_limit_minutes"),
        max_attempts=data.get("max_attempts"),
        cooldown_hours=data.get("cooldown_hours"),
        max_follow_ups_per_question=data.get("max_follow_ups_per_question", 2),
        max_hints_per_question=data.get("max_hints_per_question", 3),
        min_outcomes_to_pass=data.get("min_outcomes_to_pass"),
        supplementary_instructions=data.get("supplementary_instructions"),
        security_response_policy=data.get("security_response_policy", "warn_and_continue"),
        security_max_consecutive_attempts=data.get("security_max_consecutive_attempts", 3),
        security_custom_refusal_en=data.get("security_custom_refusal_en"),
        security_custom_refusal_vi=data.get("security_custom_refusal_vi"),
        security_incident_summary_enabled=bool(data.get("security_incident_summary_enabled", True)),
        created_by=actor.user_id,
    )
    db.add(config)
    await flush_or_conflict(db)
    # Seed the interview's rubric outcomes from the parent course's learning
    # outcomes so a freshly created interview starts with the course LOs in
    # place (no manual "import from course" needed). Course outcomes carry only
    # text; interview outcomes also need a type + weight, so seeded rows get
    # sensible defaults (knowledge / weight 3) the teacher can edit afterwards.
    course_outcome_texts = await courses_public.list_course_outcome_texts(db, course_id)
    for position, outcome_text in enumerate(course_outcome_texts, start=1):
        text_clean = (outcome_text or "").strip()
        if not text_clean:
            continue
        db.add(
            InterviewOutcome(
                interview_config_id=config.id,
                position=position,
                outcome_text=text_clean,
                outcome_type="knowledge",
                importance_weight=3,
            )
        )
    if course_outcome_texts:
        await flush_or_conflict(db)
    await _assert_config_threshold_valid(db, config)
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
    # Freeze conduct/grading settings once published. ``exclude_unset`` is what
    # makes this precise: only fields the client actually sent are considered, so
    # a UI that echoes the whole form back does not trip on untouched values.
    data = payload.model_dump(exclude_unset=True)
    changed = set(data)
    published_freeze.assert_config_settings_editable(config, changed)
    if "min_outcomes_to_pass" in data:
        outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
        _assert_threshold_within_outcome_count(
            data["min_outcomes_to_pass"],
            len(outcomes),
        )
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
    await _assert_config_threshold_valid(db, config)
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


async def adaptive_readiness(db: AsyncSession, config_id: UUID) -> dict[str, Any]:
    """Assemble the advisory adaptive-readiness report for a config (Slice 5).

    Read-only: analyses the APPROVED question pool + outcomes with the pure
    :func:`orchestrator.readiness.analyze_readiness`, and reports the deployment
    rollout status per mode. Never blocks publishing — ``blocks_publish`` is
    always False (the hard publish gates live in ``publish_interview_config``).
    """
    from abridgeai.core.config import get_settings  # noqa: PLC0415
    from abridgeai.features.interviews.orchestrator.readiness import (  # noqa: PLC0415
        ReadinessInputs,
        ReadinessOutcome,
        ReadinessQuestion,
        analyze_readiness,
    )

    config = await _require_config(db, config_id)
    approved = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)

    inputs = ReadinessInputs(
        questions=[
            ReadinessQuestion(
                question_id=str(q.id),
                linked_outcome_id=str(q.linked_outcome_id) if q.linked_outcome_id else None,
                difficulty=q.difficulty,
            )
            for q in approved
        ],
        outcomes=[
            ReadinessOutcome(
                outcome_id=str(o.id),
                importance_weight=o.importance_weight or 1,
            )
            for o in outcomes
        ],
        time_limit_minutes=config.time_limit_minutes,
    )
    warnings = analyze_readiness(inputs)

    settings = get_settings()
    return {
        "config_id": config_id,
        "warnings": [w.to_dict() for w in warnings],
        "rollout": {
            "text": settings.adaptive_enabled_for_mode("text"),
            "hybrid": settings.adaptive_enabled_for_mode("hybrid"),
            "voice": settings.adaptive_enabled_for_mode("voice"),
        },
        "blocks_publish": False,
    }


async def add_question(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewQuestion:
    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    await authoring_queries.lock_question_append(db, config_id)
    next_position = await authoring_queries.next_question_position(db, config_id)
    data = payload.model_dump(exclude_unset=True)
    if not str(data.get("prompt_text", "")).strip():
        raise AppError("Question prompt is required")
    linked_outcome_id = data.get("linked_outcome_id")
    if linked_outcome_id is not None:
        await _require_outcome(db, config_id, linked_outcome_id)
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
    await _store_question_embedding(db, question)
    return question


async def check_question_duplicate(
    db: AsyncSession,
    config_id: UUID,
    *,
    prompt_text: str,
    exclude_question_id: UUID | None = None,
) -> dict[str, Any]:
    """Report whether ``prompt_text`` duplicates something already in the bank.

    A read-only advisory check for the authoring UI: it never blocks or mutates
    anything, so a teacher who disagrees with the verdict can still save. Returns
    the verdict as a dict (see :class:`DuplicateVerdict`) plus ``enabled`` so the
    caller can distinguish "feature off" from "nothing similar found".

    Returns ``enabled: False`` without any provider call when
    ``interview_dedup_enabled`` is off.
    """
    await _require_config(db, config_id)
    settings = get_settings()
    if not settings.interview_dedup_enabled:
        return {"enabled": False, **NOT_DUPLICATE.to_dict()}

    verdict, _embedding = await check_duplicate(
        db,
        config_id=config_id,
        prompt_text=prompt_text,
        embedding_client=EmbeddingClient(),
        exclude_question_id=exclude_question_id,
    )
    return {"enabled": True, **verdict.to_dict()}


async def _store_question_embedding(db: AsyncSession, question: InterviewQuestion) -> None:
    """Best-effort: persist the question's vector so it can be matched later.

    Thin single-row wrapper over :func:`dedup.store_question_embeddings`, which
    owns the feature gate, the pgvector text formatting and the swallow-and-log
    policy — the generation pipeline needs the batch form, and two copies of that
    UPDATE is how the two paths drifted apart in the first place.
    """
    await store_question_embeddings(
        db,
        question_ids=[question.id],
        prompt_texts=[question.prompt_text],
    )


async def update_question(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewQuestion:
    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    question = await _require_question(db, config_id, question_id)
    data = payload.model_dump(exclude_unset=True)
    prompt_text = str(data.get("prompt_text", question.prompt_text) or "").strip()
    if not prompt_text:
        raise AppError("Question prompt is required")
    if data.get("question_type", question.question_type) is None:
        raise AppError("Question type is required")
    if "linked_outcome_id" in data and data["linked_outcome_id"] is not None:
        await _require_outcome(db, config_id, data["linked_outcome_id"])
    prompt_changed = "prompt_text" in data and prompt_text != question.prompt_text
    if "prompt_text" in data:
        data["prompt_text"] = prompt_text

    semantic_fields = (
        "prompt_text",
        "question_type",
        "difficulty",
        "linked_outcome_id",
    )
    semantic_group_changed = any(
        key in data and data[key] != getattr(question, key) for key in semantic_fields
    )
    variant_group_id = question.variant_group_id
    if semantic_group_changed and variant_group_id is not None:
        from sqlalchemy import update  # noqa: PLC0415

        await db.execute(
            update(InterviewQuestion)
            .where(
                InterviewQuestion.interview_config_id == config_id,
                InterviewQuestion.variant_group_id == variant_group_id,
                InterviewQuestion.deleted_at.is_(None),
            )
            .values(variant_group_id=None)
        )
        question.variant_group_id = None

    for key in (
        "prompt_text",
        "question_type",
        "difficulty",
        "model_answer",
        "linked_outcome_id",
        "position",
        "review_status",
    ):
        if key in data:
            setattr(question, key, data[key])
    question.reviewed_by = actor.user_id
    question.reviewed_at = utcnow()
    await flush_or_conflict(db)
    await db.refresh(question)
    # Re-embed only when the text actually changed: a stale vector would make the
    # bank match against wording the question no longer has, which is worse than
    # having no vector at all. Edits to difficulty/outcome/status leave it alone.
    if prompt_changed:
        await _store_question_embedding(db, question)
    return question


async def approve_question_variants(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    actor: CurrentUser,
) -> int:
    """Approve the active members of one logical question."""
    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    question = await _require_question(db, config_id, question_id)

    from sqlalchemy import select  # noqa: PLC0415

    if question.variant_group_id is None:
        questions = [question]
    else:
        result = await db.execute(
            select(InterviewQuestion)
            .where(
                InterviewQuestion.interview_config_id == config_id,
                InterviewQuestion.variant_group_id == question.variant_group_id,
                InterviewQuestion.deleted_at.is_(None),
            )
            .order_by(InterviewQuestion.position)
        )
        questions = list(result.scalars().all())

    reviewed_at = utcnow()
    for member in questions:
        member.review_status = "approved"
        member.reviewed_by = actor.user_id
        member.reviewed_at = reviewed_at
    await flush_or_conflict(db)
    return len(questions)


async def delete_question(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    actor: CurrentUser,
) -> None:
    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    question = await _require_question(db, config_id, question_id)
    await soft_delete_cascade(db, question, actor_id=actor.user_id)


async def add_outcome(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewOutcome:
    config = await _require_config(db, config_id)
    # The outcomes are the grading criteria; a published interview must not
    # grow a criterion mid-cohort (see published_freeze.py).
    published_freeze.assert_learning_outcomes_editable(config)
    data = payload.model_dump(exclude_unset=True)
    outcome_text = str(data["outcome_text"] or "").strip()
    if not outcome_text:
        raise AppError("Outcome text is required")
    next_position = await authoring_queries.next_outcome_position(db, config_id)
    outcome = InterviewOutcome(
        interview_config_id=config_id,
        position=data.get("position") or next_position,
        outcome_text=outcome_text,
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
    config = await _require_config(db, config_id)
    # Reweighting an outcome changes how every answer scores; frozen on publish.
    published_freeze.assert_learning_outcomes_editable(config)
    outcome = await _require_outcome(db, config_id, outcome_id)
    data = payload.model_dump(exclude_unset=True)
    if "outcome_text" in data:
        outcome_text = str(data["outcome_text"] or "").strip()
        if not outcome_text:
            raise AppError("Outcome text is required")
        data["outcome_text"] = outcome_text
    for key in ("outcome_text", "outcome_type", "importance_weight", "position"):
        if key in data:
            setattr(outcome, key, data[key])
    if data:
        outcome.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(outcome)
    return outcome


async def delete_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
    actor: CurrentUser,
) -> None:
    config = await _require_config(db, config_id)
    # Dropping a criterion retroactively rewrites what the interview measured.
    published_freeze.assert_learning_outcomes_editable(config)
    outcome = await _require_outcome(db, config_id, outcome_id)
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    _assert_threshold_within_outcome_count(
        config.min_outcomes_to_pass,
        max(0, len(outcomes) - 1),
    )
    await soft_delete_cascade(db, outcome, actor_id=actor.user_id)


def _assert_role_only_has_a_preferred_type(
    config: InterviewConfig,
    request_data: dict[str, Any],
) -> None:
    """Reject ``role_only`` on a config whose interviewer role has no type.

    ``role_only`` means "every question is the type this interviewer role asks".
    The mapping lives in ``orchestrator.role_question_filter.preferred_type`` and
    is deliberately ``None`` for the GENERIC_ASSISTANT — that role has no type
    bias, which is what keeps opted-out configs byte-identical to the pre-feature
    engine.

    The pipeline handles that case by silently dropping back to legacy mixed
    generation (``resolve_variant_mode``). Silently is the problem: the teacher
    picked "match the interviewer role", got a mixed bank, and nothing said so.
    Since the generic assistant is the DEFAULT role — most configs have it — the
    combination is easy to hit by accident.

    Refused at enqueue instead, naming the fix (pick a role, or pick another
    generation mode). The degrade in ``resolve_variant_mode`` stays as the safety
    net for runs already queued and for callers that bypass this service.
    """
    if resolve_variant_strategy(request_data) != "role_only":
        return
    from abridgeai.features.interviews.orchestrator.interviewer_identity import (  # noqa: PLC0415
        identity_from_config,
    )
    from abridgeai.features.interviews.orchestrator.role_question_filter import (  # noqa: PLC0415
        preferred_type,
    )

    role = identity_from_config(config.persona_profile_json).role
    if preferred_type(role) is not None:
        return
    raise AppError(
        "role_only_requires_a_typed_interviewer_role: this interview's "
        f"interviewer role ({role.value}) asks no particular question type, so "
        "matching it would produce the ordinary mixed bank. Choose an "
        "interviewer role in Settings, or pick a different question mode."
    )


def _assert_question_count_within_strategy_cap(
    request_data: dict[str, Any],
    effective_supplementary: str | None,
) -> None:
    """Reject a count whose EFFECTIVE row budget would exceed the hard cap.

    ``all_angles`` fans every logical question out into one row per angle, so a
    typed count of N asks the pipeline for N x len(VARIANT_ANGLES) rows. The
    schema's ``le=50`` bounds only the typed number, so 50 used to mean 200 rows
    — a target the pipeline cannot hit (it requires an EXACT match and accepts
    only whole 4-angle groups) and so fails AFTER burning every backfill round.

    Caught here, before the run row and the ARQ job exist, so the teacher gets
    an actionable 400 instead of a run that spends LLM budget and then dies with
    "Generation underfilled". Resolution goes through the same precedence the
    pipeline uses (form value -> supplementary JSON -> default), so a count
    smuggled in via ``supplementary_instructions`` is bounded too.
    """
    strategy = resolve_variant_strategy(request_data)
    cap = max_logical_question_count(strategy)
    requested = resolve_question_count(
        run_config_json=request_data,
        supplementary=effective_supplementary,
    )
    if requested <= cap:
        return
    raise AppError(
        "question_count_exceeds_variant_cap: multi-angle generation produces "
        f"{len(VARIANT_ANGLES)} questions per logical question, so at most {cap} "
        f"logical questions can be requested (you asked for {requested}). "
        f"Lower the count to {cap} or switch off multi-angle mode."
    )


async def _assert_target_outcomes_in_config(
    db: AsyncSession, config_id: UUID, request_data: dict[str, Any]
) -> None:
    """Reject a target-outcome filter that does not resolve inside this config.

    An empty / absent filter means "every outcome" (prior behaviour). A
    non-empty one must name live outcomes of THIS config: soft-deleted,
    unknown, or foreign-config ids would otherwise be dropped by the pipeline
    filter and silently widen the run back to all outcomes.
    """
    raw_ids = request_data.get("target_outcome_ids") or []
    target_ids = [UUID(str(outcome_id)) for outcome_id in raw_ids]
    if not target_ids:
        return
    if len(target_ids) != len(set(target_ids)):
        raise AppError("target_outcome_ids contains duplicates")
    live_ids = {
        outcome.id for outcome in await authoring_queries.list_outcomes_for_config(db, config_id)
    }
    if set(target_ids) - live_ids:
        raise AppError("target_outcome_ids must belong to the interview config")


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
    published_freeze.assert_questions_editable(config)
    request_data = request.model_dump(exclude_unset=True, mode="json") if request else {}

    request_course_id = request_data.get("course_id")
    if request_course_id is None or UUID(str(request_course_id)) != config.course_id:
        raise AppError("course_id does not match the interview config")
    request_module_id = request_data.get("module_id")
    if request_module_id is None or UUID(str(request_module_id)) != config.module_id:
        raise AppError("module_id does not match the interview config")

    # Module-scoped retrieval (§QGen-scope): each selected module expands to
    # its lessons, which are merged into the ``source_lesson_ids`` retrieval
    # filter (retrieval reads ``lesson_ids`` / ``source_lesson_ids`` from
    # config_json). When no modules are chosen we default to the interview's
    # own module so generation stays scoped to it rather than the whole course.
    raw_module_ids = request_data.get("source_module_ids") or []
    module_ids: list[UUID] = [UUID(str(m)) for m in raw_module_ids]
    if not module_ids and config.module_id is not None:
        module_ids = [config.module_id]
    for module_id in module_ids:
        await _require_module_in_course(db, module_id, config.course_id)

    explicit_lesson_ids = [UUID(str(x)) for x in (request_data.get("source_lesson_ids") or [])]
    await _require_lessons_in_course(db, explicit_lesson_ids, config.course_id)
    module_lesson_ids = await courses_public.list_lesson_ids_for_modules(db, module_ids)
    merged_lesson_ids = sorted({*explicit_lesson_ids, *module_lesson_ids})

    await _assert_target_outcomes_in_config(db, config_id, request_data)

    # ``type_weights`` is persisted here and read back by the VALIDATION stage
    # to police the produced type mix. It MUST be derived from the same
    # supplementary_instructions the GENERATION stage will prompt with — if the
    # run overrides the field, deriving the weights from the config column would
    # make validation reject the very drafts generation was asked for.
    effective_supplementary = resolve_supplementary(
        request_data, config.supplementary_instructions
    )
    _assert_question_count_within_strategy_cap(request_data, effective_supplementary)
    _assert_role_only_has_a_preferred_type(config, request_data)
    config_json: dict[str, Any] = dict(request_data) | {
        "interview_config_id": str(config.id),
        "type_weights": {
            key: value / 100 for key, value in resolve_type_mix(effective_supplementary).items()
        },
        "course_id": str(config.course_id),
        "module_id": str(config.module_id),
        "source_module_ids": [str(m) for m in module_ids],
        "source_lesson_ids": [str(x) for x in merged_lesson_ids],
    }
    run = await quizzes_public.create_generation_run(
        db,
        kind="interview",
        source_scope_kind="module",
        course_id=config.course_id,
        module_id=config.module_id,
        requested_by=actor.user_id,
        config_json=config_json,
        dedup_key=f"interview-generation:{config.id}",
    )
    config.generation_run_id = run.id
    await db.commit()

    try:
        if arq_pool is not None:
            await arq_pool.enqueue_job(  # type: ignore[attr-defined]
                _RUN_INTERVIEW_GENERATION_TASK, actor.user_id, run.id
            )
    except Exception:
        await db.rollback()
        await quizzes_public.mark_pending_generation_run_failed(
            db, run.id, "Generation worker could not be queued"
        )
        await db.commit()
        raise
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


# --------------------------------------------------------------------------- #
# Course-scoped interview question bank (§QBank-1)
# --------------------------------------------------------------------------- #


async def _require_course(db: AsyncSession, course_id: UUID) -> None:
    """Ensure the course exists (permission is enforced at the router edge)."""
    course = await courses_public.get_course_by_id(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")


async def list_question_bank(db: AsyncSession, course_id: UUID) -> list[InterviewQuestionBankItem]:
    """All non-deleted bank items for a course, newest first."""
    from sqlalchemy import select  # noqa: PLC0415

    await _require_course(db, course_id)
    stmt = (
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .order_by(InterviewQuestionBankItem.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def add_to_question_bank(
    db: AsyncSession,
    course_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewQuestionBankItemCreate at edge
    actor: CurrentUser,
) -> InterviewQuestionBankItem:
    """Add a reusable question to the course bank (copy semantics)."""
    await _require_course(db, course_id)
    data = payload.model_dump(exclude_unset=True)
    item = await _create_bank_item(
        db,
        course_id=course_id,
        data=data,
        actor=actor,
        variant_group_id=None,
    )
    await flush_or_conflict(db)
    await db.refresh(item)
    return item


_LOGICAL_QUESTION_ANGLE_ORDER = (
    "technical",
    "system_design",
    "situational",
    "behavioral",
)
_LOGICAL_QUESTION_ANGLES = set(_LOGICAL_QUESTION_ANGLE_ORDER)
_LOGICAL_QUESTION_ANGLE_RANK = {
    angle: index for index, angle in enumerate(_LOGICAL_QUESTION_ANGLE_ORDER)
}


def _logical_angle_rank(item: InterviewQuestionBankItem) -> int:
    return _LOGICAL_QUESTION_ANGLE_RANK.get(
        item.question_type, len(_LOGICAL_QUESTION_ANGLE_ORDER)
    )


def _normalise_prompt(value: str) -> str:
    return value.strip().lower()


async def _active_bank_group(
    db: AsyncSession, course_id: UUID, group_id: UUID
) -> list[InterviewQuestionBankItem]:
    from sqlalchemy import select  # noqa: PLC0415

    result = await db.execute(
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.variant_group_id == group_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .order_by(InterviewQuestionBankItem.created_at)
    )
    # Canonical interviewer-angle order (created_at/id only break ties).
    return sorted(
        result.scalars().all(),
        key=lambda item: (_logical_angle_rank(item), item.created_at, item.id),
    )


async def _create_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    data: dict[str, Any],
    actor: CurrentUser,
    variant_group_id: UUID | None,
) -> InterviewQuestionBankItem:
    question_type = data["question_type"]
    if variant_group_id is not None and question_type not in _LOGICAL_QUESTION_ANGLES:
        raise AppError("logical_question_angle_required")
    prompt_text = str(data["prompt_text"] or "").strip()
    if not prompt_text:
        raise AppError("Question prompt is required")
    source_config_id = data.get("source_config_id")
    if source_config_id is not None:
        config = await _require_config(db, UUID(str(source_config_id)))
        if config.course_id != course_id:
            raise AppError("source_config_id does not belong to course")
    item = InterviewQuestionBankItem(
        course_id=course_id,
        prompt_text=prompt_text,
        question_type=question_type,
        difficulty=data.get("difficulty"),
        model_answer=(data.get("model_answer") or "").strip() or None,
        tags_json=data.get("tags", []),
        variant_group_id=variant_group_id,
        source_config_id=source_config_id,
        created_by=actor.user_id,
    )
    db.add(item)
    return item


async def create_question_bank_logical_group(
    db: AsyncSession,
    course_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> list[InterviewQuestionBankItem]:
    """Copy one complete four-angle logical question into the course bank."""
    await _require_course(db, course_id)
    payload_items = list(payload.items)
    prompts = [_normalise_prompt(str(item.prompt_text or "")) for item in payload_items]
    if len(prompts) != len(set(prompts)):
        raise ConflictError("logical_question_prompt_duplicate")
    group_id = uuid4()
    items = [
        await _create_bank_item(
            db,
            course_id=course_id,
            data=item.model_dump(exclude_unset=True),
            actor=actor,
            variant_group_id=group_id,
        )
        for item in payload.items
    ]
    await flush_or_conflict(db)
    for item in items:
        await db.refresh(item)
    return items


async def add_question_bank_sibling(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> list[InterviewQuestionBankItem]:
    """Append one missing angle to a singleton or partial logical bank group."""
    from sqlalchemy import select, update  # noqa: PLC0415

    anchor = (
        await db.execute(
            select(InterviewQuestionBankItem)
            .where(
                InterviewQuestionBankItem.id == item_id,
                InterviewQuestionBankItem.course_id == course_id,
                InterviewQuestionBankItem.deleted_at.is_(None),
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if anchor is None:
        raise NotFoundError(f"Question bank item {item_id} not found")

    group_id = anchor.variant_group_id or uuid4()
    # Different children of an existing group are distinct rows, so their row
    # locks alone cannot serialize a same-angle sibling add. Coordinate writes
    # by the durable group id; singleton promotion is already serialized by the
    # anchor row lock above.
    if anchor.variant_group_id is not None:
        await authoring_queries.lock_bank_group(db, course_id, group_id)
    siblings = (
        await _active_bank_group(db, course_id, group_id)
        if anchor.variant_group_id
        else [anchor]
    )
    question_type = payload.question_type
    prompt_text = _normalise_prompt(str(payload.prompt_text or ""))
    if prompt_text in {_normalise_prompt(item.prompt_text) for item in siblings}:
        raise ConflictError("logical_question_prompt_duplicate")
    if question_type in {item.question_type for item in siblings}:
        raise ConflictError("logical_question_angle_already_present")
    if len(siblings) >= 4:
        raise ConflictError("logical_question_group_is_full")

    if anchor.variant_group_id is None:
        await db.execute(
            update(InterviewQuestionBankItem)
            .where(InterviewQuestionBankItem.id == anchor.id)
            .values(variant_group_id=group_id)
        )
        anchor.variant_group_id = group_id
    await _create_bank_item(
        db,
        course_id=course_id,
        data=payload.model_dump(exclude_unset=True),
        actor=actor,
        variant_group_id=group_id,
    )
    await flush_or_conflict(db)
    return await _active_bank_group(db, course_id, group_id)


def _order_bank_import(
    *,
    item_ids: list[UUID],
    selected: list[InterviewQuestionBankItem],
    expanded: list[InterviewQuestionBankItem],
) -> list[InterviewQuestionBankItem]:
    """Flatten the selection into insert order.

    Preserves the user's selected unit order, but always persists members of a
    logical question in its canonical interviewer-angle sequence. SQL and UUID
    order are intentionally never part of the authoring experience.
    """
    selected_by_id = {item.id: item for item in selected}
    unit_keys: list[tuple[str, UUID]] = []
    seen_units: set[tuple[str, UUID]] = set()
    for item_id in item_ids:
        item = selected_by_id[item_id]
        key = (
            ("group", item.variant_group_id)
            if item.variant_group_id is not None
            else ("item", item.id)
        )
        if key not in seen_units:
            seen_units.add(key)
            unit_keys.append(key)

    expanded_by_group: dict[UUID, list[InterviewQuestionBankItem]] = {}
    expanded_by_id = {item.id: item for item in expanded}
    for item in expanded:
        if item.variant_group_id is not None:
            expanded_by_group.setdefault(item.variant_group_id, []).append(item)

    ordered: list[InterviewQuestionBankItem] = []
    for kind_unit, unit_id in unit_keys:
        if kind_unit == "group":
            ordered.extend(
                sorted(
                    expanded_by_group[unit_id],
                    key=lambda item: (_logical_angle_rank(item), item.created_at, item.id),
                )
            )
        else:
            ordered.append(expanded_by_id[unit_id])
    return ordered


async def import_question_bank_items(
    db: AsyncSession,
    config_id: UUID,
    item_ids: list[UUID],
    actor: CurrentUser,
) -> list[InterviewQuestion]:
    """Atomically append chosen bank items, expanding logical siblings first."""
    from sqlalchemy import select  # noqa: PLC0415

    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    if len(item_ids) != len(set(item_ids)):
        raise AppError("question_bank_import_item_ids contains duplicates")
    await authoring_queries.lock_question_append(db, config_id)
    selected = list(
        (
            await db.execute(
                select(InterviewQuestionBankItem).where(
                    InterviewQuestionBankItem.id.in_(set(item_ids)),
                    InterviewQuestionBankItem.course_id == config.course_id,
                    InterviewQuestionBankItem.deleted_at.is_(None),
                )
            )
        ).scalars().all()
    )
    if len(selected) != len(set(item_ids)):
        raise NotFoundError("One or more question-bank items were not found")

    group_ids = {item.variant_group_id for item in selected if item.variant_group_id is not None}
    expanded = list(selected)
    if group_ids:
        siblings = list(
            (
                await db.execute(
                    select(InterviewQuestionBankItem).where(
                        InterviewQuestionBankItem.course_id == config.course_id,
                        InterviewQuestionBankItem.variant_group_id.in_(group_ids),
                        InterviewQuestionBankItem.deleted_at.is_(None),
                    )
                )
            ).scalars().all()
        )
        expanded = list({item.id: item for item in [*selected, *siblings]}.values())

    expanded_prompts = [_normalise_prompt(item.prompt_text) for item in expanded]
    if len(expanded_prompts) != len(set(expanded_prompts)):
        raise ConflictError("question_bank_import_prompt_duplicate")

    existing = {
        _normalise_prompt(question.prompt_text)
        for question in await authoring_queries.list_questions_for_config(db, config_id)
    }
    collisions = [
        item.id for item in expanded if _normalise_prompt(item.prompt_text) in existing
    ]
    if collisions:
        raise ConflictError("question_bank_import_prompt_already_exists")

    # Preserve the user's selected unit order, but always persist members of a
    # logical question in its canonical interviewer-angle sequence.
    ordered = _order_bank_import(item_ids=item_ids, selected=selected, expanded=expanded)

    next_position = await authoring_queries.next_question_position(db, config_id)
    group_map = {group_id: uuid4() for group_id in group_ids}
    created: list[InterviewQuestion] = []
    for offset, item in enumerate(ordered):
        question = InterviewQuestion(
            interview_config_id=config_id,
            position=next_position + offset,
            question_type=item.question_type,
            prompt_text=item.prompt_text,
            difficulty=item.difficulty,
            model_answer=item.model_answer,
            review_status="approved",
            ai_generated=False,
            source_refs_json=[],
            source_module_ids=[],
            variant_group_id=(
                group_map[item.variant_group_id]
                if item.variant_group_id is not None
                else None
            ),
            reviewed_by=actor.user_id,
            reviewed_at=utcnow(),
            created_by=actor.user_id,
        )
        db.add(question)
        created.append(question)
    await flush_or_conflict(db)
    for question in created:
        await db.refresh(question)
    await store_question_embeddings(
        db,
        question_ids=[question.id for question in created],
        prompt_texts=[question.prompt_text for question in created],
    )
    return created


async def _assert_grouped_edit_keeps_group_valid(
    db: AsyncSession,
    *,
    course_id: UUID,
    item: InterviewQuestionBankItem,
    final_type: str | None,
    final_prompt: str,
) -> None:
    """Guard an edit of a logical-group member against its live siblings.

    Holds the group advisory lock first so two concurrent edits of different
    children cannot both pass the check and land the same angle. Errors are
    domain errors (400 / 409) — never an IntegrityError surfacing as a 500.
    """
    group_id = item.variant_group_id
    assert group_id is not None  # noqa: S101 -- caller checks; keeps mypy narrow
    await authoring_queries.lock_bank_group(db, course_id, group_id)
    siblings = [
        sibling
        for sibling in await _active_bank_group(db, course_id, group_id)
        if sibling.id != item.id
    ]
    if final_type not in _LOGICAL_QUESTION_ANGLES:
        raise AppError("logical_question_angle_required")
    if final_type in {sibling.question_type for sibling in siblings}:
        raise ConflictError("logical_question_angle_already_present")
    if _normalise_prompt(final_prompt) in {
        _normalise_prompt(sibling.prompt_text) for sibling in siblings
    }:
        raise ConflictError("logical_question_prompt_duplicate")


async def update_question_bank_item(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewQuestionBankItemUpdate at edge
    actor: CurrentUser,
) -> InterviewQuestionBankItem:
    """Edit one bank item without breaking its logical-question siblings."""
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.id == item_id,
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .with_for_update()
    )
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Question bank item {item_id} not found")

    data = payload.model_dump(exclude_unset=True)
    final_prompt = str(data.get("prompt_text", item.prompt_text) or "").strip()
    if not final_prompt:
        raise AppError("Question prompt is required")
    if "prompt_text" in data:
        data["prompt_text"] = final_prompt
    final_type = data.get("question_type", item.question_type)
    if final_type is None:
        # NOT NULL column: an explicit null is a bad request, not a DB error.
        raise AppError("question_type cannot be null")

    if item.variant_group_id is not None:
        await _assert_grouped_edit_keeps_group_valid(
            db,
            course_id=course_id,
            item=item,
            final_type=final_type,
            final_prompt=final_prompt,
        )

    if "tags" in data:
        item.tags_json = data.pop("tags")
    for key, value in data.items():
        setattr(item, key, value)
    item.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(item)
    return item


async def delete_question_bank_item(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    actor: CurrentUser,
) -> None:
    """Soft-delete a bank item. Already-imported questions are untouched."""
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(InterviewQuestionBankItem).where(
        InterviewQuestionBankItem.id == item_id,
        InterviewQuestionBankItem.course_id == course_id,
        InterviewQuestionBankItem.deleted_at.is_(None),
    )
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Question bank item {item_id} not found")
    await soft_delete_cascade(db, item, actor_id=actor.user_id)


__all__ = [
    "add_outcome",
    "add_question",
    "add_question_bank_sibling",
    "add_to_question_bank",
    "create_question_bank_logical_group",
    "import_question_bank_items",
    "archive_interview_config",
    "approve_question_variants",
    "create_interview_config",
    "delete_interview_config",
    "delete_outcome",
    "delete_question",
    "delete_question_bank_item",
    "get_generation_run",
    "list_question_bank",
    "publish_interview_config",
    "update_question_bank_item",
    "start_generation_run",
    "unarchive_interview_config",
    "update_interview_config",
    "update_outcome",
    "update_question",
]
