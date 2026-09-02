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

from typing import TYPE_CHECKING, Any
from uuid import UUID

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
from abridgeai.features.interviews.dedup import (
    NOT_DUPLICATE,
    check_duplicate,
    store_question_embeddings,
)
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import published_freeze
from abridgeai.features.interviews.services._shared import (
    _assert_config_threshold_valid,
    _assert_threshold_within_outcome_count,
    _normalise_title,
    _require_config,
    _require_module_in_course,
    _require_outcome,
)
from abridgeai.features.interviews.services.bank import (
    add_question_bank_sibling,
    add_to_question_bank,
    create_question_bank_logical_group,
    delete_question_bank_group,
    delete_question_bank_item,
    import_question_bank_items,
    list_question_bank,
    update_question_bank_item,
)

# Outcome CRUD, generation-run enqueue, and the course question bank were
# extracted into sibling modules to keep this file under the LOC ratchet;
# re-exported so every authoring_service.* call site keeps working.
from abridgeai.features.interviews.services.generation_runs import (
    get_generation_run,
    start_generation_run,
)
from abridgeai.features.interviews.services.outcomes import (
    add_outcome,
    delete_outcome,
    update_outcome,
)

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


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


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


async def _assert_title_free_in_module(
    db: AsyncSession,
    *,
    module_id: UUID,
    title: str,
    exclude_config_id: UUID | None = None,
) -> None:
    """Reject a title already used by another live interview in the module.

    Titles are the only handle a teacher has on an interview in the curriculum
    accordion and in every picker, so two interviews called "DW Interview" in
    one module are indistinguishable in the UI. The slug is already unique per
    module (``uq_interview_configs_module_slug``); this applies the SAME scope
    to the human-readable name so the two never disagree.

    Enforced in the service, not as a UNIQUE index: rows created before this
    rule exists already collide, and a migration adding the index would fail on
    that data. Comparison ignores case and surrounding whitespace, which is how
    a teacher reads two titles as "the same".
    """
    from sqlalchemy import select  # noqa: PLC0415

    normalised = _normalise_title(title)
    if not normalised:
        return
    stmt = select(InterviewConfig.id, InterviewConfig.title).where(
        InterviewConfig.module_id == module_id,
        InterviewConfig.deleted_at.is_(None),
    )
    if exclude_config_id is not None:
        stmt = stmt.where(InterviewConfig.id != exclude_config_id)
    rows = await db.execute(stmt)
    for _existing_id, existing_title in rows.all():
        if _normalise_title(existing_title or "") == normalised:
            raise ConflictError(
                "interview_title_duplicate: another interview in this module "
                "already uses this title"
            )


async def _require_question(
    db: AsyncSession, config_id: UUID, question_id: UUID
) -> InterviewQuestion:
    question = await db.get(InterviewQuestion, question_id)
    if question is None or question.interview_config_id != config_id:
        raise NotFoundError(f"Interview question {question_id} not found")
    return question


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
    # Reject a title another live interview in this module already uses, before
    # anything is written — two same-named interviews are indistinguishable in
    # the curriculum accordion.
    await _assert_title_free_in_module(db, module_id=module_id, title=data["title"])
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
        cooldown_minutes=data.get("cooldown_minutes"),
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
    # Renaming into a sibling's title is the same collision as creating one —
    # guard the PATCH too, or the create-time rule is trivially bypassed.
    if "title" in data:
        await _assert_title_free_in_module(
            db,
            module_id=config.module_id,
            title=str(data["title"] or ""),
            exclude_config_id=config.id,
        )
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


async def delete_question_variants(
    db: AsyncSession,
    config_id: UUID,
    question_id: UUID,
    actor: CurrentUser,
) -> int:
    """Soft-delete every live member of one logical question.

    The anchored question establishes config scope. A singleton remains a
    one-row delete so callers can expose one grouped action without a separate
    branch; grouped members are locked and deleted as one transaction.
    """
    from sqlalchemy import select  # noqa: PLC0415

    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    question = await _require_question(db, config_id, question_id)
    if question.variant_group_id is None:
        await soft_delete_cascade(db, question, actor_id=actor.user_id)
        return 1

    result = await db.execute(
        select(InterviewQuestion)
        .where(
            InterviewQuestion.interview_config_id == config_id,
            InterviewQuestion.variant_group_id == question.variant_group_id,
            InterviewQuestion.deleted_at.is_(None),
        )
        .order_by(InterviewQuestion.position)
        .with_for_update()
    )
    questions = list(result.scalars().all())
    for member in questions:
        await soft_delete_cascade(db, member, actor_id=actor.user_id)
    return len(questions)


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
    "delete_question_variants",
    "delete_question_bank_group",
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
