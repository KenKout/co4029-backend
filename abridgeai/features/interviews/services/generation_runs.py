"""Interview generation-run authoring (T6.11).

Generation request validation + the ARQ-enqueue trigger and run lookup,
extracted from ``services.authoring`` (2026-09-01) to keep that module
under the interviews LOC ratchet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.exceptions import AppError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
    max_logical_question_count,
    resolve_question_count,
    resolve_supplementary,
    resolve_type_mix,
    resolve_variant_strategy,
)
from abridgeai.features.interviews.models import InterviewConfig
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import published_freeze
from abridgeai.features.interviews.services._shared import (
    _require_config,
    _require_lessons_in_course,
    _require_module_in_course,
)
from abridgeai.features.quizzes.api import public as quizzes_public
from abridgeai.features.quizzes.api.public import GenerationRunDTO

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_RUN_INTERVIEW_GENERATION_TASK = "run_interview_generation_task"


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
    effective_supplementary = resolve_supplementary(request_data, config.supplementary_instructions)
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


__all__ = ["get_generation_run", "start_generation_run"]
