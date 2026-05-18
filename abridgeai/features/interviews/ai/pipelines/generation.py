"""Interview generation pipeline orchestrator (T6.10).

Composes the four T6.4-T6.6 stages and persists accepted drafts:

    retrieval -> generation -> validation -> persistence

The pipeline does NOT make its own LLM calls; every stage emits its own
``ai_model_calls`` row via :class:`LLMGateway`. ``run.id`` is threaded
into each stage as ``pipeline_run_id`` so audit rows for one run share
a single ``ai_pipeline_runs`` parent.

Cross-feature decoupling: the ``generation_runs`` row is owned by
``features.quizzes``; this pipeline never imports the quizzes ORM.
Reads go through ``quizzes.api.public.get_generation_run`` and state
mutations use raw ``UPDATE`` against the ``generation_runs`` table
(which has no ``SoftDeleteMixin``, so the audit-bypass lint patterns
in ``tests/lint/test_no_audit_bypass.py`` do not apply).

Stages T6.7 (followup), T6.8 (evaluation), T6.9 (gap report) are NOT
in this pipeline — followup runs at session runtime; evaluation +
gap-report run after submit. Both are wired by T6.11 services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.stages.generation import (
    generate_interview_questions,
)
from abridgeai.features.interviews.ai.stages.retrieval import (
    retrieve_interview_context,
)
from abridgeai.features.interviews.ai.stages.validation import (
    validate_interview_questions,
)
from abridgeai.features.interviews.models import InterviewConfig, InterviewQuestion
from abridgeai.features.interviews.queries.authoring import (
    list_outcomes_for_config,
    next_question_position,
)
from abridgeai.features.quizzes.api import public as quizzes_public

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )
    from abridgeai.features.interviews.ai.stages.retrieval.logic import (
        InterviewRetrievalContext,
    )
    from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict


@dataclass
class _RunState:
    id: UUID
    course_id: UUID | None
    config_json: dict[str, Any]


async def run_interview_generation(
    db: AsyncSession,
    generation_run_id: UUID,
) -> None:
    """Drive retrieval → generation → validation → persistence.

    Reads ``GenerationRun.config_json['interview_config_id']`` to resolve
    the target config. Threads ``run.id`` through every LLM-emitting
    stage so ``ai_model_calls`` rows for this run share one
    ``pipeline_run_id``.

    On success: marks ``run.status='completed'``, populates
    ``run.config_json`` with stage-level summaries, and inserts one
    :class:`InterviewQuestion` row per accepted draft (with monotonically
    increasing ``position`` and ``review_status='pending'``).

    On any pipeline-stage failure: rolls the session back, re-fetches
    the run, stamps ``status='failed'`` + the failure message on
    ``config_json``, commits, and re-raises so the ARQ retry budget
    engages.
    """

    run_dto = await quizzes_public.get_generation_run(db, generation_run_id)
    if run_dto is None:
        raise NotFoundError("Generation run not found")
    state = _RunState(
        id=run_dto.id,
        course_id=run_dto.course_id,
        config_json=dict(run_dto.config_json or {}),
    )

    config_id = _config_uuid(state.config_json, "interview_config_id")
    if config_id is None:
        raise NotFoundError("Generation run is missing interview_config_id")
    config = await db.get(InterviewConfig, config_id)
    if config is None:
        raise NotFoundError("Interview config not found for generation run")

    await _update_run(db, state.id, status="running", started_at=utcnow())
    await db.commit()

    try:
        context = await retrieve_interview_context(
            db,
            run=state,
            config=config,
            pipeline_run_id=state.id,
        )
        state.config_json = state.config_json | {
            "retrieval": _retrieval_summary(context),
        }
        await _update_run(db, state.id, config_json=state.config_json)
        await db.commit()

        outcomes = await list_outcomes_for_config(db, config.id)
        drafts = await generate_interview_questions(
            db,
            run=state,
            config=config,
            context=cast("Any", context),
            outcomes=outcomes,
        )

        verdicts = await validate_interview_questions(
            db,
            run=state,
            config=config,
            drafts=cast("Any", drafts),
            context=cast("Any", context),
        )
        accepted = _accepted_drafts(drafts, verdicts)

        await _persist_questions(db, config=config, accepted=accepted)

        state.config_json = state.config_json | {
            "pipeline": {
                "stage": "completed",
                "pipeline_run_id": str(state.id),
                "generation": {
                    "drafts_total": len(drafts),
                    "drafts_accepted": len(accepted),
                    "drafts_rejected": len(drafts) - len(accepted),
                },
                "validation": _validation_summary(verdicts),
            }
        }
        await _update_run(
            db,
            state.id,
            status="completed",
            finished_at=utcnow(),
            config_json=state.config_json,
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        fresh = await quizzes_public.get_generation_run(db, generation_run_id)
        if fresh is None:
            raise
        failure_config = dict(fresh.config_json or {}) | {
            "failure": {"message": str(exc)},
        }
        await _update_run(
            db,
            fresh.id,
            status="failed",
            finished_at=utcnow(),
            config_json=failure_config,
        )
        await db.commit()
        raise


async def _update_run(
    db: AsyncSession,
    run_id: UUID,
    *,
    status: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    config_json: dict[str, Any] | None = None,
) -> None:
    sets: list[str] = []
    params: dict[str, Any] = {"id": run_id}
    if status is not None:
        sets.append("status = :status")
        params["status"] = status
    if started_at is not None:
        sets.append("started_at = :started_at")
        params["started_at"] = started_at
    if finished_at is not None:
        sets.append("finished_at = :finished_at")
        params["finished_at"] = finished_at
    if config_json is not None:
        sets.append("config_json = CAST(:config_json AS jsonb)")
        params["config_json"] = json.dumps(config_json, default=str)
    if not sets:
        return
    sets.append("updated_at = NOW()")
    sql = f"UPDATE generation_runs SET {', '.join(sets)} WHERE id = :id"  # noqa: S608  # sets list is code-controlled, no user input
    await db.execute(text(sql), params)


def _config_uuid(config_json: dict[str, Any] | None, key: str) -> UUID | None:
    if not config_json:
        return None
    raw = config_json.get(key)
    if raw is None:
        return None
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        return None


def _retrieval_summary(context: InterviewRetrievalContext) -> dict[str, Any]:
    return {
        "count": len(context.chunks),
        "kg_concept_count": len(context.kg_concepts),
        "weak_topic_count": len(context.weak_topic_chunks),
    }


def _accepted_drafts(
    drafts: list[InterviewQuestionDraft],
    verdicts: list[Verdict],
) -> list[InterviewQuestionDraft]:
    accepted: list[InterviewQuestionDraft] = []
    for index, draft in enumerate(drafts):
        if index < len(verdicts) and verdicts[index].accepted:
            accepted.append(draft)
    return accepted


def _validation_summary(verdicts: list[Verdict]) -> dict[str, Any]:
    rejected = [v for v in verdicts if not v.accepted]
    failure_codes: dict[str, int] = {}
    for verdict in rejected:
        for criterion in verdict.failed_criteria:
            failure_codes[criterion.value] = failure_codes.get(criterion.value, 0) + 1
    return {
        "accepted": sum(1 for v in verdicts if v.accepted),
        "rejected": len(rejected),
        "failures": failure_codes,
    }


_DIFFICULTY_DRAFT_TO_ORM: dict[str, str] = {
    "easy": "junior",
    "medium": "mid_level",
    "hard": "senior",
}


def _persist_difficulty(value: str | None) -> str | None:
    if value is None:
        return None
    return _DIFFICULTY_DRAFT_TO_ORM.get(value, value)


async def _persist_questions(
    db: AsyncSession,
    *,
    config: InterviewConfig,
    accepted: list[InterviewQuestionDraft],
) -> None:
    for draft in accepted:
        position = await next_question_position(db, config.id)
        db.add(
            InterviewQuestion(
                interview_config_id=config.id,
                linked_outcome_id=draft.linked_outcome_id,
                position=position,
                question_type=draft.question_type,
                prompt_text=draft.prompt_text,
                difficulty=_persist_difficulty(draft.difficulty),
                review_status="pending",
                ai_generated=True,
                source_refs_json=[str(c) for c in draft.source_refs],
            )
        )
        await db.flush()


__all__ = ["run_interview_generation"]
