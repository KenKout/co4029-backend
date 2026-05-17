"""Interview generation pipeline orchestrator (T6.10).

Composes the four T6.4-T6.6 stages and persists accepted drafts:

    retrieval -> generation -> validation -> persistence

The pipeline does NOT make its own LLM calls; every stage emits its own
``ai_model_calls`` row via :class:`LLMGateway`. ``run.id`` is threaded
into each stage as ``pipeline_run_id`` so audit rows for one run share
a single ``ai_pipeline_runs`` parent.

Stages T6.7 (followup), T6.8 (evaluation), T6.9 (gap report) are NOT in
this pipeline — followup runs at session runtime; evaluation + gap-report
run after submit. Both are wired by T6.11 services.

Mirrors :mod:`abridgeai.features.quizzes.ai.pipelines.full` (T5.10) and
:mod:`abridgeai.features.quizzes.services.generation` (T5.13) for status /
failure-recovery contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

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
from abridgeai.features.quizzes.models import GenerationRun

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )
    from abridgeai.features.interviews.ai.stages.retrieval.logic import (
        InterviewRetrievalContext,
    )
    from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict


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

    run = await db.get(GenerationRun, generation_run_id)
    if run is None:
        raise NotFoundError("Generation run not found")
    run_id = run.id

    config_id = _config_uuid(run.config_json, "interview_config_id")
    if config_id is None:
        raise NotFoundError("Generation run is missing interview_config_id")
    config = await db.get(InterviewConfig, config_id)
    if config is None:
        raise NotFoundError("Interview config not found for generation run")

    run.status = "running"
    run.started_at = utcnow()
    await db.commit()

    try:
        context = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            pipeline_run_id=run.id,
        )
        run.config_json = dict(run.config_json or {}) | {
            "retrieval": _retrieval_summary(context),
        }
        await db.commit()

        outcomes = await list_outcomes_for_config(db, config.id)
        drafts = await generate_interview_questions(
            db,
            run=run,
            config=config,
            context=cast("Any", context),
            outcomes=outcomes,
        )

        verdicts = await validate_interview_questions(
            db,
            run=run,
            config=config,
            drafts=cast("Any", drafts),
            context=cast("Any", context),
        )
        accepted = _accepted_drafts(drafts, verdicts)

        await _persist_questions(db, config=config, accepted=accepted)

        run.status = "completed"
        run.finished_at = utcnow()
        run.config_json = dict(run.config_json or {}) | {
            "pipeline": {
                "stage": "completed",
                "pipeline_run_id": str(run.id),
                "generation": {
                    "drafts_total": len(drafts),
                    "drafts_accepted": len(accepted),
                    "drafts_rejected": len(drafts) - len(accepted),
                },
                "validation": _validation_summary(verdicts),
            }
        }
        await db.commit()
    except Exception as exc:
        await db.rollback()
        fresh_run = await db.get(GenerationRun, run_id)
        if fresh_run is None:
            raise
        fresh_run.status = "failed"
        fresh_run.config_json = dict(fresh_run.config_json or {}) | {
            "failure": {"message": str(exc)},
        }
        fresh_run.finished_at = utcnow()
        await db.commit()
        raise


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
