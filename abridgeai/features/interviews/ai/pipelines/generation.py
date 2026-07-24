"""Interview generation pipeline orchestrator (T6.10).

Composes the four T6.4-T6.6 stages and persists accepted drafts:

    retrieval -> generation (with backfill) -> validation -> persistence

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

The generate+validate backfill loop (compensating for validation
dropping drafts short of the requested count) lives in :mod:`.backfill`
to keep this orchestrator focused on run-state bookkeeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.pipelines.backfill import (
    generate_with_backfill,
    validation_summary,
)
from abridgeai.features.interviews.ai.stages.generation import resolve_question_count
from abridgeai.features.interviews.ai.stages.retrieval import retrieve_interview_context
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


@dataclass
class _RunState:
    id: UUID
    course_id: UUID | None
    config_json: dict[str, Any]


async def run_interview_generation(
    db: AsyncSession,
    generation_run_id: UUID,
) -> None:
    """Drive retrieval → generation (with backfill) → validation → persistence.

    On success: marks ``run.status='completed'``, populates
    ``run.config_json`` with stage-level summaries, and inserts one
    :class:`InterviewQuestion` row per accepted draft (monotonically
    increasing ``position``, ``review_status='pending'``).

    On any pipeline-stage failure: rolls back, re-fetches the run, stamps
    ``status='failed'`` + the failure message, commits, and re-raises so
    the ARQ retry budget engages.
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
        # Optional teacher-selected subset (§interview outcome targeting): when
        # config_json carries a non-empty ``target_outcome_ids``, narrow the
        # outcomes fed to the ideation stage to just those. Empty / absent =
        # use every outcome (prior behaviour). Unknown ids are ignored; if the
        # filter would empty the list we fall back to all outcomes so a stale
        # selection never produces a zero-outcome run.
        raw_target_ids = state.config_json.get("target_outcome_ids") or []
        target_id_set = {str(x) for x in raw_target_ids}
        if target_id_set:
            filtered = [o for o in outcomes if str(o.id) in target_id_set]
            if filtered:
                outcomes = filtered
        target_count = resolve_question_count(
            run_config_json=state.config_json,
            supplementary=config.supplementary_instructions,
        )

        # Seed 0/N so the UI shows progress the moment the run goes running.
        await _write_progress(db, state, phase="generating", accepted=0, target=target_count)

        async def _on_progress(accepted_so_far: int, target: int) -> None:
            await _write_progress(
                db, state, phase="generating", accepted=accepted_so_far, target=target
            )

        all_drafts, all_verdicts, accepted, backfill_rounds = await generate_with_backfill(
            db,
            state=state,
            config=config,
            context=context,
            outcomes=outcomes,
            target_count=target_count,
            on_progress=_on_progress,
        )

        await _write_progress(
            db, state, phase="saving", accepted=len(accepted), target=target_count
        )
        await _persist_questions(
            db,
            config=config,
            accepted=accepted,
            source_module_ids=_module_ids_for_questions(state.config_json, config),
        )

        state.config_json = state.config_json | {
            "pipeline": {
                "stage": "completed",
                "pipeline_run_id": str(state.id),
                "generation": {
                    "question_count_requested": target_count,
                    "questions_persisted": len(accepted),
                    "drafts_total": len(all_drafts),
                    "drafts_accepted": len(accepted),
                    "drafts_rejected": len(all_drafts) - len(accepted),
                    "backfill_rounds": backfill_rounds,
                },
                "validation": validation_summary(all_verdicts),
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


async def _write_progress(
    db: AsyncSession,
    state: _RunState,
    *,
    phase: str,
    accepted: int,
    target: int,
) -> None:
    """Persist + commit live generation progress into ``config_json``.

    The teacher SPA polls ``generation_runs.config_json`` every 2.5s while
    the run is ``running`` and renders ``progress.accepted / progress.target``
    as a count + percentage bar.
    """
    state.config_json = state.config_json | {
        "progress": {"phase": phase, "accepted": accepted, "target": target}
    }
    await _update_run(db, state.id, config_json=state.config_json)
    await db.commit()


def _retrieval_summary(context: InterviewRetrievalContext) -> dict[str, Any]:
    return {
        "count": len(context.chunks),
        "kg_concept_count": len(context.kg_concepts),
        "weak_topic_count": len(context.weak_topic_chunks),
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


def _module_ids_for_questions(config_json: dict[str, Any], config: InterviewConfig) -> list[str]:
    """Module attribution for generated questions.

    Prefers the run's ``source_module_ids`` (the modules the teacher scoped
    generation to). Falls back to the interview config's own module so a
    question is never left unattributed.
    """
    raw = config_json.get("source_module_ids") or []
    ids = [str(m) for m in raw if m]
    if ids:
        return ids
    return [str(config.module_id)] if config.module_id is not None else []


async def _persist_questions(
    db: AsyncSession,
    *,
    config: InterviewConfig,
    accepted: list[InterviewQuestionDraft],
    source_module_ids: list[str],
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
                model_answer=draft.model_answer.strip() or None,
                review_status="pending",
                ai_generated=True,
                source_refs_json=[str(c) for c in draft.source_refs],
                source_module_ids=source_module_ids,
            )
        )
        await db.flush()


__all__ = ["run_interview_generation"]
