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

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.pipelines.backfill import (
    generate_with_backfill,
    validation_summary,
)
from abridgeai.features.interviews.ai.pipelines.persistence import (
    _module_ids_for_questions,
    _persist_questions,
)
from abridgeai.features.interviews.ai.pipelines.variant import (
    config_uuid,
    resolve_variant_mode,
)
from abridgeai.features.interviews.ai.stages.generation import resolve_question_count
from abridgeai.features.interviews.ai.stages.generation.resolve import resolve_supplementary
from abridgeai.features.interviews.ai.stages.retrieval import retrieve_interview_context
from abridgeai.features.interviews.models import InterviewConfig
from abridgeai.features.interviews.queries.authoring import list_outcomes_for_config
from abridgeai.features.quizzes.api import public as quizzes_public

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.retrieval.logic import (
        InterviewRetrievalContext,
    )


logger = get_logger(__name__)


@dataclass
class _RunState:
    id: UUID
    course_id: UUID | None
    config_json: dict[str, Any]
    requested_by: UUID | None = None


async def run_interview_generation(  # noqa: C901 -- pipeline stages stay auditable in one flow
    db: AsyncSession,
    generation_run_id: UUID,
    *,
    arq_pool: object | None = None,
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
    # Lazy import to avoid a circular import at module load: this pipeline is
    # imported by ``services/__init__`` (via services.generation), and
    # completion_notify lives under services — importing it at top level here
    # would form services → pipeline → services during package init.
    from abridgeai.features.interviews.services.completion_notify import (  # noqa: PLC0415
        notify_interview_generation_outcome,
    )

    run_dto = await quizzes_public.get_generation_run(db, generation_run_id)
    if run_dto is None:
        raise NotFoundError("Generation run not found")
    state = _RunState(
        id=run_dto.id,
        course_id=run_dto.course_id,
        config_json=dict(run_dto.config_json or {}),
        requested_by=run_dto.requested_by,
    )
    config: InterviewConfig | None = None

    try:
        if not await _claim_pending_run(db, state.id):
            return
        await db.commit()

        config_id = config_uuid(state.config_json, "interview_config_id")
        if config_id is None:
            raise NotFoundError("Generation run is missing interview_config_id")
        config = await db.get(InterviewConfig, config_id)
        if config is None:
            raise NotFoundError("Interview config not found for generation run")

        context = await retrieve_interview_context(
            db,
            run=state,
            config=config,
            pipeline_run_id=state.id,
        )
        logger.info(
            "interview_retrieval_complete",
            chunk_count=len(context.chunks),
            kg_concept_count=len(context.kg_concepts),
            weak_topic_count=len(context.weak_topic_chunks),
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
        # use every outcome (prior behaviour). The authoring service validates
        # explicit IDs before enqueueing, so an empty filtered set is a failed
        # invariant, never a reason to silently broaden generation scope.
        raw_target_ids = state.config_json.get("target_outcome_ids") or []
        target_id_set = {str(x) for x in raw_target_ids}
        if target_id_set:
            outcomes = [o for o in outcomes if str(o.id) in target_id_set]
            if not outcomes:
                raise RuntimeError("targeted generation resolved no interview outcomes")
        target_count = resolve_question_count(
            run_config_json=state.config_json,
            # Same resolved value the generation stage uses, so the pipeline's
            # target and the stage's request cannot disagree when the run
            # overrides supplementary_instructions.
            supplementary=resolve_supplementary(
                state.config_json, config.supplementary_instructions
            ),
        )
        variant_strategy, role_type, target_count = resolve_variant_mode(
            config, state.config_json, target_count
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
            variant_strategy=variant_strategy,
            role_type=role_type,
            on_progress=_on_progress,
        )

        if len(accepted) != target_count:
            raise RuntimeError(
                f"Generation underfilled: accepted {len(accepted)} of {target_count} questions"
            )

        await _write_progress(
            db, state, phase="saving", accepted=len(accepted), target=target_count
        )
        await db.refresh(config)
        # Local import: the services package imports this pipeline (ARQ entry),
        # so a module-level ``services.*`` import here is a circular-import trap
        # for any direct importer (e2e tests import the pipeline first).
        from abridgeai.features.interviews.services.published_freeze import (  # noqa: PLC0415
            assert_questions_editable,
        )

        assert_questions_editable(config)
        await _persist_questions(
            db,
            config=config,
            accepted=accepted,
            source_module_ids=_module_ids_for_questions(state.config_json, config),
            pipeline_run_id=state.id,
        )
        strategy = state.config_json.get("variant_strategy")
        if strategy == "role_only":
            config.generation_variant_strategy = "role_only"

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
        # Notify the initiating teacher (best-effort; rides its own commit and
        # never disturbs the generation transaction above).
        await notify_interview_generation_outcome(
            db,
            recipient_user_id=state.requested_by,
            course_id=state.course_id,
            config_id=config.id,
            interview_title=config.title,
            succeeded=True,
            arq_pool=arq_pool,
        )
        await db.commit()
    except asyncio.CancelledError:
        # CancelledError (BaseException) bypasses `except Exception`; roll back
        # so a cancelled run doesn't hold its row lock "idle in transaction".
        await db.rollback()
        raise
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
        # Notify the initiating teacher of the failure (best-effort).
        await notify_interview_generation_outcome(
            db,
            recipient_user_id=state.requested_by,
            course_id=state.course_id,
            config_id=config.id if config is not None else None,
            interview_title=config.title if config is not None else None,
            succeeded=False,
            error_message=str(exc),
            arq_pool=arq_pool,
        )
        await db.commit()
        raise


async def _claim_pending_run(db: AsyncSession, run_id: UUID) -> bool:
    """Atomically claim a queued run; duplicate worker deliveries no-op."""
    result = await db.execute(
        text(
            "UPDATE generation_runs SET status = 'running', started_at = :started_at, "
            "updated_at = NOW() WHERE id = :id AND status = 'pending'"
        ),
        {"id": run_id, "started_at": utcnow()},
    )
    return result.rowcount == 1


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
        # MERGE, never replace. The orphan reaper
        # (``features.materials.workers.reaper._run_reconcile_interview``)
        # writes its durable re-enqueue budget into the SAME column as
        # ``config_json["reap_count"]``. A full replace from this pipeline's
        # in-memory ``state.config_json`` snapshot — which is loaded once at
        # run start and re-committed by every ``_write_progress`` tick — would
        # silently erase that counter, resetting the reaper's budget to 0 and
        # letting a genuinely unrecoverable run be re-enqueued forever.
        # ``||`` is a shallow merge: the keys this pipeline owns (progress,
        # retrieval, pipeline, failure) win, and any key it does not know about
        # survives.
        sets.append(
            "config_json = COALESCE(config_json, '{}'::jsonb) || CAST(:config_json AS jsonb)"
        )
        params["config_json"] = json.dumps(config_json, default=str)
    if not sets:
        return
    sets.append("updated_at = NOW()")
    sql = f"UPDATE generation_runs SET {', '.join(sets)} WHERE id = :id"  # noqa: S608  # sets list is code-controlled, no user input
    await db.execute(text(sql), params)


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
        "source_chunk_ids": [str(chunk.chunk_id) for chunk in context.chunks],
        "kg_concept_count": len(context.kg_concepts),
        "weak_topic_count": len(context.weak_topic_chunks),
    }


__all__ = ["run_interview_generation"]
