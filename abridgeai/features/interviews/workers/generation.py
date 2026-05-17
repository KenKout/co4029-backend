"""ARQ task wrapping the interview generation pipeline (T6.13).

Convention (Phase 0.8 / plan §5107-5108):

* Signature is ``async def task(ctx, actor_id: UUID, ...)`` —
  ``actor_id`` is the FIRST argument after ``ctx``.
  ``set_worker_actor`` installs it into the audit context so any DB
  writes during the task automatically populate ``created_by`` /
  ``updated_by`` via the SQLAlchemy ``before_flush`` listener.
* Structured-log context is bound via ``bind_request_context`` and
  torn down in ``finally`` so neighbouring tasks in the worker pool
  never see leaked state.

Mirrors the discipline of
:mod:`abridgeai.features.quizzes.workers.generation` and
:mod:`abridgeai.features.materials.workers.ingest`. The wrapper is
deliberately thin — pipeline error handling (rollback, ``status='failed'``,
``finished_at``) lives inside
:func:`abridgeai.features.interviews.services.generation.run_interview_generation`
which the worker re-raises so ARQ records the failure.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from abridgeai.core.audit import current_actor_var
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    get_logger,
)
from abridgeai.features.interviews.services import generation as generation_service
from abridgeai.workers.actor import set_worker_actor

_logger = get_logger(__name__)


async def run_interview_generation_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    generation_run_id: UUID,
) -> None:
    """ARQ task: dispatch one interview ``GenerationRun`` to its pipeline.

    Parameters
    ----------
    ctx
        ARQ task context (unused here; reserved for ARQ internals).
    actor_id
        UUID of the user (or system actor) that initiated the run;
        propagated to audit columns via ``set_worker_actor``.
    generation_run_id
        FK into ``generation_runs``. The T6.10 pipeline orchestrator
        reads the row, marks it ``running``, runs retrieval →
        generation → validation → persistence, then stamps
        ``completed`` / ``failed`` + ``finished_at``.
    """
    _ = ctx
    set_worker_actor(actor_id)
    bind_request_context(
        generation_run_id=str(generation_run_id),
        actor_id=str(actor_id),
    )
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                await generation_service.run_interview_generation(db, generation_run_id)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _logger.exception(
                    "interview_generation_task_failed",
                    generation_run_id=str(generation_run_id),
                )
                raise
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = ["run_interview_generation_task"]
