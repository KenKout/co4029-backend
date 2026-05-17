"""ARQ task wrapping the interview evaluation + gap-report pipeline (T6.13).

Convention (Phase 0.8 / plan §5107-5108):

* Signature is ``async def task(ctx, actor_id: UUID, ...)`` —
  ``actor_id`` is the FIRST argument after ``ctx``.
  ``set_worker_actor`` installs it into the audit context so the
  ``GapReport`` and ``InterviewOutcomeEvaluation`` rows written by
  the service get ``created_by`` populated via the SQLAlchemy
  ``before_flush`` listener.
* Structured-log context is bound via ``bind_request_context`` and
  torn down in ``finally`` so neighbouring tasks in the worker pool
  never see leaked state.

Mirrors the discipline of
:mod:`abridgeai.features.quizzes.workers.generation` and
:mod:`abridgeai.features.materials.workers.ingest`. The wrapper is
deliberately thin — evaluation error handling (rollback, stamp
``internal_summary_json['evaluation_failure']``, commit) lives inside
:func:`abridgeai.features.interviews.services.evaluation.evaluate_and_generate_report`
which the worker re-raises so ARQ records the failure for retry.
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
from abridgeai.features.interviews.services import evaluation as evaluation_service
from abridgeai.workers.actor import set_worker_actor

_logger = get_logger(__name__)


async def evaluate_interview_session_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    session_id: UUID,
) -> None:
    """ARQ task: run T6.8 evaluation + T6.9 gap-report for one session.

    Parameters
    ----------
    ctx
        ARQ task context (unused here; reserved for ARQ internals).
    actor_id
        UUID of the user (or system actor) that submitted the session;
        propagated to audit columns via ``set_worker_actor``. The
        ``GapReport`` row inherits this as ``created_by``.
    session_id
        FK into ``interview_sessions``. The T6.11 service composes
        evaluation + gap-report stages and persists outcomes,
        ``internal_summary_json``, and the ``GapReport`` row in a
        single transaction.
    """
    _ = ctx
    set_worker_actor(actor_id)
    bind_request_context(
        session_id=str(session_id),
        actor_id=str(actor_id),
    )
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                await evaluation_service.evaluate_and_generate_report(db, session_id)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _logger.exception(
                    "interview_evaluation_task_failed",
                    session_id=str(session_id),
                )
                raise
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = ["evaluate_interview_session_task"]
