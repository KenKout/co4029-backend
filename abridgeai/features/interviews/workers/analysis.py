"""ARQ task: full answer analysis + coverage reconciliation for one turn.

Thin wrapper in the manner of :mod:`interviews.workers.evaluation` — the actual
work (full extraction, delta reconciliation, optimistic-lock retry, transaction
boundary) lives in
:mod:`abridgeai.features.interviews.services.turn_reconciliation`.

Registered with the default single try. The evaluation task retries because a
student is blocked on a verdict; nothing is blocked on this one. The turn already
has provisional coverage from the fast probe, the reconciliation is an accuracy
improvement to runtime question selection, and the final grade comes from the
post-session evaluator re-judging the transcript independently — so a lost
reconciliation costs some selection accuracy and never a grade.
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
from abridgeai.features.interviews.services import turn_reconciliation as reconciliation_service
from abridgeai.workers.actor import set_worker_actor

_logger = get_logger(__name__)

RECONCILE_TURN_ANALYSIS_TASK = "reconcile_turn_analysis_task"


async def reconcile_turn_analysis_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    session_id: UUID,
    turn: dict[str, Any],
) -> None:
    """Reconcile one turn's provisional coverage against the full analysis.

    ``turn`` carries the turn's coordinates as primitives — ``message_id``,
    ``question_id``, ``turn_id``, and the ``probe_verdict`` the synchronous path
    produced — and is parsed into a typed
    :class:`~interviews.services.turn_reconciliation.TurnAnalysisRequest` at this
    boundary. Keeping the wire shape a plain dict means an in-flight job enqueued
    by an older deploy stays decodable.

    ``ctx`` is unused: there is no ``job_try`` to branch on because this task is
    registered with a single try (see the module docstring).

    Never raises for a malformed payload — a job that cannot be parsed cannot be
    fixed by retrying it, and a crash here would only fill the worker log.
    """
    del ctx
    set_worker_actor(actor_id)
    bind_request_context(session_id=str(session_id), actor_id=str(actor_id))
    try:
        request = reconciliation_service.TurnAnalysisRequest.from_payload(session_id, turn)
    except (KeyError, TypeError, ValueError):
        _logger.exception(
            "interview_turn_reconcile_bad_payload",
            session_id=str(session_id),
        )
        current_actor_var.set(None)
        clear_request_context()
        return

    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            await reconciliation_service.reconcile_turn_analysis(db, request)
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = ["RECONCILE_TURN_ANALYSIS_TASK", "reconcile_turn_analysis_task"]
