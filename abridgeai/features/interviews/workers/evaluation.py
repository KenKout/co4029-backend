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
:func:`abridgeai.features.interviews.services.evaluation.evaluate_and_generate_report`.

Retry mechanism (why this wrapper is NOT a bare re-raise)
---------------------------------------------------------
The AI judge (``evaluate_outcomes`` / ``evaluate_session`` /
``generate_gap_report``) makes external LLM HTTP calls. Those can fail
transiently — provider 5xx, connection reset, a malformed JSON body,
a 429 that outlived the client's inline backoff. When they do, the
student's interview would otherwise be stamped ``failed`` on the FIRST
error and never judged again.

``WorkerSettings.max_tries=3`` alone does NOT fix this: arq 0.28's
worker only re-queues a job when the task raises :class:`arq.worker.Retry`
(or ``CancelledError`` / ``RetryJob``). Any *other* exception —
including our ``ProviderError`` — takes arq's terminal ``else`` branch,
so ``job_try`` never advances past 1 and the "3 tries" budget is never
spent. To actually retry we must translate the failure into an explicit
``Retry`` while budget remains.

Behaviour:

* Attempts ``1 .. max_tries-1`` — on ANY evaluation exception we raise
  :class:`arq.worker.Retry` with exponential backoff so arq re-enqueues
  the job (``job_try`` increments on the next pickup). The service has
  already rolled back and stamped a non-terminal ``evaluation_failure``
  note; a later successful attempt overwrites it.
* Final attempt (``job_try >= max_tries``) — we let the original
  exception propagate. ``is_final_attempt=True`` was passed to the
  service so it stamped the terminal ``status='failed'``; re-raising
  lets arq record the job-level failure. Without the terminal status
  the student-facing poll in ``course-interview.tsx`` would wait
  forever for a ``pass_verdict`` that never arrives.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from arq import Retry

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

# Exponential backoff between evaluation retries. ``job_try`` is 1-indexed,
# so the defer before attempt N+1 is ``base * 2**(job_try-1)`` seconds:
# ~5s before attempt 2, ~10s before attempt 3. Capped so a degraded provider
# is never hammered and the job never sits deferred for minutes.
_RETRY_BASE_DELAY_S = 5.0
_RETRY_MAX_DELAY_S = 60.0
# ARQ includes ``job_try`` in the runtime context but does not expose the
# configured function/worker ``max_tries`` value there. Keep the evaluation
# budget explicit and reuse this constant when registering the ARQ function.
EVALUATION_MAX_TRIES = 3


def _retry_defer_seconds(job_try: int) -> float:
    """Seconds to wait before the next evaluation attempt (exponential, capped)."""
    exponent = max(0, job_try - 1)
    return float(min(_RETRY_MAX_DELAY_S, _RETRY_BASE_DELAY_S * (2**exponent)))


async def evaluate_interview_session_task(
    ctx: dict[str, Any],
    actor_id: UUID,
    session_id: UUID,
) -> None:
    """ARQ task: run T6.8 evaluation + T6.9 gap-report for one session.

    Parameters
    ----------
    ctx
        ARQ task context. Carries ``job_try`` (1-indexed current attempt).
        ARQ does not include ``max_tries`` in this mapping, so the task uses
        :data:`EVALUATION_MAX_TRIES` to detect the FINAL attempt (and stamps a
        terminal ``status='failed'`` instead of leaving the session stuck
        at ``'completed'`` with ``pass_verdict`` forever ``null``) and to
        decide whether a transient AI-call failure should raise
        :class:`arq.worker.Retry` (budget remaining) or propagate (budget
        exhausted).
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
    job_try_raw = ctx.get("job_try")
    job_try = job_try_raw if isinstance(job_try_raw, int) else 1
    is_final_attempt = job_try >= EVALUATION_MAX_TRIES
    set_worker_actor(actor_id)
    bind_request_context(
        session_id=str(session_id),
        actor_id=str(actor_id),
    )
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            try:
                await evaluation_service.evaluate_and_generate_report(
                    db, session_id, is_final_attempt=is_final_attempt
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Retry:
                # Already a retry signal (shouldn't originate in the service,
                # but never swallow it) — let arq handle re-queueing.
                raise
            except Exception as exc:
                _logger.exception(
                    "interview_evaluation_task_failed",
                    session_id=str(session_id),
                    job_try=job_try,
                    max_tries=EVALUATION_MAX_TRIES,
                    is_final_attempt=is_final_attempt,
                )
                if is_final_attempt:
                    # Budget exhausted: the service already stamped
                    # status='failed'. Propagate so arq records the terminal
                    # job failure — no further retry.
                    raise
                # Budget remaining: translate into arq's Retry so the job is
                # re-enqueued (bare re-raise would NOT retry — see module
                # docstring). Backoff grows with each attempt.
                defer = _retry_defer_seconds(job_try)
                _logger.warning(
                    "interview_evaluation_task_retry",
                    session_id=str(session_id),
                    job_try=job_try,
                    max_tries=EVALUATION_MAX_TRIES,
                    defer_seconds=defer,
                    error=str(exc),
                )
                raise Retry(defer=defer) from exc
    finally:
        current_actor_var.set(None)
        clear_request_context()


__all__ = [
    "EVALUATION_MAX_TRIES",
    "evaluate_interview_session_task",
]
