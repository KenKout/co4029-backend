"""A transport failure must not spend a student's recovery budget.

``recover_stalled_evaluations`` charges the attempt counter and commits BEFORE
enqueueing, deliberately: a task that dies hard (OOM, worker kill) has to
consume its budget or the sweep re-queues it every five minutes forever.

That reasoning only holds when the job actually reached Redis. When the enqueue
itself fails — Redis down, connection reset — no job was ever created, yet the
budget was already spent. Three such sweeps exhaust ``max_recovery_attempts``,
the SQL-side ceiling drops the row from the candidate query, and the session is
stranded ungraded forever with the answers still sitting in the database.

The raised exception has a second victim: it propagates out of the per-candidate
loop, so every candidate behind the failing one in that sweep is skipped too.

Pinned here:

* an enqueue that raises does not consume an attempt;
* an enqueue that returns ``None`` (ARQ refused the job ID — nothing queued)
  does not consume an attempt either;
* one bad candidate does not abort the rest of the sweep;
* a verdict published between the candidate query and the charge cancels the
  re-drive instead of costing an attempt.

The counter itself now lives in SQL (``queries.sessions``) precisely so it cannot
clobber a concurrent verdict; the fakes below emulate those two helpers' return
contract so the service-level budget rules stay pinned here, while
``tests/integration/test_interview_recovery_metadata_isolation.py`` pins the SQL.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from abridgeai.features.interviews.services import lifecycle as lifecycle_service

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_LIFECYCLE = "abridgeai.features.interviews.services.lifecycle"


def _pending_session(*, attempts: int = 0):
    summary: dict[str, object] = {}
    if attempts:
        summary["evaluation_recovery"] = {"attempts": attempts}
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        status="failed",
        pass_verdict=None,
        started_at=_NOW - timedelta(hours=1),
        assessment_started_at=_NOW - timedelta(hours=1),
        ended_at=_NOW - timedelta(minutes=30),
        internal_summary_json=summary,
    )


class _RecoveryCounterStore:
    """Stand-in for the two SQL helpers, with their real return contract.

    Keyed by session id and backed by the row's own dict so the assertions below
    read the same place the service does. Mirrors the SQL semantics that matter:

    * charging is refused (``None``) once a verdict exists;
    * charging returns the NEW attempt count;
    * refunding only applies while our count is still the current one, and
      touches ONLY the ``evaluation_recovery`` sub-object.
    """

    def __init__(self, *rows: Any) -> None:
        self._rows = {row.id: row for row in rows}

    def _recovery(self, row: Any) -> dict[str, Any]:
        value = row.internal_summary_json.get("evaluation_recovery")
        return value if isinstance(value, dict) else {}

    async def stamp(self, _db: Any, session_id: UUID, *, now: datetime) -> int | None:
        row = self._rows[session_id]
        if row.pass_verdict is not None:
            return None
        attempts = int(self._recovery(row).get("attempts") or 0) + 1
        row.internal_summary_json["evaluation_recovery"] = {
            **self._recovery(row),
            "attempts": attempts,
            "last_attempt_at": now.isoformat(),
        }
        return attempts

    async def refund(
        self,
        _db: Any,
        session_id: UUID,
        *,
        attempt: int,
        previous_recovery: dict[str, Any] | None,
    ) -> bool:
        row = self._rows[session_id]
        if int(self._recovery(row).get("attempts") or 0) != attempt:
            return False
        if previous_recovery is None:
            row.internal_summary_json.pop("evaluation_recovery", None)
        else:
            row.internal_summary_json["evaluation_recovery"] = dict(previous_recovery)
        return True


def _attempts(session) -> int:
    return int(
        (session.internal_summary_json.get("evaluation_recovery") or {}).get("attempts") or 0
    )


@contextlib.contextmanager
def _sweep(*rows: Any):
    """Patch the clock, the candidate query, and both counter helpers."""
    store = _RecoveryCounterStore(*rows)
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        patch.object(
            lifecycle_service.sessions_queries,
            "list_pending_evaluation_sessions",
            AsyncMock(return_value=list(rows)),
        ),
        patch.object(
            lifecycle_service.sessions_queries,
            "stamp_evaluation_recovery_attempt",
            store.stamp,
        ),
        patch.object(
            lifecycle_service.sessions_queries,
            "refund_evaluation_recovery_attempt",
            store.refund,
        ),
    ):
        yield store


@pytest.mark.asyncio
async def test_a_failed_enqueue_does_not_consume_a_recovery_attempt():
    """Redis was down. No job exists, so no attempt was made."""
    session = _pending_session()
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.side_effect = ConnectionError("redis unreachable")

    with _sweep(session):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 0
    assert _attempts(session) == 0, (
        "a transport failure spent a grading opportunity the student never got"
    )


@pytest.mark.asyncio
async def test_a_refused_enqueue_does_not_consume_a_recovery_attempt():
    """``enqueue_job`` returning None means ARQ queued nothing at all."""
    session = _pending_session()
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.return_value = None

    with _sweep(session):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 0
    assert _attempts(session) == 0, "None is not evidence that a job was enqueued"


@pytest.mark.asyncio
async def test_repeated_dispatch_outage_leaves_the_budget_intact():
    """Three failed sweeps must not exhaust a 3-attempt budget."""
    session = _pending_session()
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.side_effect = ConnectionError("redis unreachable")

    for _ in range(3):
        with _sweep(session):
            await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert _attempts(session) == 0

    # Redis recovers: the session must still be recoverable.
    arq.enqueue_job.side_effect = None
    arq.enqueue_job.return_value = object()
    with _sweep(session):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 1
    assert _attempts(session) == 1


@pytest.mark.asyncio
async def test_one_failing_candidate_does_not_strand_the_rest_of_the_sweep():
    """A per-candidate enqueue error is contained, not fatal to the loop."""
    broken = _pending_session()
    healthy = _pending_session()
    db = AsyncMock()
    arq = AsyncMock()

    async def _enqueue(_task, _student_id, session_id, *, _job_id):  # noqa: ANN001, ANN202
        if session_id == broken.id:
            raise ConnectionError("redis unreachable")
        return object()

    arq.enqueue_job.side_effect = _enqueue

    with _sweep(broken, healthy):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 1, "the healthy candidate behind the failure was skipped"
    assert _attempts(broken) == 0
    assert _attempts(healthy) == 1


@pytest.mark.asyncio
async def test_a_successful_enqueue_still_spends_its_attempt_up_front():
    """The original invariant survives: a queued job has consumed its budget."""
    session = _pending_session(attempts=1)
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.return_value = object()

    with _sweep(session):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 1
    assert _attempts(session) == 2
    recovery = session.internal_summary_json["evaluation_recovery"]
    assert recovery["last_attempt_at"] == _NOW.isoformat()


@pytest.mark.asyncio
async def test_a_verdict_published_before_the_charge_cancels_the_redrive():
    """The evaluator won the race. Do not re-queue, and do not bill the student.

    The candidate query ran minutes earlier (the grace window), so by the time
    the sweep reaches a row the evaluation it is trying to repair may have
    finished. Charging + enqueueing here would spend an attempt on a graded
    session and start a second grader against a published verdict.
    """
    session = _pending_session()
    session.pass_verdict = True  # published between the query and the charge
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.return_value = object()

    with _sweep(session):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 0
    assert _attempts(session) == 0
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_refund_only_restores_the_recovery_bookkeeping():
    """A refund must not reach anything the evaluator owns.

    The old refund wrote back a whole pre-enqueue snapshot of
    ``internal_summary_json``. When a verdict landed while the enqueue was in
    flight, that snapshot deleted the rubric totals the evaluator had just
    committed — and resurrected the stale ``evaluation_failure`` note it cleared.
    """
    session = _pending_session()
    session.internal_summary_json = {"evaluation_recovery": {"attempts": 1}}
    db = AsyncMock()
    arq = AsyncMock()

    async def _enqueue_after_a_verdict_lands(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        # The evaluator publishes while our enqueue is in flight.
        session.internal_summary_json["total_score"] = 82.5
        session.internal_summary_json["evaluated_at"] = "2026-01-01T12:00:30+00:00"
        raise ConnectionError("redis unreachable")

    arq.enqueue_job.side_effect = _enqueue_after_a_verdict_lands

    with _sweep(session):
        await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert _attempts(session) == 1, "the refund should restore the pre-charge count"
    assert session.internal_summary_json["total_score"] == 82.5, (
        "the refund clobbered results written by the evaluator"
    )
    assert session.internal_summary_json["evaluated_at"] == "2026-01-01T12:00:30+00:00"
