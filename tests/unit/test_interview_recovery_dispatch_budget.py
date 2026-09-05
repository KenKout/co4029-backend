"""A transport failure must not spend a student's recovery budget.

``recover_stalled_evaluations`` stamps the attempt counter and commits BEFORE
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
* one bad candidate does not abort the rest of the sweep.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

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
        started_at=_NOW - timedelta(hours=1),
        assessment_started_at=_NOW - timedelta(hours=1),
        ended_at=_NOW - timedelta(minutes=30),
        internal_summary_json=summary,
    )


def _attempts(session) -> int:
    return int(
        (session.internal_summary_json.get("evaluation_recovery") or {}).get("attempts") or 0
    )


def _patch_candidates(candidates):
    return patch.object(
        lifecycle_service.sessions_queries,
        "list_pending_evaluation_sessions",
        AsyncMock(return_value=candidates),
    )


@pytest.mark.asyncio
async def test_a_failed_enqueue_does_not_consume_a_recovery_attempt():
    """Redis was down. No job exists, so no attempt was made."""
    session = _pending_session()
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.side_effect = ConnectionError("redis unreachable")

    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([session]):
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

    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([session]):
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

    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([session]):
        for _ in range(3):
            await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert _attempts(session) == 0

    # Redis recovers: the session must still be recoverable.
    arq.enqueue_job.side_effect = None
    arq.enqueue_job.return_value = object()
    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([session]):
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

    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([broken, healthy]):
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

    with patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW), _patch_candidates([session]):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 1
    assert _attempts(session) == 2
    recovery = session.internal_summary_json["evaluation_recovery"]
    assert recovery["last_attempt_at"] == _NOW.isoformat()
    db.commit.assert_awaited()
