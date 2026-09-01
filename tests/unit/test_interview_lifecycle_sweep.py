"""Unit tests for the configured-deadline interview-session sweep (Phase 4).

Exercises ``sweep_expired_interview_sessions`` decision logic by patching the
``queries.sessions`` read helpers and ``utcnow`` — no postgres needed. Untimed
sessions never enter the candidate query; only a configured ``time_limit_minutes``
may expire an active assessment.
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
_DEFAULT_ASSESSMENT_START = object()


def _make_session(
    *,
    started_at: datetime,
    assessment_started_at: datetime | None | object = _DEFAULT_ASSESSMENT_START,
):
    """Minimal stand-in for an InterviewSession ORM row the sweep mutates."""
    if assessment_started_at is _DEFAULT_ASSESSMENT_START:
        assessment_started_at = started_at
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        started_at=started_at,
        assessment_started_at=assessment_started_at,
        status="in_progress",
        ended_at=None,
    )


def _patch_queries(*, candidates, user_turns=0, claimed=True, finalize=None):
    """Patch the atomic lifecycle transition used by the deadline sweep."""
    sessions_by_id = {session.id: session for session, _limit in candidates}

    async def _finalize(db, session_id, *, ended_at):  # noqa: ANN001 - mock helper
        if not claimed:
            return None
        session = sessions_by_id[session_id]
        session.status = "timed_out" if user_turns >= 1 else "abandoned"
        session.ended_at = ended_at
        return session.status

    finalize_mock = finalize or AsyncMock(side_effect=_finalize)
    return patch.multiple(
        f"{_LIFECYCLE}.sessions_queries",
        list_in_progress_sessions_with_time_limit=AsyncMock(return_value=candidates),
        finalize_expired_in_progress_session=finalize_mock,
    )


@pytest.mark.asyncio
async def test_time_limit_breached_with_transcript_times_out_and_enqueues():
    """Configured deadline passed + >=1 user turn → timed_out + eval enqueue."""
    session = _make_session(started_at=_NOW - timedelta(minutes=40))
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)], user_turns=2),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=arq)

    assert count == 1
    assert session.status == "timed_out"
    assert session.ended_at == _NOW
    arq.enqueue_job.assert_awaited_once()
    assert arq.enqueue_job.await_args.kwargs["_job_id"] == (f"interview-evaluation:{session.id}")


@pytest.mark.asyncio
async def test_time_limit_breached_without_transcript_is_abandoned():
    """Configured deadline passed + no user turn → abandoned without evaluation."""
    session = _make_session(started_at=_NOW - timedelta(minutes=40))
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)], user_turns=0),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=arq)

    assert count == 1
    assert session.status == "abandoned"
    assert session.ended_at == _NOW
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_recover_stalled_evaluation_uses_deduplicated_job_id():
    session = _make_session(started_at=_NOW - timedelta(hours=1))
    session.status = "completed"
    session.ended_at = _NOW - timedelta(minutes=30)
    db = AsyncMock()
    arq = AsyncMock()
    arq.enqueue_job.return_value = object()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        patch.object(
            lifecycle_service.sessions_queries,
            "list_pending_evaluation_sessions",
            AsyncMock(return_value=[session]),
        ) as pending_query,
    ):
        count = await lifecycle_service.recover_stalled_evaluations(db, arq_pool=arq)

    assert count == 1
    assert pending_query.await_args.kwargs["ended_before"] == _NOW - timedelta(minutes=15)
    arq.enqueue_job.assert_awaited_once_with(
        "evaluate_interview_session_task",
        session.student_id,
        session.id,
        _job_id=f"interview-evaluation:{session.id}",
    )


@pytest.mark.asyncio
async def test_recover_stalled_evaluation_skips_without_queue():
    count = await lifecycle_service.recover_stalled_evaluations(
        AsyncMock(),
        arq_pool=None,
    )

    assert count == 0


@pytest.mark.asyncio
async def test_untimed_stale_session_is_never_swept():
    """Untimed sessions, however old, do not reach the deadline sweep."""
    stale_untimed = _make_session(started_at=_NOW - timedelta(hours=2))
    db, arq = AsyncMock(), AsyncMock()
    finalize_mock = AsyncMock(return_value=None)
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[], finalize=finalize_mock),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=arq)

    assert count == 0
    assert stale_untimed.status == "in_progress"
    assert stale_untimed.ended_at is None
    finalize_mock.assert_not_awaited()
    db.commit.assert_not_awaited()
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_submission_prevents_sweep_overwrite():
    """A stale sweep snapshot must not overwrite a concurrent completion."""
    session = _make_session(started_at=_NOW - timedelta(minutes=40))
    db, arq = AsyncMock(), AsyncMock()
    finalize_mock = AsyncMock(return_value=None)  # claimed=False → no transition
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)], user_turns=1, finalize=finalize_mock),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=arq)

    assert count == 0
    assert session.status == "in_progress"
    finalize_mock.assert_awaited_once_with(
        db,
        session.id,
        ended_at=_NOW,
    )
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_time_limit_not_yet_breached_is_not_swept():
    """Timed session younger than its limit is left in_progress."""
    session = _make_session(started_at=_NOW - timedelta(minutes=15))
    db = AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)]),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=None)

    assert count == 0
    assert session.status == "in_progress"
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_in_onboarding_does_not_expire_before_assessment_starts():
    """A timed session has no deadline until onboarding completes."""
    session = _make_session(
        started_at=_NOW - timedelta(hours=2),
        assessment_started_at=None,
    )
    db = AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)]),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db)

    assert count == 0
    assert session.status == "in_progress"
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_count_across_timed_candidates():
    """Sweep finalises expired timed candidates and leaves fresh ones live."""
    stale_timed = _make_session(started_at=_NOW - timedelta(minutes=40))
    fresh_timed = _make_session(started_at=_NOW - timedelta(minutes=5))
    candidates = [(stale_timed, 30), (fresh_timed, 30)]
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=candidates, user_turns=1),
    ):
        count = await lifecycle_service.sweep_expired_interview_sessions(db, arq_pool=arq)

    assert count == 1
    assert stale_timed.status == "timed_out"
    assert fresh_timed.status == "in_progress"
    arq.enqueue_job.assert_awaited_once()
