"""Unit tests for the stale voice-session sweep (Phase 4, no DB).

Exercises ``sweep_stale_voice_sessions`` decision logic by patching the
``queries.sessions`` read helpers and ``utcnow`` — no postgres needed. Focus is
the two deadline branches:

* config HAS ``time_limit_minutes`` → ``started_at + limit``.
* config has NO ``time_limit_minutes`` → idle fallback
  (``last_activity`` or ``started_at``) ``+ idle_timeout_minutes`` so an untimed
  session can never stay ``in_progress`` forever.

and the terminal-state choice (``timed_out`` + eval enqueue when there is a
transcript, else ``abandoned``).
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


def _make_session(*, started_at: datetime):
    """Minimal stand-in for an InterviewSession ORM row the sweep mutates."""
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        started_at=started_at,
        status="in_progress",
        ended_at=None,
    )


def _patch_queries(*, candidates, last_activity=None, user_turns=0):
    """Patch the three query helpers the sweep calls on the queries module."""
    return patch.multiple(
        f"{_LIFECYCLE}.sessions_queries",
        list_in_progress_voice_sessions_with_limit=AsyncMock(return_value=candidates),
        get_last_activity_at=AsyncMock(return_value=last_activity),
        count_user_messages=AsyncMock(return_value=user_turns),
    )


@pytest.mark.asyncio
async def test_time_limit_breached_with_transcript_times_out_and_enqueues():
    """Config time-limit passed + >=1 user turn → timed_out + eval enqueued."""
    session = _make_session(started_at=_NOW - timedelta(minutes=40))
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)], user_turns=2),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(db, arq_pool=arq)

    assert count == 1
    assert session.status == "timed_out"
    assert session.ended_at == _NOW
    arq.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_time_limit_idle_breached_with_no_transcript_is_abandoned():
    """NEW PATH: untimed session silent past idle window + 0 turns → abandoned."""
    # No messages → idle anchor falls back to started_at (now - 31m); window 30m.
    session = _make_session(started_at=_NOW - timedelta(minutes=31))
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, None)], last_activity=None, user_turns=0),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(
            db, arq_pool=arq, idle_timeout_minutes=30
        )

    assert count == 1
    assert session.status == "abandoned"
    assert session.ended_at == _NOW
    arq.enqueue_job.assert_not_awaited()  # nothing to evaluate


@pytest.mark.asyncio
async def test_no_time_limit_idle_breached_with_transcript_times_out():
    """NEW PATH: untimed session idle past window but with a turn → timed_out + eval."""
    session = _make_session(started_at=_NOW - timedelta(hours=2))
    last_activity = _NOW - timedelta(minutes=31)  # silent 31m > 30m window
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, None)], last_activity=last_activity, user_turns=1),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(
            db, arq_pool=arq, idle_timeout_minutes=30
        )

    assert count == 1
    assert session.status == "timed_out"
    arq.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_time_limit_recently_active_is_not_swept():
    """NEW PATH guard: untimed session active within idle window stays in_progress."""
    session = _make_session(started_at=_NOW - timedelta(hours=5))
    last_activity = _NOW - timedelta(minutes=10)  # 10m < 30m window → still live
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, None)], last_activity=last_activity),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(
            db, arq_pool=arq, idle_timeout_minutes=30
        )

    assert count == 0
    assert session.status == "in_progress"  # untouched
    assert session.ended_at is None
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_time_limit_not_yet_breached_is_not_swept():
    """Timed session younger than its limit is left in_progress."""
    session = _make_session(started_at=_NOW - timedelta(minutes=15))
    db = AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        _patch_queries(candidates=[(session, 30)]),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(db, arq_pool=None)

    assert count == 0
    assert session.status == "in_progress"


@pytest.mark.asyncio
async def test_returns_count_across_mixed_candidates():
    """Sweep returns the number actually finalised, skipping live ones."""
    stale_timed = _make_session(started_at=_NOW - timedelta(minutes=40))
    stale_idle = _make_session(started_at=_NOW - timedelta(minutes=45))
    fresh_idle = _make_session(started_at=_NOW - timedelta(minutes=45))
    candidates = [(stale_timed, 30), (stale_idle, None), (fresh_idle, None)]
    # Per-session async returns, in candidate order.
    last_activity_by_call = [_NOW - timedelta(minutes=40), _NOW - timedelta(minutes=5)]
    db, arq = AsyncMock(), AsyncMock()
    with (
        patch(f"{_LIFECYCLE}.utcnow", return_value=_NOW),
        patch.multiple(
            f"{_LIFECYCLE}.sessions_queries",
            list_in_progress_voice_sessions_with_limit=AsyncMock(return_value=candidates),
            get_last_activity_at=AsyncMock(side_effect=last_activity_by_call),
            count_user_messages=AsyncMock(return_value=1),
        ),
    ):
        count = await lifecycle_service.sweep_stale_voice_sessions(
            db, arq_pool=arq, idle_timeout_minutes=30
        )

    assert count == 2  # timed + stale-idle finalised; fresh-idle skipped
    assert stale_timed.status == "timed_out"
    assert stale_idle.status == "timed_out"
    assert fresh_idle.status == "in_progress"
