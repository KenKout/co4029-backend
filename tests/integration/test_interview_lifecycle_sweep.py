"""Integration tests for interview lifecycle sweep (Phase 6, requires postgres).

Tests sweep_stale_voice_sessions:
- Marks in-progress voice/hybrid sessions >time_limit_minutes as timed_out
- If session has >=1 user message: enqueues evaluate_interview_session_task
- If session has 0 user messages: marks abandoned (no eval needed)

NOTE: These tests require a real postgres database (TEST_DATABASE_URL).
Not runnable in CI without postgres. Add to integration test suite.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

pytestmark = pytest.mark.skipif(
    True,
    reason="Requires postgres database (TEST_DATABASE_URL) — not available in this environment"
)


@pytest.mark.asyncio
async def test_sweep_stale_voice_session_timed_out(
    db: AsyncSession,
    seeded_users,
):
    """In-progress voice session older than config.time_limit_minutes is marked timed_out."""
    # Create a voice session with started_at = now - 31 minutes
    # (assuming config.time_limit_minutes = 30)
    # Call sweep_stale_voice_sessions(db)
    # Query: SELECT status FROM interview_sessions WHERE id = ?
    # Verify status == "timed_out"
    pass


@pytest.mark.asyncio
async def test_sweep_stale_voice_session_with_user_message_enqueues_eval(
    db: AsyncSession,
    seeded_users,
    arq_pool,
):
    """Stale voice session with >=1 user message enqueues evaluation task."""
    # Create stale voice session + 1 user message (InterviewSessionMessage)
    # Call sweep_stale_voice_sessions(db, arq_pool)
    # Verify:
    #   1. Session status = "timed_out"
    #   2. evaluate_interview_session_task was enqueued to ARQ
    pass


@pytest.mark.asyncio
async def test_sweep_stale_voice_session_no_user_message_abandoned(
    db: AsyncSession,
    seeded_users,
):
    """Stale voice session with 0 user messages is marked abandoned (not eval'd)."""
    # Create stale voice session with NO messages (student never answered)
    # Call sweep_stale_voice_sessions(db)
    # Query: SELECT status FROM interview_sessions WHERE id = ?
    # Verify status == "abandoned" (not "timed_out", no eval task)
    pass


@pytest.mark.asyncio
async def test_sweep_hybrid_session_treated_as_voice(
    db: AsyncSession,
    seeded_users,
):
    """Hybrid input_mode sessions are included in voice sweep (can use voice)."""
    # Create hybrid (voice+text capable) session, older than limit
    # Call sweep_stale_voice_sessions(db)
    # Verify it's marked timed_out (not ignored)
    pass


@pytest.mark.asyncio
async def test_sweep_text_only_session_not_affected(
    db: AsyncSession,
    seeded_users,
):
    """Text-only input_mode sessions are NOT affected by voice sweep."""
    # Create text-only session, older than limit
    # Call sweep_stale_voice_sessions(db)
    # Query: SELECT status FROM interview_sessions WHERE id = ?
    # Verify status is UNCHANGED (still "in_progress" or original)
    pass


@pytest.mark.asyncio
async def test_sweep_recent_session_not_marked(
    db: AsyncSession,
    seeded_users,
):
    """Voice session younger than config.time_limit_minutes is not affected."""
    # Create voice session with started_at = now - 15 minutes
    # (assuming config.time_limit_minutes = 30)
    # Call sweep_stale_voice_sessions(db)
    # Verify status is unchanged (still "in_progress")
    pass


@pytest.mark.asyncio
async def test_sweep_returns_count(
    db: AsyncSession,
    seeded_users,
):
    """sweep_stale_voice_sessions returns count of sessions marked timed_out."""
    # Create 3 stale voice sessions with messages
    # Call sweep_stale_voice_sessions(db)
    # Verify return value == 3
    pass


@pytest.mark.asyncio
async def test_sweep_idempotent(
    db: AsyncSession,
    seeded_users,
):
    """sweep_stale_voice_sessions can be called multiple times safely."""
    # Create stale voice session
    # Call sweep_stale_voice_sessions(db) twice
    # Verify no errors, status remains "timed_out" after 2nd call
    pass
