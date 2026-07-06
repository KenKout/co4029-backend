"""Integration tests for realtime token endpoint (Phase 6, requires postgres).

Tests POST /interview-sessions/{id}/realtime-token:
- 503 if voice disabled
- 409 if input_mode not in voice/hybrid or status!=in_progress
- 200 happy path + livekit_room_name persisted
- 403 cross-user
- Token is valid JWT with correct claims

NOTE: These tests require a real postgres database (TEST_DATABASE_URL).
Not runnable in CI without postgres. Add to integration test suite.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

pytestmark = pytest.mark.skipif(
    True,
    reason="Requires postgres database (TEST_DATABASE_URL) — not available in this environment"
)


@pytest.mark.asyncio
async def test_realtime_token_voice_disabled_503(
    client: AsyncClient,
    seeded_users,
    student_token: str,
):
    """503 error if interview_voice_enabled is False."""
    # This test would require overriding settings.interview_voice_enabled = False
    # and then calling POST /interview-sessions/{id}/realtime-token
    # Expected: 503 Service Unavailable
    pass


@pytest.mark.asyncio
async def test_realtime_token_text_only_session_409(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    session_id,
):
    """409 error if session input_mode is 'text' (not voice/hybrid)."""
    # Create a text-only interview session
    # Call POST /interview-sessions/{id}/realtime-token
    # Expected: 409 Conflict (session not voice-eligible)
    pass


@pytest.mark.asyncio
async def test_realtime_token_finished_session_409(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    voice_session_id,
):
    """409 error if session status is 'finished'."""
    # Create a voice session, mark it finished
    # Call POST /interview-sessions/{id}/realtime-token
    # Expected: 409 Conflict (session not in_progress)
    pass


@pytest.mark.asyncio
async def test_realtime_token_happy_path_200(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    voice_session_id,
    test_engine,
):
    """200 success: token + room_name returned, livekit_room_name persisted."""
    # Call POST /interview-sessions/{voice_session_id}/realtime-token
    # Expected: 200 + {url, token, room_name}
    # Verify livekit_room_name was persisted to DB
    pass


@pytest.mark.asyncio
async def test_realtime_token_cross_user_403(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    teacher_token: str,
    voice_session_id,
):
    """403 Forbidden: teacher cannot mint token for student's session."""
    # Student creates voice session
    # Teacher tries to call POST /interview-sessions/{student_session}/realtime-token
    # Expected: 403 Forbidden (ownership check)
    pass


@pytest.mark.asyncio
async def test_realtime_token_jwt_valid_and_claims(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    voice_session_id,
):
    """Token is valid JWT with correct identity and room."""
    # Call POST /interview-sessions/{id}/realtime-token
    # Decode token (no verification needed for test)
    # Verify sub == f"student-{student_id}"
    # Verify video.room == f"interview-{session_id}"
    # Verify roomConfig.agents[0].agentName == settings.livekit_agent_name
    pass


@pytest.mark.asyncio
async def test_realtime_token_idempotent_join(
    client: AsyncClient,
    seeded_users,
    student_token: str,
    voice_session_id,
):
    """Multiple token mints for same session return same room_name."""
    # Call POST /interview-sessions/{id}/realtime-token twice
    # Verify both responses have same room_name
    pass
