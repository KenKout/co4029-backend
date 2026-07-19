"""LiveKit realtime token + agent-dispatch service (Phase 2).

Pure token factory for the voice interview. Deliberately free of any
SQLAlchemy import so it stays inside the "services do not touch the ORM"
import-linter contract — the caller (learner router) loads the session,
enforces eligibility, and persists ``livekit_room_name``; this module only
turns ``(session_id, student_id, settings)`` into a signed join token.

The minted token carries an explicit *agent dispatch* via
``RoomConfiguration``: when LiveKit first creates the room it dispatches the
worker registered under ``settings.livekit_agent_name`` and hands it the
session/student ids as JSON metadata. So no separate room-create or
dispatch API call is needed — the participant token itself is the trigger.
See ``features/interviews/realtime/`` (Phase 3) for the worker side.
"""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import UUID

from livekit import api

from abridgeai.core.config import Settings
from abridgeai.features.interviews.schemas.real_time import RealtimeTokenResponse

# Metadata keys handed to the dispatched agent. Kept as module constants so
# the Phase 3 worker parses the exact same shape (single source of truth).
META_SESSION_ID = "session_id"
META_STUDENT_ID = "student_id"
# Phase 18: language the adaptive interviewer should speak in (voice parity with
# the REST path, which reads Accept-Language). Normalized to "vi" or "en".
META_LANGUAGE = "language"


def normalize_language(value: str | None) -> str:
    """Normalize an Accept-Language-ish value to 'vi' or 'en' (default 'en')."""
    if value and value.strip().lower().startswith("vi"):
        return "vi"
    return "en"


def build_room_name(session_id: UUID) -> str:
    """Deterministic per-session room name.

    Derived purely from the interview session id so a given session always
    maps to the same room (idempotent mint, safe rejoin). LiveKit always
    joins by this name; upstream slug/module/config ids only resolve or
    create the session, never the room.
    """
    return f"interview-{session_id}"


def build_agent_metadata(session_id: UUID, student_id: UUID, *, language: str = "en") -> str:
    """JSON payload attached to the agent dispatch (parsed by the worker)."""
    return json.dumps(
        {
            META_SESSION_ID: str(session_id),
            META_STUDENT_ID: str(student_id),
            META_LANGUAGE: normalize_language(language),
        }
    )


def mint_participant_token(
    *,
    session_id: UUID,
    student_id: UUID,
    room_name: str,
    settings: Settings,
    language: str = "en",
) -> RealtimeTokenResponse:
    """Mint a room-scoped participant JWT that dispatches the interview agent.

    Caller MUST have already verified voice is enabled + the session is
    owned by ``student_id`` and in a voice-eligible state. Raises
    ``ValueError`` if credentials are missing (defensive — startup
    validation in :class:`Settings` should prevent this reaching here).
    """
    if not (settings.livekit_ws_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise ValueError("LiveKit credentials are not configured")

    token = (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(f"student-{student_id}")
        .with_name(f"student-{student_id}")
        .with_ttl(timedelta(seconds=settings.interview_voice_token_ttl_seconds))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                # Data channel used for live transcript sync (agent → client).
                can_publish_data=True,
            )
        )
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=settings.livekit_agent_name,
                        metadata=build_agent_metadata(session_id, student_id, language=language),
                    )
                ],
            )
        )
        .to_jwt()
    )

    return RealtimeTokenResponse(
        url=settings.livekit_ws_url,
        token=token,
        room_name=room_name,
    )


__all__ = [
    "META_LANGUAGE",
    "META_SESSION_ID",
    "META_STUDENT_ID",
    "build_agent_metadata",
    "build_room_name",
    "mint_participant_token",
    "normalize_language",
]
