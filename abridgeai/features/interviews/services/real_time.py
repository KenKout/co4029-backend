"""LiveKit realtime token + agent-dispatch service (Phase 2).

Pure token factory for the voice interview. Deliberately free of any
SQLAlchemy import so it stays inside the "services do not touch the ORM"
import-linter contract — the caller (learner router) loads the session,
enforces eligibility, and persists ``livekit_room_name``; this module only
turns ``(session_id, student_id, settings)`` into a signed join token.

Agent dispatch happens one of two ways, and the choice is the whole point of
this module's shape:

* **Token-embedded** (``dispatch_agent=True``, the original path). The minted
  token carries ``RoomConfiguration.agents``, so LiveKit dispatches the worker
  the moment it creates the room. Simple, but it welds "candidate joined" to
  "interview started".
* **Warm room + explicit dispatch** (``dispatch_agent=False`` then
  :func:`dispatch_interview_agent`). The candidate joins during setup with a
  token that starts nothing, and the interviewer is sent in afterwards through
  the agent-dispatch API. This is what lets the ~10-13s worker startup overlap
  the onboarding the candidate is doing anyway, instead of being dead air in
  front of question one.

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
    dispatch_agent: bool = True,
) -> RealtimeTokenResponse:
    """Mint a room-scoped participant JWT.

    Caller MUST have already verified voice is enabled + the session is
    owned by ``student_id`` and in a voice-eligible state. Raises
    ``ValueError`` if credentials are missing (defensive — startup
    validation in :class:`Settings` should prevent this reaching here).

    ``dispatch_agent=False`` mints a **warm-up** token: identical grants, but
    no ``RoomConfiguration.agents``, so joining creates the room WITHOUT
    starting the interviewer. That split exists because embedding the dispatch
    in the token welds two separate things together — "the candidate is in the
    room" and "the interview has begun" — which is why the room could not be
    opened until onboarding finished, and why the candidate then waited
    10-13s (measured) for the worker to spin up before question one.

    With the warm token the room can be joined during setup and the agent
    dispatched later via :func:`dispatch_interview_agent`, once
    ``language``/``preferred_name`` are actually settled. The agent still
    cannot speak early, because it is not there yet.
    """
    if not (settings.livekit_ws_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise ValueError("LiveKit credentials are not configured")

    builder = (
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
    )
    if dispatch_agent:
        builder = builder.with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=settings.livekit_agent_name,
                        metadata=build_agent_metadata(session_id, student_id, language=language),
                    )
                ],
            )
        )

    return RealtimeTokenResponse(
        url=settings.livekit_ws_url,
        token=builder.to_jwt(),
        room_name=room_name,
    )


async def dispatch_interview_agent(
    *,
    session_id: UUID,
    student_id: UUID,
    room_name: str,
    settings: Settings,
    language: str = "en",
) -> None:
    """Send the interviewer into an already-open room.

    The other half of the warm-room split: the candidate joins early with a
    token minted at ``dispatch_agent=False``, and this runs once onboarding is
    complete — at which point ``language`` is finally known, so the agent is
    handed the value the candidate actually chose rather than a guess made
    before the language check ran.

    Idempotent in the way that matters: LiveKit refuses a second dispatch of
    the same agent into the same room, and the caller (the router) already
    guards on session state. A failure here must be surfaced, NOT swallowed —
    an interview with no interviewer in the room is a dead session, and the
    caller falls back to the token-embedded dispatch path.
    """
    if not (settings.livekit_ws_url and settings.livekit_api_key and settings.livekit_api_secret):
        raise ValueError("LiveKit credentials are not configured")

    lkapi = api.LiveKitAPI(
        url=settings.livekit_ws_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
    )
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=settings.livekit_agent_name,
                room=room_name,
                metadata=build_agent_metadata(session_id, student_id, language=language),
            )
        )
    finally:
        await lkapi.aclose()


__all__ = [
    "META_LANGUAGE",
    "META_SESSION_ID",
    "META_STUDENT_ID",
    "build_agent_metadata",
    "build_room_name",
    "dispatch_interview_agent",
    "mint_participant_token",
    "normalize_language",
]
