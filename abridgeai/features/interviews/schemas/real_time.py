"""Realtime (LiveKit voice) DTOs for the interview-taking flow (Phase 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RealtimeTokenResponse(BaseModel):
    """Client payload for joining the LiveKit voice-interview room.

    ``url`` is the public WS endpoint (LiveKit Cloud), ``token`` is a
    short-lived, room-scoped participant JWT (it also dispatches the AI
    interview agent), and ``room_name`` is the deterministic per-session
    room. The server-side API secret is never returned — only the signed
    token derived from it.
    """

    url: str = Field(description="LiveKit WS URL, e.g. wss://<project>.livekit.cloud")
    token: str = Field(description="Short-lived room-scoped participant JWT")
    room_name: str = Field(description="Deterministic per-session room name")


__all__ = ["RealtimeTokenResponse"]
