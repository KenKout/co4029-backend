"""Realtime (LiveKit voice) DTOs for the interview-taking flow (Phase 2)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RealtimeTokenResponse(BaseModel):
    """Client payload for joining the LiveKit voice-interview room.

    ``url`` is the public WS endpoint (LiveKit Cloud), ``token`` is a
    short-lived, room-scoped participant JWT (it also dispatches the AI
    interview agent), and ``room_name`` is the deterministic per-session
    room. The server-side API secret is never returned — only the signed
    token derived from it.

    ``server_url`` and ``participant_token`` carry the same two values again
    under the names LiveKit's own client reads. The protobuf message
    ``livekit.TokenSourceResponse`` declares exactly two fields —
    ``server_url`` and ``participant_token`` — and ``TokenSource`` parses a
    token response with ``ignore_unknown_fields``, so it sees ``url``/``token``
    as unknown and drops them. Emitting both spellings lets this endpoint back
    a stock ``TokenSource.endpoint(...)`` client without breaking the existing
    ``url``/``token`` consumers.

    Duplicated rather than renamed on purpose: renaming is a breaking change to
    a response the frontend already destructures, and an alias generated via
    ``serialization_alias`` would emit only ONE of the two names, which defeats
    the point.
    """

    url: str = Field(description="LiveKit WS URL, e.g. wss://<project>.livekit.cloud")
    token: str = Field(description="Short-lived room-scoped participant JWT")
    room_name: str = Field(description="Deterministic per-session room name")
    server_url: str = Field(
        default="",
        description="Alias of `url`, named for LiveKit TokenSource compatibility",
    )
    participant_token: str = Field(
        default="",
        description="Alias of `token`, named for LiveKit TokenSource compatibility",
    )

    @model_validator(mode="after")
    def _mirror_livekit_field_names(self) -> RealtimeTokenResponse:
        """Derive the LiveKit-named fields so they cannot drift from the originals.

        Defaulted + derived rather than required: a construction site that only
        knows about ``url``/``token`` still emits a complete, TokenSource-readable
        payload. Two names for one value can only stay consistent if exactly one
        of them is authoritative.
        """
        if not self.server_url:
            self.server_url = self.url
        if not self.participant_token:
            self.participant_token = self.token
        return self


__all__ = ["RealtimeTokenResponse"]
