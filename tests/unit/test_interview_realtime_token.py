"""Unit tests for LiveKit realtime token minting (Phase 2, no DB).

Tests voice-interview token generation, room naming, agent metadata.
No database required — the one test that builds a real ``Settings`` passes
dummy DB/JWT values as explicit constructor kwargs.

NOTE: we deliberately do NOT mutate ``os.environ`` at import time. Doing so
leaked dummy DB URLs into the whole pytest process (``os.environ`` outranks the
``.env`` file in pydantic-settings), which broke every other test that resolves
``get_settings().database_url`` when this module was imported first.
"""

from __future__ import annotations

import json
from uuid import uuid4

import jwt
import pytest

from abridgeai.core.config import Settings
from abridgeai.features.interviews.schemas.real_time import RealtimeTokenResponse
from abridgeai.features.interviews.services.real_time import (
    META_SESSION_ID,
    META_STUDENT_ID,
    build_agent_metadata,
    build_room_name,
    mint_participant_token,
)


class MockSettings:
    """Duck-typed settings for token minting tests."""

    def __init__(
        self,
        livekit_api_key: str = "test-api-key",
        livekit_api_secret: str = "test-api-secret",  # noqa: S107 -- fixture value, not a credential
        livekit_ws_url: str = "wss://test.livekit.cloud",
        livekit_agent_name: str = "interview-agent",
        interview_voice_token_ttl_seconds: int = 3600,
    ):
        self.livekit_api_key = MockSecretStr(livekit_api_key)
        self.livekit_api_secret = MockSecretStr(livekit_api_secret)
        self.livekit_ws_url = livekit_ws_url
        self.livekit_agent_name = livekit_agent_name
        self.interview_voice_token_ttl_seconds = interview_voice_token_ttl_seconds


class MockSecretStr:
    """Duck-typed SecretStr for mocking."""

    def __init__(self, value: str):
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class TestBuildRoomName:
    """Deterministic room naming from session ID."""

    def test_build_room_name_format(self):
        """Room name follows f'interview-{session_id}' format."""
        session_id = uuid4()
        room_name = build_room_name(session_id)
        assert room_name == f"interview-{session_id}"

    def test_build_room_name_idempotent(self):
        """Same session_id always produces same room name."""
        session_id = uuid4()
        assert build_room_name(session_id) == build_room_name(session_id)


class TestBuildAgentMetadata:
    """Agent metadata JSON construction."""

    def test_build_agent_metadata_json_structure(self):
        """Metadata JSON contains session_id and student_id keys."""
        session_id = uuid4()
        student_id = uuid4()
        metadata_json = build_agent_metadata(session_id, student_id)
        metadata = json.loads(metadata_json)

        assert META_SESSION_ID in metadata
        assert META_STUDENT_ID in metadata
        assert metadata[META_SESSION_ID] == str(session_id)
        assert metadata[META_STUDENT_ID] == str(student_id)

    def test_build_agent_metadata_string_ids(self):
        """IDs are stored as strings in JSON."""
        session_id = uuid4()
        student_id = uuid4()
        metadata = json.loads(build_agent_metadata(session_id, student_id))

        assert isinstance(metadata[META_SESSION_ID], str)
        assert isinstance(metadata[META_STUDENT_ID], str)


class TestMintParticipantToken:
    """LiveKit JWT token minting and contents."""

    def test_mint_token_response_shape(self):
        """Token response contains url, token, room_name."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings()

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        assert isinstance(response, RealtimeTokenResponse)
        assert response.url == settings.livekit_ws_url
        assert response.room_name == room_name
        assert isinstance(response.token, str)
        assert len(response.token) > 0

    def test_mint_token_emits_livekit_field_names(self):
        """Response also carries `server_url`/`participant_token`.

        LiveKit's `TokenSource` parses a token response into the protobuf
        message `livekit.TokenSourceResponse`, which declares exactly
        `server_url` and `participant_token`, with unknown fields ignored. A
        payload carrying only `url`/`token` therefore parses to two empty
        strings rather than failing loudly, so this asserts the LiveKit-named
        fields are present and mirror the originals.
        """
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings()

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        assert response.server_url == response.url
        assert response.participant_token == response.token

        # Assert on the serialized payload, not the attributes: the wire JSON is
        # what a TokenSource client reads, and only that catches a field lost to
        # an exclude/alias change.
        payload = response.model_dump()
        assert payload["server_url"] == settings.livekit_ws_url
        assert payload["participant_token"] == response.token
        assert payload["url"] == settings.livekit_ws_url
        assert payload["token"] == response.token

    def test_livekit_field_names_derive_when_omitted(self):
        """Constructing with only url/token still yields a complete payload.

        The mirroring is a model validator rather than two extra arguments at
        the call site, so a future construction site cannot emit half the
        contract. This pins that behaviour.
        """
        response = RealtimeTokenResponse(
            url="wss://example.livekit.cloud",
            token="jwt-value",  # noqa: S106 -- fixture value, not a credential
            room_name="interview-abc",
        )

        assert response.server_url == "wss://example.livekit.cloud"
        assert response.participant_token == "jwt-value"  # noqa: S105

    def test_mint_token_jwt_claims(self):
        """Minted JWT contains correct identity, room, and video grants."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings()

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        # Decode without verification (test only — no need for secret)
        decoded = jwt.decode(response.token, options={"verify_signature": False})

        # Claims set by AccessToken.with_identity()
        assert decoded["sub"] == f"student-{student_id}"
        assert decoded["name"] == f"student-{student_id}"

        # Video grants
        assert "video" in decoded
        video = decoded["video"]
        assert video["roomJoin"] is True
        assert video["room"] == room_name
        assert video["canPublish"] is True
        assert video["canSubscribe"] is True
        assert video["canPublishData"] is True

    def test_mint_token_agent_metadata(self):
        """JWT contains agent metadata from RoomConfiguration."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings(livekit_agent_name="test-agent-worker")

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        decoded = jwt.decode(response.token, options={"verify_signature": False})

        # RoomConfiguration serialized in roomConfig
        assert "roomConfig" in decoded
        room_config = decoded["roomConfig"]
        assert "agents" in room_config
        agents = room_config["agents"]
        assert len(agents) > 0

        agent = agents[0]
        assert agent["agentName"] == "test-agent-worker"
        assert "metadata" in agent

        # Metadata is a JSON string (livekit SDK behavior)
        agent_metadata = json.loads(agent["metadata"])
        assert agent_metadata[META_SESSION_ID] == str(session_id)
        assert agent_metadata[META_STUDENT_ID] == str(student_id)

    def test_mint_token_ttl_set(self):
        """Token TTL comes from settings.interview_voice_token_ttl_seconds."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        ttl_seconds = 7200
        settings = MockSettings(interview_voice_token_ttl_seconds=ttl_seconds)

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        decoded = jwt.decode(response.token, options={"verify_signature": False})

        # exp is present; we check the token was minted with a future expiry
        exp = decoded.get("exp")
        assert exp is not None
        # exp should be in the future (at least some seconds away)
        assert exp > 0

    def test_mint_token_missing_api_key(self):
        """ValueError raised if livekit_api_key is None."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings(livekit_api_key=None)

        # LiveKit SDK raises ValueError on missing credentials
        with pytest.raises(ValueError):
            mint_participant_token(
                session_id=session_id,
                student_id=student_id,
                room_name=room_name,
                settings=settings,
            )

    def test_mint_token_missing_api_secret(self):
        """ValueError raised if livekit_api_secret is None."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings(livekit_api_secret=None)

        # LiveKit SDK raises ValueError on missing credentials
        with pytest.raises(ValueError):
            mint_participant_token(
                session_id=session_id,
                student_id=student_id,
                room_name=room_name,
                settings=settings,
            )

    def test_mint_token_missing_ws_url(self):
        """ValueError raised if livekit_ws_url is None."""
        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)
        settings = MockSettings(livekit_ws_url=None)

        with pytest.raises(ValueError, match="LiveKit credentials"):
            mint_participant_token(
                session_id=session_id,
                student_id=student_id,
                room_name=room_name,
                settings=settings,
            )

    def test_mint_token_with_real_settings(self):
        """Token minting works with real Settings object (dummy DB URLs)."""
        from pydantic import SecretStr

        session_id = uuid4()
        student_id = uuid4()
        room_name = build_room_name(session_id)

        # Build a real Settings object. DB/JWT values are passed explicitly as
        # dummy kwargs (never connected) instead of polluting os.environ, so this
        # test can't corrupt get_settings() for the rest of the suite.
        settings = Settings(
            jwt_secret_key="x" * 40,
            database_url="postgresql+psycopg://u:p@localhost:5432/db",
            test_database_url="postgresql+psycopg://u:p@localhost:5432/db_test",
            livekit_api_key=SecretStr("test-key"),
            livekit_api_secret=SecretStr("test-secret"),
            livekit_ws_url="wss://real.livekit.cloud",
            livekit_agent_name="prod-agent",
            interview_voice_token_ttl_seconds=1800,
        )

        response = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room_name,
            settings=settings,
        )

        decoded = jwt.decode(response.token, options={"verify_signature": False})
        assert decoded["sub"] == f"student-{student_id}"
        assert decoded["video"]["room"] == room_name
