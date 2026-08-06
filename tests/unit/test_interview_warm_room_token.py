"""Warm-room split: joining the room is not the same as starting the interview.

The token used to carry ``RoomConfiguration.agents``, so the only way to open a
room was to simultaneously dispatch the interviewer. That is why the room could
not be opened until onboarding finished — and why the candidate then waited for
the worker to spin up (measured 10.0s / 13.2s / 13.3s from mint to
``voice.room_join``) as dead air in front of question one.

A warm token grants the same room access with no dispatch, so the client can
join during setup and the ~10-13s startup overlaps work the candidate is doing
anyway. The interviewer is sent in afterwards, which is also when ``language``
is finally settled (the language check is part of onboarding).

These tests pin the security-relevant half: a warm token must be identical to a
normal one EXCEPT for the dispatch, and must not smuggle extra grants.
"""

from __future__ import annotations

import base64
import json
from uuid import uuid4

import pytest

from abridgeai.core.config import Settings
from abridgeai.features.interviews.services.real_time import (
    build_room_name,
    mint_participant_token,
)


def _settings() -> Settings:
    return Settings(
        livekit_ws_url="wss://example.livekit.cloud",
        livekit_api_key="devkey",  # noqa: S106 - test fixture
        livekit_api_secret="secret-for-tests-only",  # noqa: S106 - test fixture
        livekit_agent_name="interview-agent",
        interview_voice_enabled=True,
    )


def _claims(token: str) -> dict:
    """Decode a JWT payload without verifying (we only inspect claims)."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _mint(*, dispatch_agent: bool) -> dict:
    session_id, student_id = uuid4(), uuid4()
    result = mint_participant_token(
        session_id=session_id,
        student_id=student_id,
        room_name=build_room_name(session_id),
        settings=_settings(),
        language="vi",
        dispatch_agent=dispatch_agent,
    )
    return _claims(result.token)


class TestWarmToken:
    def test_warm_token_carries_no_agent_dispatch(self) -> None:
        """The whole point: joining with this must not start the interviewer."""
        claims = _mint(dispatch_agent=False)
        assert "roomConfig" not in claims or not claims.get("roomConfig", {}).get(
            "agents"
        )

    def test_default_token_still_dispatches(self) -> None:
        """The original path is unchanged — this is additive, not a swap."""
        claims = _mint(dispatch_agent=True)
        agents = claims["roomConfig"]["agents"]
        assert agents
        assert agents[0]["agentName"] == "interview-agent"

    def test_warm_token_grants_the_same_room_access(self) -> None:
        """A warm token is not a weaker token — it just starts nothing.

        If the grants differed, the client could not actually use the room it
        warmed, and the whole optimisation would be a no-op with extra moving
        parts.
        """
        warm = _mint(dispatch_agent=False)["video"]
        normal = _mint(dispatch_agent=True)["video"]
        assert warm["roomJoin"] == normal["roomJoin"] is True
        assert warm["canPublish"] == normal["canPublish"] is True
        assert warm["canSubscribe"] == normal["canSubscribe"] is True
        assert warm["canPublishData"] == normal["canPublishData"] is True

    def test_warm_token_grants_no_extra_privileges(self) -> None:
        """Guard the other direction: warming must not widen the grant set."""
        # Same ids for both mints — `_mint` generates fresh ones per call, so
        # comparing two independent mints would compare two different rooms.
        session_id, student_id = uuid4(), uuid4()
        common = {
            "session_id": session_id,
            "student_id": student_id,
            "room_name": build_room_name(session_id),
            "settings": _settings(),
        }
        warm = _claims(mint_participant_token(**common, dispatch_agent=False).token)[
            "video"
        ]
        normal = _claims(mint_participant_token(**common, dispatch_agent=True).token)[
            "video"
        ]
        assert set(warm) == set(normal)
        for key, value in warm.items():
            assert value == normal[key], f"warm token differs on {key}"

    def test_warm_token_is_scoped_to_its_own_room(self) -> None:
        """Room scoping is what stops a warm token being a lobby key."""
        session_id, student_id = uuid4(), uuid4()
        room = build_room_name(session_id)
        token = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=room,
            settings=_settings(),
            dispatch_agent=False,
        )
        assert _claims(token.token)["video"]["room"] == room

    def test_both_shapes_identify_the_same_participant(self) -> None:
        """Identity must not change with the dispatch flag.

        The agent correlates the candidate by participant identity; a warm join
        followed by a dispatched agent has to look like one person, not two.
        """
        session_id, student_id = uuid4(), uuid4()
        room = build_room_name(session_id)
        common = {
            "session_id": session_id,
            "student_id": student_id,
            "room_name": room,
            "settings": _settings(),
        }
        warm = _claims(mint_participant_token(**common, dispatch_agent=False).token)
        normal = _claims(mint_participant_token(**common, dispatch_agent=True).token)
        assert warm["sub"] == normal["sub"] == f"student-{student_id}"


class TestDispatchMetadata:
    def test_dispatch_metadata_carries_the_resolved_language(self) -> None:
        """Language reaches the agent through the dispatch, not the room.

        This is the reason the dispatch has to happen AFTER onboarding: the
        language check is one of the onboarding steps, so a dispatch embedded
        in an early token would ship whatever the session defaulted to.
        """
        session_id, student_id = uuid4(), uuid4()
        token = mint_participant_token(
            session_id=session_id,
            student_id=student_id,
            room_name=build_room_name(session_id),
            settings=_settings(),
            language="vi",
            dispatch_agent=True,
        )
        meta = json.loads(_claims(token.token)["roomConfig"]["agents"][0]["metadata"])
        assert meta["language"] == "vi"
        assert meta["session_id"] == str(session_id)
        assert meta["student_id"] == str(student_id)


@pytest.mark.parametrize("dispatch_agent", [True, False])
def test_room_name_is_stable_across_both_shapes(dispatch_agent: bool) -> None:
    """Warm-then-dispatch only works if both resolve the same room."""
    session_id = uuid4()
    token = mint_participant_token(
        session_id=session_id,
        student_id=uuid4(),
        room_name=build_room_name(session_id),
        settings=_settings(),
        dispatch_agent=dispatch_agent,
    )
    assert token.room_name == f"interview-{session_id}"
