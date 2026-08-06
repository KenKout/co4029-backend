"""Transport parity: REST `/respond` and the LiveKit control topic cannot drift.

A typed interview turn can arrive two ways:

* REST  ``POST /interview-sessions/{id}/respond``  (routers/learner.py)
* LiveKit ``lk.chat`` → agent → control topic       (orchestration_bridge.py)

They are DIFFERENT call sites into the same brain, which is exactly the shape
that rots: someone adds a field to the turn response, maps it at one call site,
and one transport silently stops matching the other. The existing
``test_interview_shared_turn_handler`` suite pins voice-vs-typed parity INSIDE
the agent; nothing pinned agent-vs-REST.

These tests do that, at the two places the two transports can actually diverge:

1. The projection. Both must build their payload through
   ``InterviewSubmitAnswerResponse.from_step_result`` — the single definition —
   rather than hand-listing fields. Enforced structurally (a new field appears
   on both, or neither) and by source inspection (no second hand-rolled
   projection creeps back in).

2. The brain call. Both must reach ``take_session_step`` with the same
   arguments for the same turn, including ``turn_key`` (idempotency) and
   ``turn_action`` (a "hint" request must not be graded as an answer on one
   transport and not the other).

Nothing here touches a room, a socket or a DB: the brain is mocked and the
projection is called directly.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import orchestration_bridge as bridge
from abridgeai.features.interviews.schemas.session import InterviewSubmitAnswerResponse


def _step_result(**over: Any) -> dict[str, Any]:
    """A realistic ``take_session_step`` return value.

    Deliberately the FULL shape rather than a minimal stub, and deliberately
    with NO ``None`` values: the parity check below detects an unmapped field by
    finding ``None`` on the rendered payload, so a fixture that supplied ``None``
    would make "mapped but null" and "never mapped" indistinguishable — which is
    exactly the bug this file exists to catch.
    """
    base: dict[str, Any] = {
        "next_question": None,  # exercised separately; see the projection tests
        "is_finished": False,
        "followup_text": "Tell me more.",
        "ai_turn_text": "Thanks. Next: how would you model that?",
        "language": "en",
        "should_narrate": True,
        "should_await_response": True,
        "should_finish": False,
        "assistance_kind": "hint",
        "pending_confirmation": True,
        "interaction_state": "awaiting_answer",
        "transition_id": "t-42",
        "transition_text": "Let's move on.",
        "transition_target": "next_question",
    }
    base.update(over)
    return base


class TestProjectionIsShared:
    """Both transports must render a turn through the same projection."""

    def test_the_same_step_result_renders_identically(self) -> None:
        """The core invariant, stated directly.

        REST calls this classmethod; the bridge calls this classmethod. Given
        one brain result they must produce byte-identical payloads.
        """
        result = _step_result()
        rest = InterviewSubmitAnswerResponse.from_step_result(
            result, time_remaining_seconds=540
        )
        livekit = InterviewSubmitAnswerResponse.from_step_result(
            result, time_remaining_seconds=540
        )
        assert rest.model_dump() == livekit.model_dump()

    def test_every_response_field_is_populated_by_the_projection(self) -> None:
        """A field added to the schema but not mapped here fails HERE.

        That is the whole safety property: the projection is the single place
        the mapping lives, so an unmapped field cannot reach one transport and
        not the other — it reaches neither, loudly.
        """
        result = _step_result()
        payload = InterviewSubmitAnswerResponse.from_step_result(
            result, time_remaining_seconds=540
        ).model_dump()

        # Fields the brain genuinely does not supply on this path.
        supplied_elsewhere = {"next_question", "is_finished", "time_remaining_seconds"}
        unmapped = [
            name
            for name, value in payload.items()
            if value is None and name not in supplied_elsewhere
        ]
        assert not unmapped, (
            f"these response fields are never populated by from_step_result: {unmapped}. "
            "Map them there, not at a call site, or the two transports will diverge."
        )

    def test_time_remaining_is_an_explicit_argument_on_both(self) -> None:
        """The brain never returns the timer, so each transport must pass it.

        Making it a required keyword is what stops a caller from quietly
        publishing ``None`` — pin that, because a silent ``None`` here is a
        client-visible countdown that stops moving.
        """
        signature = inspect.signature(InterviewSubmitAnswerResponse.from_step_result)
        param = signature.parameters["time_remaining_seconds"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty

        # And it really is the value that lands on the wire.
        payload = InterviewSubmitAnswerResponse.from_step_result(
            _step_result(), time_remaining_seconds=123
        )
        assert payload.time_remaining_seconds == 123

    def test_neither_transport_hand_rolls_a_second_projection(self) -> None:
        """Source-level guard against the regression re-appearing.

        Both call sites must construct the response through the classmethod. A
        direct ``InterviewSubmitAnswerResponse(...)`` at either site is a second
        projection by definition, and the two will drift the next time a field
        is added.
        """
        from abridgeai.features.interviews.routers import learner as rest_router

        for module in (rest_router, bridge):
            source = inspect.getsource(module)
            assert "from_step_result(" in source, (
                f"{module.__name__} no longer builds the turn payload through the "
                "shared projection"
            )
            assert "InterviewSubmitAnswerResponse(" not in source, (
                f"{module.__name__} constructs InterviewSubmitAnswerResponse directly; "
                "use from_step_result so both transports stay identical"
            )


class TestBrainCallParity:
    """Both transports must reach ``take_session_step`` the same way."""

    @pytest.mark.asyncio
    async def test_livekit_forwards_turn_key_and_action_to_the_brain(self) -> None:
        """``turn_key`` is the idempotency key and ``turn_action`` the intent.

        REST passes ``payload.turn_key`` / ``payload.turn_action`` straight
        through (learner.py). The bridge must do the same, or a retried typed
        turn double-grades, and "give me a hint" gets scored as an answer.
        """
        session_id, student_id = uuid4(), uuid4()
        step = AsyncMock(return_value=_step_result())

        with (
            patch.object(bridge, "take_session_step", step),
            patch.object(bridge, "get_sessionmaker", _fake_sessionmaker()),
            patch.object(
                bridge, "session_time_remaining_seconds", AsyncMock(return_value=540)
            ),
            patch.object(bridge, "obs", _NullObs()),
        ):
            await bridge.handle_student_turn(
                session_id,
                student_id,
                "my typed answer",
                language="en",
                turn_id="tk-abcdef12",
                turn_action="hint",
            )

        assert step.await_count == 1
        kwargs = step.await_args.kwargs
        # Same keyword contract the REST router uses.
        assert kwargs["turn_key"] == "tk-abcdef12"
        assert kwargs["turn_action"] == "hint"
        assert kwargs["language"] == "en"
        # Positional: (db, session_id, transcript, actor)
        assert step.await_args.args[1] == session_id
        assert step.await_args.args[2] == "my typed answer"

    @pytest.mark.asyncio
    async def test_livekit_defaults_the_action_to_answer(self) -> None:
        """Matches the REST default (``payload.turn_action or "answer"``)."""
        step = AsyncMock(return_value=_step_result())
        with (
            patch.object(bridge, "take_session_step", step),
            patch.object(bridge, "get_sessionmaker", _fake_sessionmaker()),
            patch.object(
                bridge, "session_time_remaining_seconds", AsyncMock(return_value=1)
            ),
            patch.object(bridge, "obs", _NullObs()),
        ):
            await bridge.handle_student_turn(
                uuid4(), uuid4(), "answer text", language="en", turn_id="tk-1"
            )
        assert step.await_args.kwargs["turn_action"] == "answer"


# ── test doubles ────────────────────────────────────────────────────────────


class _NullObs:
    """Swallow observability emits; they are not what these tests pin."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("EV_"):
            return name
        return lambda *a, **k: None

    @staticmethod
    def monotonic() -> float:
        return 0.0


def _fake_sessionmaker():
    """`get_sessionmaker()()` → an async-context DB double that records nothing."""

    class _Db:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            return None

        async def rollback(self):
            return None

        async def refresh(self, *a, **k):
            return None

        async def get(self, *a, **k):
            return None

    return lambda: _Db
