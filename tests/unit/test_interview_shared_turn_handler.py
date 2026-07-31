"""Shared turn handler: typed (`lk.chat`) turns behave identically to spoken ones.

The point of these tests is the invariant that motivated the refactor: BOTH
modalities go through ``InterviewAgent._process_turn``, so grading,
observability, thinking filler, TTS, closing playout and shutdown cannot diverge.
Before the refactor a typed turn would have bypassed all of it — LiveKit's
``_on_chat_text_stream`` calls ``text_input_cb`` directly and never touches
``on_user_turn_completed``.

LiveKit is not exercised: ``InterviewAgent.session`` is patched with a fake, and
the bridge is mocked. No room, no DB, no audio.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from livekit.agents import StopResponse

from abridgeai.features.interviews.realtime import text_protocol as tp
from abridgeai.features.interviews.realtime.orchestration_bridge import TurnResult
from abridgeai.features.interviews.realtime.session_runtime import InterviewAgent


class FakeSpeechHandle:
    """Awaitable stand-in for a livekit SpeechHandle."""

    def __init__(self) -> None:
        self.playout_awaited = False

    def __await__(self):
        async def _noop():
            return None

        return _noop().__await__()

    async def wait_for_playout(self) -> None:
        self.playout_awaited = True


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, text: str, *, topic: str = "") -> None:
        self.sent.append((topic, text))


class FakeSession:
    """Minimal AgentSession surface used by InterviewAgent."""

    def __init__(self) -> None:
        self.said: list[tuple[str, dict]] = []
        self.interrupted = 0
        self.claims = 0
        self.local_participant = FakeLocalParticipant()
        self.room = SimpleNamespace(local_participant=self.local_participant)

    def say(self, text: str, **kwargs) -> FakeSpeechHandle:
        self.said.append((text, kwargs))
        return FakeSpeechHandle()

    async def interrupt(self) -> None:
        self.interrupted += 1

    @asynccontextmanager
    async def _claim_user_turn(self):
        self.claims += 1
        yield


def make_agent(session: FakeSession, *, language: str = "en") -> InterviewAgent:
    agent = InterviewAgent(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        first_question_text="Q1?",
        language=language,
    )
    # `Agent.session` is a read-only property on the real class; patch the
    # instance's lookup so the fake is returned.
    patcher = patch.object(type(agent), "session", property(lambda self: session))
    patcher.start()
    agent._test_patcher = patcher  # type: ignore[attr-defined]
    return agent


def control_events(session: FakeSession) -> list[dict]:
    return [
        json.loads(body)
        for topic, body in session.local_participant.sent
        if topic == tp.TOPIC_CONTROL
    ]


def text_event(text: str, attributes: dict[str, str] | None = None):
    info = SimpleNamespace(attributes=attributes) if attributes is not None else None
    return SimpleNamespace(text=text, info=info, participant=None)


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture(autouse=True)
def _no_filler_delay():
    # The thinking filler sleeps 0.8s before speaking; collapse it so tests do
    # not wait, while leaving the cancellation logic intact.
    with patch(
        "abridgeai.features.interviews.realtime.session_runtime._THINKING_FILLER_DELAY_S",
        0,
    ):
        yield


NOT_FINISHED = TurnResult(
    speak_text="Good. Next question?",
    is_finished=False,
    next_question_text="Next question?",
    state_version=3,
)
FINISHED = TurnResult(
    speak_text="That concludes the interview.",
    is_finished=True,
    suppress_default_closing=True,
    state_version=9,
)


class TestSharedHandlerParity:
    """Voice and typed turns must produce the same brain call and the same speech."""

    @pytest.mark.asyncio
    async def test_voice_and_text_call_the_brain_identically(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            # Spoken turn
            msg = MagicMock()
            msg.text_content = "spoken answer"
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(MagicMock(), msg)
            voice_call = brain.call_args

            # Typed turn, same content
            await agent.on_text_input(session, text_event("spoken answer"))
            text_call = brain.call_args

        # Same session, same student, same transcript, same language.
        assert voice_call.args == text_call.args
        assert voice_call.kwargs["language"] == text_call.kwargs["language"]
        # Both default the action; only the ids differ (typed may carry a client key).
        assert voice_call.kwargs["turn_action"] == "answer"
        assert text_call.kwargs["turn_action"] == "answer"

    @pytest.mark.asyncio
    async def test_both_paths_speak_the_brains_utterance(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            msg = MagicMock()
            msg.text_content = "a"
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(MagicMock(), msg)
            spoken_after_voice = [t for t, _ in session.said]

            session.said.clear()
            await agent.on_text_input(session, text_event("a"))
            spoken_after_text = [t for t, _ in session.said]

        assert "Good. Next question?" in spoken_after_voice
        assert "Good. Next question?" in spoken_after_text

    @pytest.mark.asyncio
    async def test_typed_final_turn_shuts_the_room_down(self, session):
        # The regression this refactor exists to prevent. A typed final answer
        # must run the SAME finish path as a spoken one: non-interruptible
        # closing, playout wait, then shutdown. Before the shared handler, a
        # typed turn reached the brain but never got here.
        agent = make_agent(session)
        job_ctx = MagicMock()
        with (
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
                new=AsyncMock(return_value=FINISHED),
            ),
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.get_job_context",
                return_value=job_ctx,
            ),
        ):
            await agent.on_text_input(session, text_event("my final answer"))

        closing_says = [kw for text, kw in session.said if "concludes" in text]
        assert closing_says, "closing utterance was never spoken"
        assert closing_says[0]["allow_interruptions"] is False
        job_ctx.shutdown.assert_called_once()
        assert job_ctx.shutdown.call_args.kwargs["reason"] == "interview_complete"

    @pytest.mark.asyncio
    async def test_voice_final_turn_still_shuts_down(self, session):
        # Guards the refactor from the other direction: the spoken path must not
        # have lost the shutdown when its body moved.
        agent = make_agent(session)
        job_ctx = MagicMock()
        with (
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
                new=AsyncMock(return_value=FINISHED),
            ),
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.get_job_context",
                return_value=job_ctx,
            ),
        ):
            msg = MagicMock()
            msg.text_content = "final spoken answer"
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(MagicMock(), msg)

        job_ctx.shutdown.assert_called_once()


class TestTurnActionPassthrough:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["repeat", "clarify", "explain_term", "hint"])
    async def test_turn_action_reaches_the_brain(self, session, action):
        # Without this, "give me a hint" would be graded as an answer.
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            await agent.on_text_input(session, text_event("?", {tp.ATTR_TURN_ACTION: action}))
        assert brain.call_args.kwargs["turn_action"] == action

    @pytest.mark.asyncio
    async def test_client_turn_key_is_used_as_the_idempotency_key(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            await agent.on_text_input(session, text_event("a", {tp.ATTR_TURN_KEY: "tk-12345678"}))
        # Reused as turn_id so a retry with the same key is idempotent in the
        # brain AND correlates in telemetry.
        assert brain.call_args.kwargs["turn_id"] == "tk-12345678"


class TestGuards:
    @pytest.mark.asyncio
    async def test_rejects_a_second_turn_while_one_is_in_flight(self, session):
        agent = make_agent(session)
        agent._turn_in_flight = True
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            await agent.on_text_input(session, text_event("second"))
        brain.assert_not_called()
        evs = control_events(session)
        assert evs[-1]["status"] == "rejected"
        assert evs[-1]["rejection"] == "turn_in_flight"

    @pytest.mark.asyncio
    async def test_rejects_a_turn_typed_during_the_closing(self, session):
        # A turn accepted here would be graded after the session was submitted.
        agent = make_agent(session)
        agent._closing = True
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            await agent.on_text_input(session, text_event("wait, one more thing"))
        brain.assert_not_called()
        assert control_events(session)[-1]["rejection"] == "session_closing"

    @pytest.mark.asyncio
    async def test_closing_flag_is_set_before_the_closing_is_spoken(self, session):
        agent = make_agent(session)
        with (
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
                new=AsyncMock(return_value=FINISHED),
            ),
            patch(
                "abridgeai.features.interviews.realtime.session_runtime.get_job_context",
                return_value=None,
            ),
        ):
            await agent.on_text_input(session, text_event("final"))
        assert agent._closing is True

    @pytest.mark.asyncio
    async def test_in_flight_clears_after_a_failed_turn(self, session):
        # Otherwise one brain error would wedge the composer shut for the rest of
        # the session.
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(side_effect=RuntimeError("brain exploded")),
        ):
            await agent.on_text_input(session, text_event("a"))
        assert agent._turn_in_flight is False

    @pytest.mark.asyncio
    async def test_malformed_attributes_never_reach_the_brain(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ) as brain:
            await agent.on_text_input(
                session, text_event("x", {tp.ATTR_TURN_ACTION: "not_a_real_action"})
            )
        brain.assert_not_called()
        assert control_events(session)[-1]["rejection"] == "invalid_turn_action"

    @pytest.mark.asyncio
    async def test_typed_turn_interrupts_agent_speech_and_claims_the_turn(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a"))
        assert session.interrupted == 1
        assert session.claims == 1


class TestControlStream:
    @pytest.mark.asyncio
    async def test_emits_accepted_then_completed(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a", {tp.ATTR_TURN_KEY: "tk-12345678"}))
        evs = control_events(session)
        assert [e["status"] for e in evs] == ["accepted", "completed"]
        assert all(e["turn_key"] == "tk-12345678" for e in evs)

    @pytest.mark.asyncio
    async def test_seq_increases_strictly(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a"))
            await agent.on_text_input(session, text_event("b"))
        seqs = [e["seq"] for e in control_events(session)]
        assert len(seqs) == 4
        assert seqs == sorted(set(seqs))

    @pytest.mark.asyncio
    async def test_completed_carries_the_brains_state_version(self, session):
        # `state_version` must be the BRAIN's value (for reconciling persisted
        # history), not the agent's own control sequence.
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a"))
        completed = [e for e in control_events(session) if e["status"] == "completed"][0]
        assert completed["state_version"] == 3
        assert completed["state"]["next_question_text"] == "Next question?"

    @pytest.mark.asyncio
    async def test_failed_event_reports_only_the_error_class(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(side_effect=RuntimeError("prompt leaked in here")),
        ):
            await agent.on_text_input(session, text_event("a"))
        failed = [e for e in control_events(session) if e["status"] == "failed"][0]
        assert failed["error_class"] == "RuntimeError"
        assert "prompt leaked" not in json.dumps(failed)

    @pytest.mark.asyncio
    async def test_voice_turns_publish_no_control_events(self, session):
        # Control exists for typed clients. A voice-only session must not pay for
        # extra data messages it never reads.
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            msg = MagicMock()
            msg.text_content = "spoken"
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(MagicMock(), msg)
        assert control_events(session) == []

    @pytest.mark.asyncio
    async def test_control_publish_failure_does_not_break_the_turn(self, session):
        # The brain has already committed by then; a failed convenience message
        # must never surface as a failed turn.
        agent = make_agent(session)
        session.local_participant.send_text = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("transport down")
        )
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a"))
        assert any("Next question?" in t for t, _ in session.said)

    @pytest.mark.asyncio
    async def test_control_uses_the_application_topic_not_lk_chat(self, session):
        agent = make_agent(session)
        with patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=AsyncMock(return_value=NOT_FINISHED),
        ):
            await agent.on_text_input(session, text_event("a"))
        topics = {topic for topic, _ in session.local_participant.sent}
        assert topics == {tp.TOPIC_CONTROL}
        assert tp.TOPIC_CHAT not in topics
