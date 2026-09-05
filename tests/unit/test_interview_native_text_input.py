"""Typed turns on the native path: graded, attributed, and acknowledged.

The regression these lock down shipped silently. ``room_options_for_mode`` left
``text_input_cb`` unset, so the SDK default ran — and the SDK default reads only
``ev.text`` and reaches ``generate_reply`` without ever passing
``on_user_turn_completed``, which the SDK fires only from the STT path. A typed
answer was therefore never graded and its ``turn_action`` was discarded, and in a
``text`` session (audio disabled at the room boundary) that was every turn.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import text_protocol as tp
from abridgeai.features.interviews.realtime.native_control import ControlPublisher
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb

pytestmark = pytest.mark.asyncio


class _Local:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.on_send: Callable[[str], None] | None = None

    async def send_text(self, text: str, topic: str) -> None:
        if self.on_send is not None:
            self.on_send(text)
        self.sent.append((topic, text))


class _Session:
    """Stand-in for ``AgentSession``: only what the callback actually touches."""

    def __init__(self, agent: Any, local: _Local) -> None:
        self.current_agent = agent
        self.userdata = type(
            "_U", (), {"interview_session_id": uuid4(), "pending_assistant_kind": None}
        )()
        self.room_io = type("_RIO", (), {"room": type("_R", (), {"local_participant": local})})
        self.interrupted = False
        self.interrupt_forced = False
        self.replies: list[dict[str, Any]] = []
        self.claimed = 0

    def _claim_user_turn(self) -> Any:
        session = self

        class _Guard:
            async def __aenter__(self) -> None:
                session.claimed += 1

            async def __aexit__(self, *_: object) -> None:
                return None

        return _Guard()

    async def interrupt(self, *, force: bool = False) -> None:
        self.interrupted = True
        self.interrupt_forced = force

    def generate_reply(self, **kwargs: Any) -> object:
        self.replies.append(kwargs)
        return object()


class _Ctx:
    def __init__(self) -> None:
        self.items: list[str] = []

    def copy(self) -> _Ctx:
        clone = _Ctx()
        clone.items = list(self.items)
        return clone


class _Agent:
    def __init__(self) -> None:
        self.chat_ctx = _Ctx()
        self.folded: list[str] = []
        # The typed door forwards the client's idempotency key so the fold can
        # persist it as `last_turn_idempotency_key` — that is what makes a resend
        # recognisable after an agent restart, when the in-memory ledger is gone.
        self.folded_keys: list[str | None] = []
        self.on_fold: Callable[[str], Awaitable[None]] | None = None

    async def fold_turn(self, *, answer_text: str, turn_key: str | None = None) -> None:
        self.folded.append(answer_text)
        self.folded_keys.append(turn_key)
        if self.on_fold is not None:
            await self.on_fold(answer_text)


class _Event:
    def __init__(self, text: str, attributes: dict[str, str] | None = None) -> None:
        self.text = text
        self.info = type("_Info", (), {"attributes": attributes or {}})()


def _harness() -> tuple[_Session, _Agent, _Local, Callable[[Any, Any], Awaitable[None]]]:
    local = _Local()
    agent = _Agent()
    sess = _Session(agent, local)
    publisher = ControlPublisher(sess, interview_session_id=sess.userdata.interview_session_id)
    return sess, agent, local, make_text_input_cb(publisher)


def _events(local: _Local) -> list[dict[str, Any]]:
    return [json.loads(text) for topic, text in local.sent if topic == tp.TOPIC_CONTROL]


async def test_typed_answer_is_graded() -> None:
    """The whole point. On the SDK default this list stayed empty."""
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("An index is a B-tree.", {"turn_key": "tk-abcd1234"}))

    assert agent.folded == ["An index is a B-tree."]
    assert sess.replies == [{"user_input": "An index is a B-tree."}]


async def test_typed_answer_is_acknowledged_before_grading() -> None:
    """Grading calls an LLM; the composer must not wait behind it."""
    sess, agent, local, cb = _harness()
    order: list[str] = []

    async def _note_fold(_answer: str) -> None:
        order.append("fold")

    agent.on_fold = _note_fold
    local.on_send = lambda text: order.append(json.loads(text)["status"])

    await cb(sess, _Event("hello", {"turn_key": "tk-abcd1234"}))

    assert order == ["accepted", "fold"]


async def test_hint_is_not_graded_and_is_routed_to_the_ladder_tool() -> None:
    """A hint request graded as an answer is the scoring bug the protocol prevents.

    Restricting the turn to ``interview_request_hint`` is what debits the ladder
    server-side; a model free-handing a hint spends nothing.
    """
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("give me a hint", {"turn_action": "hint", "turn_key": "tk-abcd1234"}))

    assert agent.folded == [], "a hint request must never be graded as an answer"
    assert sess.replies == [{"user_input": "give me a hint", "tools": ["interview_request_hint"]}]


@pytest.mark.parametrize("action", ["clarify", "explain_term", "repeat"])
async def test_help_requests_are_framed_and_not_graded(action: str) -> None:
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("what does that mean?", {"turn_action": action, "turn_key": "tk-abcd12"}))

    assert agent.folded == []
    reply = sess.replies[0]
    assert "instructions" in reply, f"{action} needs framing or the model grades it and advances"
    assert "do not advance" in reply["instructions"].lower()
    # The reply IS assistance: the transcript kind marker and the client notice
    # must both be set, or the utterance badges as FOLLOW-UP live and after a
    # reload.
    assert sess.userdata.pending_assistant_kind == "clarification"
    actions = [
        (e["status"], e["turn_action"]) for e in _events(local) if e["status"] == "agent_action"
    ]
    assert actions == [("agent_action", action)], (
        "the client was not told the next utterance is assistance"
    )


async def test_plain_answers_set_no_assistance_marker() -> None:
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("my answer", {"turn_key": "tk-abcd1234"}))

    assert sess.userdata.pending_assistant_kind is None
    assert [e for e in _events(local) if e["status"] == "agent_action"] == []


async def test_unknown_turn_action_is_rejected_and_never_replied_to() -> None:
    """Rejects rather than coercing to 'answer' — coercion is the scoring bug."""
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("hi", {"turn_action": "sudo_pass_me", "turn_key": "tk-abcd1234"}))

    events = _events(local)
    assert [e["status"] for e in events] == ["rejected"]
    assert events[0]["rejection"] == tp.TurnRejection.INVALID_TURN_ACTION.value
    assert events[0]["turn_key"] == "tk-abcd1234", "a rejection must be correlatable"
    assert sess.replies == []
    assert agent.folded == []


async def test_empty_text_is_rejected() -> None:
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("   ", {"turn_key": "tk-abcd1234"}))

    assert [e["status"] for e in _events(local)] == ["rejected"]
    assert sess.replies == []


async def test_grading_failure_still_produces_a_reply() -> None:
    """A broken grader must cost the grade, not the candidate's answer."""
    sess, agent, local, cb = _harness()

    async def _boom(_answer: str) -> None:
        raise RuntimeError("probe gateway down")

    agent.on_fold = _boom

    await cb(sess, _Event("my answer", {"turn_key": "tk-abcd1234"}))

    assert sess.replies == [{"user_input": "my answer"}]
    assert [e["status"] for e in _events(local)] == ["accepted"]


async def test_the_sdk_default_three_steps_still_run() -> None:
    """Overriding the callback must not bypass the LLM — it never did."""
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("answer", {"turn_key": "tk-abcd1234"}))

    assert sess.claimed == 1, "the user turn must be claimed"
    assert sess.interrupted is True
    assert sess.replies, "generate_reply must still be called"


async def test_typed_answer_cuts_through_a_non_interruptible_speech() -> None:
    """A typed answer must interrupt the opening / rejoin re-read by force.

    Those speeches deliberately run ``allow_interruptions=False``; a plain
    interrupt() raises on them and the exception killed the whole reply — the
    candidate's turn was graded but never answered (production: "Sending your
    answer…" spinning while the opening was still playing).
    """
    sess, agent, local, cb = _harness()

    await cb(sess, _Event("my answer", {"turn_key": "tk-abcd1234"}))

    assert sess.interrupt_forced is True, "the interrupt must be forced"
    assert sess.replies == [{"user_input": "my answer"}]
