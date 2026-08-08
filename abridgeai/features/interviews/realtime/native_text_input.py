"""Typed-turn intake for the NATIVE interview agent.

The SDK's default text callback (``room_io._types._default_text_input_cb``) does
two things this interview cannot accept:

1. **It discards the stream attributes.** ``turn_action`` and ``turn_key`` ride on
   the ``lk.chat`` stream as attributes (:mod:`text_protocol`), and the default
   reads only ``ev.text``. A typed "give me a hint" would therefore be graded as
   an attempt at the answer — the exact scoring bug the protocol exists to stop.

2. **It never reaches the graded path.** ``Agent.on_user_turn_completed`` is
   invoked only from ``AgentActivity.on_end_of_turn``, whose sole caller is
   ``audio_recognition``. It is an STT-path hook. So on the default callback a
   typed turn is never graded, never refreshes the state note, never persists
   runtime state and never emits a shadow verdict — and for a ``text`` session,
   where audio is disabled at the room boundary, that is EVERY turn.

Overriding the callback does NOT bypass the LLM. The routed path's bypass came
from having no ``llm=`` plus ``StopResponse``; here the same three steps the SDK
default performs (claim the turn, interrupt, ``generate_reply``) still run, with
the attributes parsed and the graded fold performed first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import text_protocol as tp

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from abridgeai.features.interviews.realtime.native_control import ControlPublisher

logger = logging.getLogger(__name__)

# The tool that debits the server-side hint ladder. Restricting a `hint` turn to
# this one tool is what keeps the ladder honest: a model free-handing a hint
# without the tool call grants a hint the server never counted.
_HINT_TOOL = "interview_request_hint"

# Framing for the non-answer actions. These need no tool — they are conversation
# — but the model must know the text is a request for help rather than an attempt,
# or it grades and moves on. Kept in English: this steers the model, while the
# language it REPLIES in is fixed by `agent_instructions.build_instructions`.
_ACTION_INSTRUCTIONS: dict[str, str] = {
    "repeat": (
        "The candidate asked you to repeat the current question. Say it again, "
        "rephrased more simply. Do not treat this as an answer and do not advance."
    ),
    "clarify": (
        "The candidate does not understand the current question. Explain what the "
        "question is asking, without answering it. Do not treat this as an answer "
        "and do not advance."
    ),
    "explain_term": (
        "The candidate asked what a term in the current question means. Define it "
        "plainly, without answering the question. Do not treat this as an answer "
        "and do not advance."
    ),
}


def make_text_input_cb(
    publisher: ControlPublisher,
) -> Callable[[Any, Any], Awaitable[None]]:
    """Build the ``text_input_cb`` for a native session.

    Takes the publisher rather than reaching for it through the session so the
    callback can be wired into ``RoomInputOptions`` BEFORE ``session.start`` —
    which it must be, since the options are an argument to it. The publisher
    resolves the room lazily on each publish, so it tolerates being built early.
    """

    async def _on_text_input(sess: Any, ev: Any) -> None:  # noqa: ANN401 - SDK passes AgentSession/TextInputEvent; typing them here would import the SDK at module scope
        try:
            turn = tp.parse_inbound_attributes(ev.text, getattr(ev.info, "attributes", None))
        except tp.InboundTurnError as err:
            obs.emit(
                obs.EV_TEXT_TURN_REJECTED,
                session_id=_session_id(sess),
                rejection=err.rejection.value,
            )
            await publisher.reject(
                turn_key=_raw_turn_key(ev),
                turn_action=tp.DEFAULT_TURN_ACTION,
                rejection=err.rejection,
            )
            return

        # Ack BEFORE the fold. Grading calls an LLM and can take seconds; the
        # composer must not sit spinning behind it, and an ack means "received",
        # not "graded".
        await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)

        if turn.turn_action == tp.DEFAULT_TURN_ACTION:
            await _fold_typed_answer(sess, turn.text)

        await _reply(sess, turn)

    return _on_text_input


async def _fold_typed_answer(sess: Any, text: str) -> None:  # noqa: ANN401 - see _on_text_input
    """Run the graded fold a spoken turn gets from ``on_user_turn_completed``.

    ``agent.chat_ctx`` is a read-only view, so the SDK's prescribed
    copy → mutate → ``update_chat_ctx`` sequence is the only way to land the
    refreshed state note on the context the reply will actually be generated
    from.

    Never raises: a failure here must cost the grade, not the candidate's reply.
    """
    agent = sess.current_agent
    fold = getattr(agent, "fold_turn", None)
    if fold is None:
        logger.warning("typed turn on an agent with no fold_turn; answer not graded")
        return
    try:
        chat_ctx = agent.chat_ctx.copy()
        await fold(chat_ctx, answer_text=text)
        await agent.update_chat_ctx(chat_ctx)
    except Exception:  # noqa: BLE001 -- grading must never cost the reply
        logger.exception("typed turn fold failed")


async def _reply(sess: Any, turn: tp.InboundTurn) -> None:  # noqa: ANN401 - see _on_text_input
    """The SDK default's three steps, with the action's framing applied.

    ``generate_reply`` is deliberately NOT awaited: it returns a handle and
    awaiting it would hold the text-stream handler open for the whole spoken
    reply.
    """
    kwargs: dict[str, Any] = {"user_input": turn.text}
    if turn.turn_action == "hint":
        kwargs["tools"] = [_HINT_TOOL]
    elif (framing := _ACTION_INSTRUCTIONS.get(turn.turn_action)) is not None:
        kwargs["instructions"] = framing

    async with sess._claim_user_turn():  # noqa: SLF001 - the SDK's own default callback does this
        await sess.interrupt()
        sess.generate_reply(**kwargs)


def _session_id(sess: Any) -> Any:  # noqa: ANN401 - see _on_text_input
    userdata = getattr(sess, "userdata", None)
    return getattr(userdata, "interview_session_id", "unknown")


def _raw_turn_key(ev: Any) -> str | None:  # noqa: ANN401 - see _on_text_input
    """The client's turn_key for a REJECTED event, echoed back unvalidated.

    A rejection must be correlatable or the client cannot clear the right pending
    turn — including when the key itself is what failed validation. Bounded and
    never persisted, unlike the accepted path's key.
    """
    attributes = getattr(ev.info, "attributes", None) or {}
    raw = attributes.get(tp.ATTR_TURN_KEY)
    return str(raw)[:128] if raw else None


__all__ = ["make_text_input_cb"]
