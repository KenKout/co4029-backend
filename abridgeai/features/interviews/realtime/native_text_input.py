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
from abridgeai.features.interviews.realtime.native_turn_intake import TurnIntake

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
    *,
    intake: TurnIntake | None = None,
) -> Callable[[Any, Any], Awaitable[None]]:
    """Build the ``text_input_cb`` for a native session.

    Takes the publisher rather than reaching for it through the session so the
    callback can be wired into ``RoomInputOptions`` BEFORE ``session.start`` —
    which it must be, since the options are an argument to it. The publisher
    resolves the room lazily on each publish, so it tolerates being built early.

    ``intake`` carries the session's ``turn_key`` ledger and its in-flight count.
    It is optional only so a diagnostic harness can build a callback without one;
    a real session always passes it, because without it a resent turn is graded
    twice and a finish can submit while an answer is still being graded.
    """
    turn_intake = intake if intake is not None else TurnIntake()

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

        # A resend of a key we have already taken is the SAME turn arriving twice
        # — the client lost the ack (a reconnect mid-turn is the normal cause) and
        # retried with its idempotency key, exactly as the protocol invites it to.
        # Re-acking is the whole response: the client needs the settle signal it
        # missed, and re-grading would apply this answer's coverage points a
        # second time, charge another follow-up, and — because the server may have
        # advanced in between — fold an answer to the previous question against
        # the current one.
        if not turn_intake.claim(turn.turn_key):
            obs.emit(
                obs.EV_TEXT_TURN_DUPLICATE,
                session_id=_session_id(sess),
                turn_action=turn.turn_action,
            )
            logger.info("duplicate typed turn re-acked, not re-graded (key=%s)", turn.turn_key)
            await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
            return

        # Ack BEFORE the fold. Grading calls an LLM and can take seconds; the
        # composer must not sit spinning behind it, and an ack means "received",
        # not "graded".
        await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)

        # Held across the fold AND the reply handoff, so a finish that starts now
        # waits for this answer instead of submitting a transcript without it.
        async with turn_intake.processing():
            if turn.turn_action == tp.DEFAULT_TURN_ACTION:
                await _fold_typed_answer(sess, turn.text, turn_key=turn.turn_key)

            await _reply(sess, turn, publisher)

    return _on_text_input


async def _fold_typed_answer(
    sess: Any,  # noqa: ANN401 - see _on_text_input
    text: str,
    *,
    turn_key: str | None,
) -> None:
    """Run the graded fold a spoken turn gets from ``on_user_turn_completed``.

    No chat-context handling: the state note lives in the agent's SYSTEM
    instructions, so ``fold_turn`` refreshes it directly and there is no per-turn
    copy to mutate and write back.

    ``turn_key`` is forwarded so the fold can persist it as the session's last
    processed turn. Without that, the only duplicate protection is this process's
    in-memory ledger, which an agent restart empties — and a client that reconnects
    to a NEW agent process and retries would be graded again.

    Never raises: a failure here must cost the grade, not the candidate's reply.
    """
    agent = sess.current_agent
    fold = getattr(agent, "fold_turn", None)
    if fold is None:
        logger.warning("typed turn on an agent with no fold_turn; answer not graded")
        return
    try:
        await fold(answer_text=text, turn_key=turn_key)
    except Exception:  # noqa: BLE001 -- grading must never cost the reply
        logger.exception("typed turn fold failed")


async def _reply(
    sess: Any,  # noqa: ANN401 - see _on_text_input
    turn: tp.InboundTurn,
    publisher: ControlPublisher,
) -> None:
    """The SDK default's three steps, with the action's framing applied.

    ``generate_reply`` is deliberately NOT awaited: it returns a handle and
    awaiting it would hold the text-stream handler open for the whole spoken
    reply.

    For the assistance actions the reply IS assistance: the pending-kind marker
    and the client notice are set here, so the utterance is badged as
    clarification live and persisted as one across a reload. ``repeat`` is
    conversation rather than help, but it renders best in the same nested rail,
    so it shares the clarification kind.
    """
    userdata = getattr(sess, "userdata", None)
    kwargs: dict[str, Any] = {"user_input": turn.text}
    if turn.turn_action == "hint":
        kwargs["tools"] = [_HINT_TOOL]
    elif (framing := _ACTION_INSTRUCTIONS.get(turn.turn_action)) is not None:
        kwargs["instructions"] = framing
        if userdata is not None and turn.turn_action in _ASSISTANCE_KINDS:
            userdata.pending_assistant_kind = _ASSISTANCE_KINDS[turn.turn_action]
            await publisher.agent_action(kind=turn.turn_action)

    async with sess._claim_user_turn():  # noqa: SLF001 - the SDK's own default callback does this
        # force=True: the opening and the rejoin re-read deliberately run
        # with allow_interruptions=False, and a candidate typing an answer
        # while the question is being read must cut through them. A plain
        # interrupt() raises on those handles and the exception killed the
        # WHOLE reply — the candidate's turn was graded but never answered
        # (production: "Sending your answer…" spinning at the opening).
        try:
            await sess.interrupt(force=True)
        except RuntimeError:
            logger.warning("interrupt before typed reply failed; replying anyway")
        sess.generate_reply(**kwargs)


_ASSISTANCE_KINDS: dict[str, str] = {
    "repeat": "clarification",
    "clarify": "clarification",
    "explain_term": "clarification",
}


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
