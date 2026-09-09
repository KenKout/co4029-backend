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

from abridgeai.features.interviews.realtime import native_typed_turn
from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import text_protocol as tp
from abridgeai.features.interviews.realtime.native_turn_intake import (
    TurnIntake,
    TypedTurnIntakeError,
)
from abridgeai.features.interviews.realtime.native_typed_turn import TypedTurnStore

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
    receipts: TypedTurnStore | None = None,
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

    ``receipts`` is the durable typed-turn store. Optional so the unit tests of
    the pure intake/ledger behaviour can run without a database; the runtime
    wiring ALWAYS passes one. With a store wired, the ack is published only
    AFTER the receipt commits: it means "your answer is durable", not "in RAM".
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

        # Begin BEFORE any await: the closing gate and the single-flight
        # reservation must be decided synchronously, or a finish racing the
        # reserve could slip between the reserve and the receipt.
        try:
            reserved = turn_intake.begin()
        except TypedTurnIntakeError:
            await publisher.reject(
                turn_key=turn.turn_key,
                turn_action=turn.turn_action,
                rejection=tp.TurnRejection.SESSION_CLOSING,
            )
            return

        # A resend of a key we have already taken is the SAME turn arriving twice
        # — the client lost the ack (a reconnect mid-turn is the normal cause) and
        # retried with its idempotency key, exactly as the protocol invites it to.
        #
        # The MEMORY ledger is only the fast path. When a durable store is wired,
        # the RECEIPT is the source of truth for what "already done" means: an
        # applied receipt is re-acked (never re-graded), but a FAILED receipt
        # must fall through and be folded again — short-circuiting on the ledger
        # alone left a failed fold permanently un-retryable in-process.
        if not turn_intake.claim(turn.turn_key):
            if receipts is not None and turn.turn_key:
                outcome = await _duplicate_receipt_outcome(
                    receipts, sess, turn, turn_intake, publisher
                )
                if outcome == "handled":
                    return
                if outcome == "retry-failed":
                    # The durable receipt says FAILED: fall through and process
                    # this retry like a first delivery (the receipt path
                    # re-acks durably and folds again).
                    async with reserved:
                        should_reply = await _process_answer_turn(
                            sess, turn, publisher, turn_intake, receipts
                        )
                        if should_reply:
                            await _reply(sess, turn, publisher)
                    return
            obs.emit(
                obs.EV_TEXT_TURN_DUPLICATE,
                session_id=_session_id(sess),
                turn_action=turn.turn_action,
            )
            logger.info("duplicate typed turn re-acked, not re-graded (key=%s)", turn.turn_key)
            await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
            return

        async with reserved:
            if turn.turn_action != tp.DEFAULT_TURN_ACTION:
                await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
                await _reply(sess, turn, publisher)
                return
            should_reply = await _process_answer_turn(
                sess, turn, publisher, turn_intake, receipts
            )
            if should_reply:
                await _reply(sess, turn, publisher)

    return _on_text_input


async def _duplicate_receipt_outcome(
    receipts: TypedTurnStore,
    sess: Any,  # noqa: ANN401 - see _on_text_input
    turn: tp.InboundTurn,
    turn_intake: TurnIntake,
    publisher: ControlPublisher,
) -> str:
    """Resolve a ledger-duplicate through the durable receipt.

    Returns "handled" (the callback is done) or "retry-failed" (fall through to
    a full re-process of the failed receipt).

    applied → re-ack (+ confirm), done. A FAILED receipt → False: the caller
    falls through to the normal receipt path, which re-acks durably and folds
    again. The ledger claim is released for the fall-through case so the retry
    re-enters cleanly.
    """
    if turn.turn_key is None:
        return "handled"
    row = await receipts.lookup(
        session_id=sess.userdata.interview_session_id,
        turn_key=turn.turn_key,
    )
    if row is None:
        return "handled"
    state = native_typed_turn.receipt_state(row)
    if state == "applied":
        obs.emit(
            obs.EV_TYPED_TURN_APPLIED,
            session_id=_session_id(sess),
            turn_action="resume",
        )
        await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
        return "handled"
    if state == "failed":
        obs.emit(
            obs.EV_TYPED_TURN_RESUMED,
            session_id=_session_id(sess),
            turn_action="failed-retry",
        )
        # Let the standard path re-process: it loads the SAME receipt, re-acks
        # durably, folds strictly and settles honestly.
        turn_intake.release_claim(turn.turn_key)
        return "retry-failed"
    # A RECEIVED receipt (still owned or lease-live): the original duplicate
    # semantics stand — re-ack, never re-fold.
    return "handled"


async def _process_answer_turn(
    sess: Any,  # noqa: ANN401 - see _on_text_input
    turn: tp.InboundTurn,
    publisher: ControlPublisher,
    turn_intake: TurnIntake,
    receipts: TypedTurnStore | None,
) -> bool:
    """The durable-receipt answer path. Returns True when the model must reply.

    Order of truth: receipt committed (the ACK), fold landed (applied + the
    confirming snapshot), fold failed (receipt failed + a turn-scoped FAILED —
    no confirmation, no reply). Anything else would let the client clear a
    draft whose fold never landed, or lock a composer behind a dead fold.
    """
    receipt_row = None
    if receipts is not None:
        try:
            receipt_row, _created = await receipts.persist(
                session_id=sess.userdata.interview_session_id,
                session_question_id=None,
                bank_question_id=_current_bank_question_id(sess),
                text=turn.text,
                turn_key=turn.turn_key or "",
            )
        except Exception:  # noqa: BLE001 - no durable receipt means NO ack
            obs.emit(
                obs.EV_TYPED_RECEIPT_FAILED,
                session_id=_session_id(sess),
                turn_action=turn.turn_action,
            )
            logger.exception(
                "typed receipt failed; turn NOT acked (key=%s)", turn.turn_key
            )
            await publisher.reject(
                turn_key=turn.turn_key,
                turn_action=turn.turn_action,
                rejection=tp.TurnRejection.SERVER_ERROR,
            )
            return False

    # A FAILED receipt from an earlier fold attempt: reclaim it under a fresh
    # token so THIS caller owns the retry. Reclaim refused (another caller won)
    # → ack durably and stand down: the receipt is being retried elsewhere.
    if (
        receipt_row is not None
        and receipts is not None
        and turn.turn_key
        and native_typed_turn.receipt_state(receipt_row) == "failed"
    ):
        receipt_row, reclaimed = await receipts.reclaim(
            session_id=sess.userdata.interview_session_id,
            turn_key=turn.turn_key,
        )
        if not reclaimed:
            obs.emit(
                obs.EV_TEXT_TURN_DUPLICATE,
                session_id=_session_id(sess),
                turn_action=turn.turn_action,
            )
            await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
            return False

    # Durable BEFORE ack. If the receipt was already applied (a restart between
    # commit and the client settling), this is the re-ack path — no fold, no
    # echo marker, no reply.
    if (
        receipt_row is not None
        and native_typed_turn.receipt_state(receipt_row) == "applied"
    ):
        obs.emit(
            obs.EV_TYPED_TURN_APPLIED,
            session_id=_session_id(sess),
            turn_action="resume",
        )
        await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)
        return False

    # Arm the one-shot echo marker BEFORE the ack, so the SDK's
    # conversation_item_added copy of this text can be recognized and dropped —
    # the receipt already IS the transcript row.
    if receipt_row is not None and turn.turn_key:
        turn_intake.arm_echo(turn_key=turn.turn_key, text=turn.text)
        userdata = getattr(sess, "userdata", None)
        if userdata is not None:
            userdata.echo_turn_key = turn.turn_key

    # Ack AFTER the durable receipt. Grading still runs after it and can take
    # seconds; the composer must not sit spinning behind it.
    await publisher.ack(turn_key=turn.turn_key, turn_action=turn.turn_action)

    if receipt_row is None:
        # No store wired (diagnostic harness): old ordering, best-effort fold.
        await _fold_typed_answer(sess, turn.text, turn_key=turn.turn_key)
        return True

    # STRICT typed fold: a persistence failure here must NOT be swallowed into
    # an applied receipt — the candidate's draft stays retryable and the
    # receipt goes failed, not applied.
    try:
        await _fold_typed_answer(sess, turn.text, turn_key=turn.turn_key, strict=True)
    except Exception as exc:  # noqa: BLE001 - converted to FAILED below
        await _mark_receipt_failed(receipts, receipt_row, turn, publisher, sess, exc)
        return False
    await _mark_receipt_applied(receipts, receipt_row, turn_intake, turn, publisher, sess)
    return True


async def _mark_receipt_applied(
    receipts: TypedTurnStore | None,
    receipt_row: Any,  # noqa: ANN401 - store's own row
    turn_intake: TurnIntake,
    turn: tp.InboundTurn,
    publisher: ControlPublisher,
    sess: Any,  # noqa: ANN401 - see _on_text_input
) -> None:
    """Settle the receipt received→applied after the fold's own commit.

    The fold persists runtime state through its own writer; this marks the
    RECEIPT applied. A crash in between is safe by construction: the retry
    re-folds (idempotent by turn_key) and re-settles — the candidate's answer
    is never graded twice because the ledger + ``last_turn_idempotency_key``
    dedupe, and never lost because the receipt row survives.

    Once applied, the client is told via an ACKNOWLEDGED snapshot
    (``confirmed_turn_key=K``): that is what lets it drop the parked sent-draft.
    The ack itself deliberately does not — an ack only ever meant "durable",
    and "durable" has always been true the moment the receipt committed.
    """
    if receipts is None or turn.turn_key is None:
        return
    try:
        applied = await receipts.mark_applied(receipt_row)
    except Exception:  # noqa: BLE001 - settle is best-effort; the ledger dedupes
        logger.exception("marking typed receipt applied failed (key=%s)", turn.turn_key)
        obs.emit(
            obs.EV_TYPED_TURN_RESUMED,
            session_id=_session_id(sess),
            turn_action="settle-failed",
        )
        return
    if applied:
        obs.emit(
            obs.EV_TYPED_TURN_APPLIED,
            session_id=_session_id(sess),
            turn_action=turn.turn_action,
        )
        try:
            await publisher.acknowledge(turn_key=turn.turn_key)
        except Exception:  # noqa: BLE001 - a lost confirmation self-heals on retry
            logger.exception("publishing the confirmed snapshot failed (key=%s)", turn.turn_key)


async def _mark_receipt_failed(
    receipts: TypedTurnStore | None,
    receipt_row: Any,  # noqa: ANN401 - store's own row
    turn: tp.InboundTurn,
    publisher: ControlPublisher,
    sess: Any,  # noqa: ANN401 - see _on_text_input
    exc: Exception,
) -> None:
    """A failed fold: receipt → failed, turn-scoped FAILED, nothing confirmed.

    The parked sent-draft survives on the client (no confirmed_turn_key was
    ever published), the receipt stays retryable (a resend with the SAME
    turn_key reclaims the failed receipt and folds again), and the candidate
    gets an honest "your answer could not be processed" instead of a fake
    success. Only an allowlisted error class is emitted.
    """
    error_class = _safe_error_class(exc)
    obs.emit(
        obs.EV_TYPED_TURN_FOLD_FAILED,
        session_id=_session_id(sess),
        turn_action=turn.turn_action,
        error_class=error_class,
    )
    logger.exception("typed turn fold failed; receipt marked failed (key=%s)", turn.turn_key)
    if receipts is None:
        return
    try:
        await receipts.mark_failed(receipt_row, error_class=error_class)
    except Exception:  # noqa: BLE001 - best-effort bookkeeping after a failure
        logger.exception("marking typed receipt failed also failed (key=%s)", turn.turn_key)
    try:
        await publisher.fail(
            turn_key=turn.turn_key,
            turn_action=turn.turn_action,
            error_class=error_class,
        )
    except Exception:  # noqa: BLE001 - best-effort notification
        logger.exception("publishing the turn FAILED event failed (key=%s)", turn.turn_key)


_SAFE_ERROR_CLASSES = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "RuntimeError",
        "ValueError",
        "IntegrityError",
        "OperationalError",
        "StaleStateError",
        "TypedTurnReceiptError",
        "TypedTurnFoldError",
    }
)


def _safe_error_class(exc: Exception) -> str:
    """Allowlist the exception's class so no prompt/DB detail reaches the wire."""
    name = type(exc).__name__
    return name if name in _SAFE_ERROR_CLASSES else "InternalError"


class TypedTurnFoldError(RuntimeError):
    """The agent has no fold_turn; a strict typed fold cannot be processed."""


def _current_bank_question_id(sess: Any) -> Any:  # noqa: ANN401 - see _on_text_input
    """The bank question the answer folds against, at receipt time."""
    state = getattr(getattr(sess, "userdata", None), "state", None)
    return getattr(state, "current_question_id", None) if state else None


async def _fold_typed_answer(
    sess: Any,  # noqa: ANN401 - see _on_text_input
    text: str,
    *,
    turn_key: str | None,
    strict: bool = False,
) -> None:
    """Run the graded fold a spoken turn gets from ``on_user_turn_completed``.

    No chat-context handling: the state note lives in the agent's SYSTEM
    instructions, so ``fold_turn`` refreshes it directly and there is no per-turn
    copy to mutate and write back.

    ``turn_key`` is forwarded so the fold can persist it as the session's last
    processed turn. Without that, the only duplicate protection is this process's
    in-memory ledger, which an agent restart empties — and a client that reconnects
    to a NEW agent process and retries would be graded again.

    When ``strict`` is True (the durable-receipt path) the exception PROPAGATES:
    the caller marks the receipt failed and the client keeps a retryable draft —
    a swallowed error would let the fold be recorded as applied and confirmed
    while the state/coverage never landed. The spoken path stays best-effort
    (strict=False): there is no receipt there, and a failure must not cost the
    candidate their reply.
    """
    agent = sess.current_agent
    fold = getattr(agent, "fold_turn", None)
    if fold is None:
        if strict:
            raise TypedTurnFoldError("agent has no fold_turn")
        logger.warning("typed turn on an agent with no fold_turn; answer not graded")
        return
    if strict:
        await fold(answer_text=text, turn_key=turn_key)
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
