"""Wire protocol for typed interview turns over LiveKit text streams.

Three topics, each with one job:

* ``lk.chat`` (inbound, SDK-standard) — the candidate's typed text. Carries
  ``turn_action`` and ``turn_key`` as stream ATTRIBUTES; the text body is only
  ever the answer itself.
* ``lk.transcription`` (outbound, SDK-standard) — the agent's presentation
  (spoken transcript + what it says). Published by RoomIO, not by us.
* :data:`TOPIC_CONTROL` (outbound, ours) — the structured turn state the REST
  ``/respond`` endpoint returns today, so a typed client learns whether its turn
  was accepted, what the next question is, and whether the session finished.

Why a third topic at all: ``lk.chat`` is the user's text and
``lk.transcription`` is presentation. Neither can carry "this turn was rejected
because another is in flight" or "the session is now finished, here is
state_version 7" without overloading a channel the SDK also writes to. Keeping
control separate means a client can ignore it and still render a transcript, or
consume it and get exact state.

SECURITY: every attribute here arrives from the browser and is untrusted.
``session_id`` and ``student_id`` are deliberately NOT part of the inbound
contract — they come from the job's dispatch metadata (minted server-side into
the join token), never from the client. :func:`parse_inbound_attributes`
validates and normalises everything else, and rejects rather than coerces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

# Inbound: the candidate's typed text. SDK-standard topic (livekit.agents
# `TOPIC_CHAT`); RoomIO already routes this to our `text_input_cb`.
TOPIC_CHAT: Final = "lk.chat"

# Outbound: agent presentation. SDK-standard (`TOPIC_TRANSCRIPTION`), published
# by RoomIO's text output. Listed for completeness — we never write it directly.
TOPIC_TRANSCRIPTION: Final = "lk.transcription"

# Outbound: our structured turn state, correlated to an inbound turn by
# `turn_key`. Application-namespaced so it can never collide with an `lk.*`
# topic the SDK may add later.
TOPIC_CONTROL: Final = "abridge.interview.control"

# Attribute names on an inbound `lk.chat` stream.
ATTR_TURN_ACTION: Final = "turn_action"
ATTR_TURN_KEY: Final = "turn_key"

# The turn actions the interview brain understands. MUST stay in sync with
# `services.taking.take_session_step`'s `turn_action` parameter and the
# frontend's `InterviewTurnAction` union.
VALID_TURN_ACTIONS: Final = frozenset({"answer", "repeat", "clarify", "explain_term", "hint"})
DEFAULT_TURN_ACTION: Final = "answer"

# A turn_key is an idempotency key generated client-side (crypto.randomUUID or a
# `tk-<ts>-<rand>` fallback). We do not require a UUID — the fallback is not one
# — but we do bound the shape so it cannot be used to smuggle data into logs or
# the DB column.
_TURN_KEY_RE: Final = re.compile(r"\A[A-Za-z0-9_-]{8,128}\Z")

# Hard cap on a typed answer. Long-but-real answers must survive; this only stops
# a client streaming megabytes into an LLM prompt. Mirrors the REST validator.
MAX_TEXT_CHARS: Final = 8_000


class TurnRejection(StrEnum):
    """Why a typed turn was refused before it reached the brain."""

    EMPTY_TEXT = "empty_text"
    TEXT_TOO_LONG = "text_too_long"
    INVALID_TURN_ACTION = "invalid_turn_action"
    INVALID_TURN_KEY = "invalid_turn_key"
    TURN_IN_FLIGHT = "turn_in_flight"
    SESSION_CLOSING = "session_closing"


class ControlStatus(StrEnum):
    """Lifecycle of one typed turn, as seen by the client."""

    # The turn passed validation and is being processed. Lets the composer show
    # "sending" state driven by the server rather than by a local optimistic flag.
    ACCEPTED = "accepted"
    # The brain finished. Carries the same structured state REST `/respond`
    # returns (next question, finished flag, ...).
    COMPLETED = "completed"
    # Refused before processing; `rejection` says why. The client keeps the draft.
    REJECTED = "rejected"
    # Processing raised. The client keeps the draft and may retry with the SAME
    # turn_key — `take_session_step` is idempotent on it.
    FAILED = "failed"


class InboundTurnError(ValueError):
    """Raised when inbound stream attributes are malformed."""

    def __init__(self, rejection: TurnRejection, detail: str = "") -> None:
        super().__init__(detail or rejection.value)
        self.rejection = rejection


@dataclass(frozen=True)
class InboundTurn:
    """A validated typed turn. All fields are safe to pass to the brain."""

    text: str
    turn_action: str
    turn_key: str | None


def parse_inbound_attributes(
    text: str,
    attributes: dict[str, str] | None,
) -> InboundTurn:
    """Validate an inbound `lk.chat` turn.

    Rejects rather than coerces: an unrecognised ``turn_action`` is an error, not
    silently downgraded to ``"answer"``. Silently downgrading is exactly the
    scoring bug this protocol exists to avoid — a "give me a hint" request
    treated as an answer gets graded as one.

    ``session_id`` / ``student_id`` are intentionally absent: they are taken from
    the job's dispatch metadata, never from the client.

    :raises InboundTurnError: with a :class:`TurnRejection` the caller reports on
        the control topic.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise InboundTurnError(TurnRejection.EMPTY_TEXT)
    if len(cleaned) > MAX_TEXT_CHARS:
        raise InboundTurnError(
            TurnRejection.TEXT_TOO_LONG,
            f"text exceeds {MAX_TEXT_CHARS} chars",
        )

    attrs = attributes or {}

    raw_action = attrs.get(ATTR_TURN_ACTION)
    if raw_action is None or raw_action == "":
        turn_action = DEFAULT_TURN_ACTION
    elif raw_action in VALID_TURN_ACTIONS:
        turn_action = raw_action
    else:
        raise InboundTurnError(
            TurnRejection.INVALID_TURN_ACTION,
            # Value is echoed for debuggability but the set is closed, so this
            # cannot reflect arbitrary client text back into the DB.
            f"unknown turn_action: {raw_action[:32]!r}",
        )

    raw_key = attrs.get(ATTR_TURN_KEY)
    if raw_key is None or raw_key == "":
        # Absent is allowed: the brain generates its own correlation id. The
        # client then loses idempotency-on-retry, which is its choice to make.
        turn_key = None
    elif _TURN_KEY_RE.match(raw_key):
        turn_key = raw_key
    else:
        raise InboundTurnError(
            TurnRejection.INVALID_TURN_KEY,
            "turn_key must be 8-128 chars of [A-Za-z0-9_-]",
        )

    return InboundTurn(text=cleaned, turn_action=turn_action, turn_key=turn_key)


@dataclass
class ControlEvent:
    """One outbound control message.

    Two distinct counters, deliberately: ``seq`` is the agent's control-stream
    sequence (always present, orders THIS stream) and ``state_version`` is the
    interview brain's own version (only on COMPLETED, reconciles against
    persisted history). Timestamps are unusable for either — client clocks skew
    and LiveKit does not guarantee delivery order across streams.
    """

    status: ControlStatus
    turn_key: str | None
    # The AGENT's control-stream sequence. Always present; strictly increasing per
    # session. Use this to order control events and discard a stale one.
    seq: int
    turn_action: str = DEFAULT_TURN_ACTION
    # The BRAIN's per-session state version, present only on COMPLETED (earlier
    # statuses are emitted before the brain has run). This is the value to
    # reconcile persisted history against — `seq` orders the stream, this
    # identifies the interview state.
    state_version: int | None = None
    rejection: TurnRejection | None = None
    # Present on COMPLETED. Mirrors the REST `/respond` response body so a typed
    # client needs no extra round-trip to learn the new state.
    state: dict[str, Any] = field(default_factory=dict)
    # Present on FAILED. An allowlisted class name, never a raw exception
    # message — those can contain prompt or DB detail.
    error_class: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "turn_key": self.turn_key,
            "seq": self.seq,
            "turn_action": self.turn_action,
        }
        if self.state_version is not None:
            payload["state_version"] = self.state_version
        if self.rejection is not None:
            payload["rejection"] = self.rejection.value
        if self.state:
            payload["state"] = self.state
        if self.error_class is not None:
            payload["error_class"] = self.error_class
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


__all__ = [
    "ATTR_TURN_ACTION",
    "ATTR_TURN_KEY",
    "DEFAULT_TURN_ACTION",
    "MAX_TEXT_CHARS",
    "TOPIC_CHAT",
    "TOPIC_CONTROL",
    "TOPIC_TRANSCRIPTION",
    "VALID_TURN_ACTIONS",
    "InboundTurnError",
    "ControlEvent",
    "ControlStatus",
    "InboundTurn",
    "TurnRejection",
    "parse_inbound_attributes",
]
