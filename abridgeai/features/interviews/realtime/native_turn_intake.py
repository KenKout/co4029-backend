"""Idempotency and in-flight tracking for typed turns on the native path.

Two guarantees, both about the window between "the agent has your text" and "the
answer is durable", which on this path is seconds of LLM grading wide.

Idempotency
-----------
``turn_key`` is the client's idempotency key. It arrived on the ``lk.chat``
attributes and was used only to correlate the ack — never passed into the graded
fold or persisted. So a candidate whose ack was lost to a reconnect and who
resent the SAME key had the turn graded a second time: coverage points applied
twice, a second follow-up charged, and (because the server may advance between
the two) the resend could be folded against the NEXT question, grading an answer
to Q1 as an answer to Q2.

The routed path already had the mechanism — ``repository.save`` writes
``last_turn_idempotency_key`` and ``repository.is_duplicate_turn`` reads it — so
this is that contract applied to typed turns, plus an in-process ledger. Both
layers are needed and cover different failures:

* the ledger catches a resend while the first copy is STILL GRADING, where
  nothing has been persisted yet and the DB check would happily let it through.
  asyncio is single-threaded, so the check-and-insert here cannot interleave.
* ``last_turn_idempotency_key`` catches a resend that arrives after an agent
  restart, where the ledger is empty. ``seed`` loads it at setup.

Finish barrier
--------------
``TranscriptWriteBarrier`` waits for transcript writes it was HANDED. A turn that
is still inside its grading call has handed it nothing yet, so a finish (the
model ending, or the hard stop) could submit while an answer was mid-flight and
the evaluator would grade a transcript missing the candidate's last answer.
:meth:`drain` closes that: it waits for turns that have been accepted but not yet
recorded.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# How many keys to remember in-process. A turn_key is at most 128 chars and an
# interview is tens of turns, so this is generous; the cap only exists so a
# client looping forever cannot grow the ledger without bound.
_LEDGER_CAPACITY = 512

# Ceiling on how long a finish waits for turns still being graded. The grading
# probe is ~0.7s and the fold does a little DB work after it, so this is several
# times the expected cost — long enough that a slow turn still makes the
# transcript, short enough that a wedged turn cannot hang the job's teardown.
DRAIN_TIMEOUT_S = 10.0


class TurnLedger:
    """The turn keys this session has already accepted.

    Insertion-ordered so the oldest key is evicted first: a resend follows its
    original closely, so recent keys are the ones worth remembering.
    """

    def __init__(self, *, capacity: int = _LEDGER_CAPACITY) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._capacity = capacity

    def seed(self, turn_key: str | None) -> None:
        """Adopt a key persisted by an earlier run of this session.

        Called at setup with ``last_turn_idempotency_key``, so a turn the
        previous agent process graded before dying is still recognised as a
        duplicate by the process that replaces it.
        """
        if turn_key:
            self._remember(turn_key)

    def claim(self, turn_key: str | None) -> bool:
        """Take ownership of ``turn_key``. False when it was already claimed.

        A ``None`` key can never be a duplicate: the client opted out of
        idempotency for this turn, which is its choice to make (the protocol
        allows an absent key). It is also not remembered — there is nothing to
        match a later turn against.
        """
        if turn_key is None:
            return True
        if turn_key in self._seen:
            return False
        self._remember(turn_key)
        return True

    def _remember(self, turn_key: str) -> None:
        self._seen[turn_key] = None
        self._seen.move_to_end(turn_key)
        while len(self._seen) > self._capacity:
            self._seen.popitem(last=False)


class TurnIntake:
    """Accepts typed turns at most once, and lets a finish wait for them.

    One instance per session, owned by the text-input callback closure.
    """

    def __init__(self, *, ledger: TurnLedger | None = None) -> None:
        self._ledger = ledger if ledger is not None else TurnLedger()
        self._in_flight = 0
        # Set while nothing is in flight, so `drain` on an idle session is free.
        self._idle = asyncio.Event()
        self._idle.set()

    def seed(self, turn_key: str | None) -> None:
        self._ledger.seed(turn_key)

    def claim(self, turn_key: str | None) -> bool:
        return self._ledger.claim(turn_key)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _enter(self) -> None:
        self._in_flight += 1
        self._idle.clear()

    def _exit(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        if self._in_flight == 0:
            self._idle.set()

    def processing(self) -> _ProcessingScope:
        """Mark one turn as being processed for as long as the scope is held."""
        return _ProcessingScope(self)

    async def drain(self, *, timeout_seconds: float = DRAIN_TIMEOUT_S) -> bool:
        """Wait until no turn is mid-processing. True when the session went idle.

        Bounded and non-cancelling: a turn wedged on a hung gateway call must not
        stop the session being submitted, and the transcript barrier that runs
        after this still waits for whatever writes did land.
        """
        if self._in_flight == 0:
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_seconds)
        except TimeoutError:
            logger.error(
                "typed turns still processing at finish (in_flight=%d, timeout_seconds=%s)",
                self._in_flight,
                timeout_seconds,
            )
            return False
        return True


class _ProcessingScope:
    """Async context manager form of :meth:`TurnIntake.processing`."""

    __slots__ = ("_intake",)

    def __init__(self, intake: TurnIntake) -> None:
        self._intake = intake

    async def __aenter__(self) -> None:
        self._intake._enter()  # noqa: SLF001 -- same-module collaborator

    async def __aexit__(self, *_exc: object) -> None:
        self._intake._exit()  # noqa: SLF001 -- same-module collaborator


__all__ = ["DRAIN_TIMEOUT_S", "TurnIntake", "TurnLedger"]
