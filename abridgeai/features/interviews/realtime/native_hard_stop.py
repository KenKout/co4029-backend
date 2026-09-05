"""The hard stop, the transcript barrier, and the deadline arithmetic.

Split out of :mod:`native_runtime` when that file crossed the feature's 800-LOC
gate. The seam is a real one rather than a size convenience: everything here is
about ENDING a session — the wall clock the model cannot argue with, the waits
that make the ending honest, and the once-only guard both finish routes share —
and none of it touches the conversation loop or the agent's tools.

Why a hard stop exists at all
-----------------------------
It is the third anti-deadlock layer. The other two are the bounded refusal
counters in ``orchestrator/tools.py`` (``MAX_END_REFUSALS`` /
``MAX_ADVANCE_REFUSALS``), and both only help a model that TRIES to advance or
end. A model that simply keeps talking defeats them, which is why this layer sits
outside the model's reach entirely.

What "ending honestly" means here, and what it cost to learn
-----------------------------------------------------------
* Everything the candidate said must be in the transcript BEFORE the submit —
  the evaluator grades that table. Two waits are needed and they are not
  interchangeable: ``drain_turns`` waits for typed turns still inside their
  grading call (which have created no write yet), and ``flush_transcript`` waits
  for the writes those turns produced.
* A submit can be REFUSED. ``services.taking.submit_session`` validates the
  finish reason against the config, so ``reason="timed_out"`` raises for a
  session with no ``time_limit_minutes``, and again when the stop fires before a
  configured deadline. Both used to be swallowed here, after which the finish was
  announced and the job shut down — leaving the row ``in_progress`` with nothing
  left to finish it. For an untimed session that is permanent, because the
  deadline sweep only selects sessions that have a limit. Hence
  ``close_fallback``, and hence: a finish that did not persist is NOT announced.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from livekit.agents import get_job_context

from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from uuid import UUID

    from livekit.agents import AgentSession

logger = logging.getLogger(__name__)

# Bound on waiting for the closing to finish playing out before teardown, so a
# stuck SpeechHandle cannot hang the job. Mirrors ``session_runtime``.
_CLOSING_PLAYOUT_TIMEOUT_S = 30.0

# Wall-clock budget the hard stop allows per remaining question when the config
# sets no time limit. Generous on purpose: this is a backstop against a model
# that never ends, not a pacing mechanism, and cutting a productive interview
# short is a worse failure than running a few minutes long.
_SECONDS_PER_REMAINING_QUESTION = 240.0
# Never arm the stop closer than this, even with no questions left: a session
# joined at the very end of its window still deserves a closing exchange rather
# than an immediate teardown.
_MIN_HARD_STOP_SECONDS = 120.0
# Fired past the deadline, not at it: `submit_session` refuses a `timed_out`
# reason until the limit has actually elapsed (with a 2s scheduling-tolerance
# buffer of its own), and a hard stop that fires early therefore cannot submit.
_DEADLINE_GRACE_SECONDS = 5.0
# Used only when neither a time limit nor a question budget is known. Long
# enough that no real interview trips it, short enough that an abandoned room
# still closes.
_DEFAULT_HARD_STOP_SECONDS = 45.0 * 60.0
_TRANSCRIPT_FLUSH_TIMEOUT_S = 3.0


def hard_stop_deadline_seconds(
    *, time_remaining_seconds: int | None, questions_remaining: int
) -> float:
    """Seconds until the server-side hard stop fires.

    The config's time limit is the authoritative deadline when one exists; the
    question budget only applies to an UNTIMED session. Using ``min`` of the two
    let a small question pool end a 30-minute interview after 8 — the reported
    session was killed mid-conversation, and the timed-out submission was then
    REJECTED because the real limit had not elapsed. Pure so the arithmetic is
    testable without a room.
    """
    if time_remaining_seconds is not None:
        return max(_MIN_HARD_STOP_SECONDS, float(time_remaining_seconds) + _DEADLINE_GRACE_SECONDS)
    budget = max(0.0, questions_remaining) * _SECONDS_PER_REMAINING_QUESTION
    if budget <= 0.0:
        return _DEFAULT_HARD_STOP_SECONDS
    return max(_MIN_HARD_STOP_SECONDS, budget)


async def _no_flush() -> None:
    return None


class TranscriptWriteBarrier:
    """Track native transcript writes until session finalization.

    Conversation callbacks cannot await ``record_turn`` directly. Before either
    native finalization path submits and enqueues evaluation, this barrier waits
    for every write task already scheduled, so the evaluator sees the final
    candidate answer.
    """

    def __init__(self) -> None:
        self._pending: set[asyncio.Task[None]] = set()

    def create(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush(self) -> None:
        """Wait briefly for writes in flight without cancelling them.

        Loop with a single deadline so a write task created WHILE the flush is
        already running (an SDK callback landing exactly as finalization
        begins) is also drained before submit. Writers still pending at the
        deadline are left to finish on their own — never cancelled — so a
        wedged DB session cannot hang the finalizer, and a merely slow write
        is not lost, only unawaited.
        """
        deadline = asyncio.get_running_loop().time() + _TRANSCRIPT_FLUSH_TIMEOUT_S
        while self._pending:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.error(
                    "transcript flush timed out (pending=%d, timeout_seconds=%s)",
                    len(self._pending),
                    _TRANSCRIPT_FLUSH_TIMEOUT_S,
                )
                return
            done, _still = await asyncio.wait(
                tuple(self._pending),
                timeout=remaining,
            )
            for task in done:
                try:
                    task.result()
                except Exception:  # noqa: BLE001 -- record_turn normally swallows failures
                    logger.exception("unexpected transcript task failure")


@dataclass
class HardStopPlan:
    """What the hard stop needs to close a session the model never ends.

    ``close`` submits the session and returns the canonical ceremony closing to
    speak — the same ``orchestration_bridge.finalize_session`` the
    ``interview_end_interview`` tool reaches, so both routes submit once, the
    same way.
    """

    deadline_seconds: float
    close: Callable[[], Awaitable[str | None]]
    interview_session_id: UUID
    flush_transcript: Callable[[], Awaitable[None]] = _no_flush
    # Wait for typed turns still being graded before submitting. Returns False when
    # some were still running at the timeout. The transcript barrier alone is not
    # enough: it waits for writes it was HANDED, and a turn inside its grading call
    # has handed it nothing yet, so a finish could submit without the last answer.
    drain_turns: Callable[[], Awaitable[bool]] | None = None
    # Last-resort finish for a submit that was refused. ``reason="timed_out"`` is
    # only legal on a session whose config HAS a time limit, and an untimed session
    # is excluded from the deadline sweep — so a refused hard-stop submit there left
    # the row `in_progress` with nothing anywhere that would ever finish it.
    close_fallback: Callable[[], Awaitable[str | None]] | None = None


class HardStopTimer:
    """The third anti-deadlock layer: a wall clock the model cannot argue with.

    The other two are the bounded refusal counters in ``orchestrator/tools.py``
    (``MAX_END_REFUSALS`` / ``MAX_ADVANCE_REFUSALS``), and both only help a model
    that TRIES to advance or end. A model that simply keeps talking defeats them,
    which is why this layer exists outside the model's reach entirely.

    :meth:`finalize_once` is the shared entry for both routes, so a session
    cannot be submitted twice when the model ends just as the timer fires.
    """

    def __init__(self, plan: HardStopPlan, *, session: AgentSession) -> None:
        self._plan = plan
        self._session = session
        self._task: asyncio.Task[None] | None = None
        self._finalized = False
        # Set by the runtime once the publisher exists. Runs after EITHER route
        # submits, so "the interview is over" reaches the client whether the model
        # ended it or the wall clock did.
        self.on_finalized: Callable[[], Awaitable[None]] | None = None

    @property
    def fired(self) -> bool:
        return self._finalized

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def finalize_once(self, inner: Callable[[], Awaitable[None]]) -> None:
        """Run ``inner`` at most once per session, and disarm the timer.

        Wrapping the tool's finalizer rather than guarding inside it keeps
        ``agent_tools`` unaware that a timer exists.
        """
        if self._finalized:
            return
        self._finalized = True
        self.cancel()
        # Typed turns first, THEN their transcript writes. A turn mid-grading has
        # not created its write yet, so flushing without draining submits a
        # transcript the candidate's last answer never reached.
        await self._drain_turns()
        await self._plan.flush_transcript()
        await inner()
        await self._announce_finished()

    async def _drain_turns(self) -> None:
        """Wait for typed turns still being graded. Never raises, always bounded."""
        if self._plan.drain_turns is None:
            return
        try:
            drained = await self._plan.drain_turns()
        except Exception:  # noqa: BLE001 -- a finish must not fail on the wait itself
            logger.exception(
                "waiting for in-flight turns failed (session=%s)",
                self._plan.interview_session_id,
            )
            return
        if not drained:
            obs.emit(
                obs.EV_TURN_DRAIN_TIMEOUT,
                session_id=self._plan.interview_session_id,
            )

    async def _close_via_fallback(self) -> tuple[str | None, bool]:
        """Retry the submit under a reason the session can legally be finished as.

        The primary reason is refused outright on a mismatch: ``submit_session``
        raises "Cannot time out an interview without a time limit" for
        ``reason="timed_out"`` on an untimed config, and "Interview time limit has
        not elapsed" when the hard stop fires early. Both used to end here — the
        exception was swallowed, the finish was announced anyway, and the job shut
        down leaving the row ``in_progress``. For an untimed session that is
        permanent: the deadline sweep only selects sessions WITH a time limit.
        """
        if self._plan.close_fallback is None:
            return None, False
        try:
            closing = await self._plan.close_fallback()
        except Exception:  # noqa: BLE001 -- nothing left to try; the caller stays live
            logger.exception(
                "hard stop fallback finalize also failed (session=%s)",
                self._plan.interview_session_id,
            )
            return None, False
        logger.warning(
            "hard stop finalized through the fallback reason (session=%s)",
            self._plan.interview_session_id,
        )
        return closing, True

    async def _announce_finished(self) -> None:
        """Tell the client the interview is over. Never raises."""
        if self.on_finalized is None:
            return
        try:
            await self.on_finalized()
        except Exception:  # noqa: BLE001 -- a submitted session must not fail on a notification
            logger.exception(
                "failed to announce finish (session=%s)", self._plan.interview_session_id
            )

    # How long to wait before retrying a stop whose submit could not be persisted.
    # Short enough that a session is not left live for long, long enough that a
    # transient database or gateway fault has a chance to clear — the retry costs a
    # submit attempt, not a grading pass.
    _RETRY_AFTER_FAILED_STOP_S = 30.0

    async def _run(self) -> None:
        """Sleep to the deadline, stop, and keep retrying a stop that could not land.

        The retry loop matters because ``_stop`` deliberately does NOT finalize when
        its submit was refused: it leaves the session live rather than announcing a
        finish that did not happen. Without a retry the timer would then be spent,
        so the row would stay ``in_progress`` until the model ended it or the
        deadline sweep did — and for an untimed session the sweep never will.
        """
        try:
            await asyncio.sleep(self._plan.deadline_seconds)
            while not self._finalized:
                await self._stop()
                if self._finalized:
                    return
                await asyncio.sleep(self._RETRY_AFTER_FAILED_STOP_S)
        except asyncio.CancelledError:
            return

    async def _stop(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Submit BEFORE speaking. The candidate's answers are already graded
        # evidence at this point, and a transport failure during the goodbye must
        # not be the reason a completed interview was never submitted.
        closing: str | None = None
        submitted = False
        try:
            await self._drain_turns()
            await self._plan.flush_transcript()
            closing = await self._plan.close()
            submitted = True
        except Exception:
            logger.exception(
                "hard stop failed to finalize (session=%s)",
                self._plan.interview_session_id,
            )
        if not submitted:
            closing, submitted = await self._close_via_fallback()
        obs.emit(
            obs.EV_CLOSING_EMITTED,
            session_id=self._plan.interview_session_id,
            reason="hard_stop_deadline",
            closing_chars=len(closing or ""),
        )
        # Only tell the client the interview is over once it actually IS. Announcing
        # a finish we failed to persist, then shutting the job down, is what left
        # sessions reading `in_progress` behind a completion screen with no agent
        # left to finish them. An un-submitted session keeps the job alive instead,
        # so the model or a rejoin can still end it.
        if not submitted:
            logger.error(
                "hard stop could not submit; leaving the session live rather than "
                "announcing a finish that did not happen (session=%s)",
                self._plan.interview_session_id,
            )
            self._finalized = False
            return
        await self._announce_finished()
        handle = None
        if closing:
            handle = self._session.say(closing, allow_interruptions=False)
            await handle
        await _await_playout(handle)
        job = get_job_context(required=False)
        if job is not None:
            job.shutdown(reason="interview_hard_stop")


async def _await_playout(handle: object) -> None:
    """Wait for a closing utterance to reach the candidate, bounded.

    Awaiting ``say()`` only means the speech was scheduled and generated, not
    that the audio arrived — tearing the room down here would cut the closing.
    Never raises: a stuck handle must not block shutdown.
    """
    waiter = getattr(handle, "wait_for_playout", None)
    if waiter is None:
        return
    try:
        await asyncio.wait_for(waiter(), timeout=_CLOSING_PLAYOUT_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - playout is best-effort; shutdown proceeds
        logger.warning("closing playout wait failed; shutting down anyway")


__all__ = [
    "HardStopPlan",
    "HardStopTimer",
    "TranscriptWriteBarrier",
    "hard_stop_deadline_seconds",
]
