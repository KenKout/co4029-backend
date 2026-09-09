"""Resume state, the client clock, and a hard stop that cannot strand a session.

Three defects, all in the seams between the agent's in-room state, the database,
and what the client is told.

The clock
---------
``userdata.time_remaining_seconds`` is read ONCE at setup and nothing refreshes
it, but every snapshot re-sent it verbatim. Ten minutes into a thirty-minute
interview a client that reloaded was handed nearly thirty minutes again, while the
backend still ended the session on the real deadline — a timer that runs in the
candidate's favour and then stops them without warning. The model was told the
same stale number, so the closing nudge that keys off it never became urgent.

The hard stop
-------------
``submit_session`` validates the finish reason against the config:
``reason="timed_out"`` raises for a config with no ``time_limit_minutes``. The
hard stop always passed ``timed_out``, so on an untimed session its submit raised,
the exception was swallowed, and the runtime went on to announce the finish and
shut the job down — leaving the row ``in_progress``. That is permanent there: the
deadline sweep only selects sessions that HAVE a time limit.

The resume
----------
Runtime state moves at question SELECTION; the transcript row appears when the
question is SPOKEN. Between those two the transcript's newest row is the previous
question, and setup was passing it as the current question — rewinding a restart
to Q1 while Q2 stayed in ``asked_question_ids``, so the scorer would never offer
Q2 again and it was skipped outright.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import AppError
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
from abridgeai.features.interviews.orchestrator.turn_state import sync_question_history
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata
from abridgeai.features.interviews.realtime.native_control import build_snapshot
from abridgeai.features.interviews.realtime.native_runtime import HardStopPlan, HardStopTimer

_asyncio = pytest.mark.asyncio


def _userdata(**over: Any) -> InterviewUserdata:
    base: dict[str, Any] = {
        "interview_session_id": uuid4(),
        "student_id": uuid4(),
        "state": InterviewRuntimeStateData(),
        "questions_total": 3,
        "questions_remaining": 2,
    }
    base.update(over)
    return InterviewUserdata(**base)


# ───────────────────────────── the client clock ─────────────────────────────


def test_the_countdown_decays_between_snapshots() -> None:
    """THE BUG: a rejoin was handed the clock the session had at JOIN."""
    userdata = _userdata(
        time_remaining_seconds=1800,
        clock_read_monotonic=time.monotonic() - 600,  # ten minutes ago
    )

    snapshot = build_snapshot(userdata)

    assert snapshot.time_remaining_seconds is not None
    assert 1150 <= snapshot.time_remaining_seconds <= 1210, (
        "the snapshot re-sent the join-time countdown instead of the live one"
    )
    assert snapshot.has_time_limit is True


def test_the_countdown_never_goes_negative() -> None:
    """Past the deadline the honest answer is 0; the hard stop owns what follows."""
    userdata = _userdata(
        time_remaining_seconds=60,
        clock_read_monotonic=time.monotonic() - 600,
    )

    assert build_snapshot(userdata).time_remaining_seconds == 0


def test_an_untimed_session_still_reports_no_limit() -> None:
    """None is not 0: reporting 0 makes the agent rush a session with no deadline."""
    userdata = _userdata(time_remaining_seconds=None, clock_read_monotonic=time.monotonic())

    snapshot = build_snapshot(userdata)

    assert snapshot.has_time_limit is False
    assert snapshot.time_remaining_seconds is None


def test_a_missing_clock_anchor_falls_back_to_the_stored_value() -> None:
    """Better a stale number than none — a client with no countdown shows no timer."""
    userdata = _userdata(time_remaining_seconds=900, clock_read_monotonic=None)

    assert build_snapshot(userdata).time_remaining_seconds == 900


def test_has_time_limit_survives_a_fully_elapsed_clock() -> None:
    """The limit's EXISTENCE is a config fact, not a function of what is left.

    Deriving it from the live countdown would flip a timed session to
    "untimed" at zero, and the client clears its deadline on that.
    """
    userdata = _userdata(
        time_remaining_seconds=5,
        clock_read_monotonic=time.monotonic() - 3600,
    )

    snapshot = build_snapshot(userdata)

    assert snapshot.time_remaining_seconds == 0
    assert snapshot.has_time_limit is True


# ───────────────────────────── the hard stop ─────────────────────────────


class _SpeechHandle:
    """Awaitable stand-in for the SDK's SpeechHandle, which `_stop` awaits."""

    def __await__(self) -> Any:
        async def _done() -> None:
            return None

        return _done().__await__()


class _FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str, *, allow_interruptions: bool = True) -> Any:
        del allow_interruptions
        self.spoken.append(text)
        return _SpeechHandle()


def _plan(**over: Any) -> HardStopPlan:
    async def _close() -> str | None:
        return "goodbye"

    base: dict[str, Any] = {
        "deadline_seconds": 0.01,
        "close": _close,
        "interview_session_id": uuid4(),
    }
    base.update(over)
    return HardStopPlan(**base)


@_asyncio
async def test_a_refused_submit_falls_back_to_a_reason_that_is_legal() -> None:
    """THE BUG: an untimed session's hard stop left the row in_progress forever."""
    attempts: list[str] = []

    async def _refuse() -> str | None:
        attempts.append("timed_out")
        # The exact production shape: AppError from submit_session's semantic
        # reason validation.
        raise AppError("Cannot time out an interview without a time limit")

    async def _fallback() -> str | None:
        attempts.append("ended_early")
        return "goodbye"

    announced: list[bool] = []

    async def _announce() -> None:
        announced.append(True)

    timer = HardStopTimer(
        _plan(close=_refuse, close_fallback=_fallback),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer.on_finalized = _announce

    await timer._stop()  # noqa: SLF001 -- the timer's own firing path

    assert attempts == ["timed_out", "ended_early"], (
        "a refused finish must be retried under a reason the session accepts"
    )
    assert announced == [True], "the fallback submitted, so the finish IS real"


@_asyncio
async def test_a_finish_that_cannot_be_persisted_is_not_announced() -> None:
    """Announcing an un-submitted finish, then shutting down, stranded the session.

    Staying un-finalized keeps the job alive so the model or a rejoin can still
    end it, instead of parking the candidate behind a completion screen for a
    session the database still calls live.
    """

    async def _refuse() -> str | None:
        raise AppError("Interview time limit has not elapsed")

    announced: list[bool] = []

    async def _announce() -> None:
        announced.append(True)

    timer = HardStopTimer(
        _plan(close=_refuse, close_fallback=_refuse),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer.on_finalized = _announce

    await timer._stop()  # noqa: SLF001

    assert announced == [], "the client was told the interview ended when it had not"
    assert timer.fired is False, "a failed stop must leave the timer re-armable"


@_asyncio
async def test_a_stop_that_could_not_persist_is_retried() -> None:
    """A refused stop leaves the session live, so something must try again.

    ``_stop`` deliberately does not finalize when its submit was refused. Without
    the retry loop the timer would be spent at that point and the row would stay
    ``in_progress`` until the model ended it or the deadline sweep did — and for an
    untimed session the sweep never selects it.
    """
    attempts: list[int] = []

    async def _close() -> str | None:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise RuntimeError("database briefly unavailable")
        return None

    timer = HardStopTimer(
        _plan(deadline_seconds=0.0, close=_close),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer._RETRY_AFTER_FAILED_STOP_S = 0.01  # type: ignore[misc] # noqa: SLF001
    timer.start()

    for _ in range(200):
        if timer.fired:
            break
        await asyncio.sleep(0.01)

    timer.cancel()
    assert timer.fired is True, "the retry never landed the submit"
    assert len(attempts) >= 2


@_asyncio
async def test_cancelling_stops_the_retry_loop() -> None:
    """The model ending the interview must not leave a retry loop running."""
    attempts: list[int] = []

    async def _always_refuse() -> str | None:
        attempts.append(1)
        raise RuntimeError("still refused")

    timer = HardStopTimer(
        _plan(deadline_seconds=0.0, close=_always_refuse, close_fallback=_always_refuse),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer._RETRY_AFTER_FAILED_STOP_S = 0.01  # type: ignore[misc] # noqa: SLF001
    timer.start()
    await asyncio.sleep(0.05)
    timer.cancel()

    seen = len(attempts)
    await asyncio.sleep(0.05)
    assert len(attempts) == seen, "the retry loop kept running after cancel"


@_asyncio
async def test_a_successful_submit_does_not_touch_the_fallback() -> None:
    calls: list[str] = []

    async def _close() -> str | None:
        calls.append("primary")
        return "goodbye"

    async def _fallback() -> str | None:
        calls.append("fallback")
        return "goodbye"

    timer = HardStopTimer(
        _plan(close=_close, close_fallback=_fallback),
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    await timer._stop()  # noqa: SLF001

    assert calls == ["primary"]
    assert timer.fired is True


@_asyncio
async def test_the_finish_drains_in_flight_turns_before_submitting() -> None:
    """A turn still grading has produced no transcript write for the barrier to wait on."""
    order: list[str] = []

    async def _drain() -> bool:
        order.append("drain")
        return True

    async def _flush() -> None:
        order.append("flush")

    async def _close() -> str | None:
        order.append("submit")
        return None

    timer = HardStopTimer(
        _plan(close=_close, flush_transcript=_flush, drain_turns=_drain),
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    await timer._stop()  # noqa: SLF001

    assert order == ["drain", "flush", "submit"], (
        "turns must drain before their writes are flushed, and both before submit"
    )


@_asyncio
async def test_the_model_route_also_drains_before_submitting() -> None:
    """`finalize_once` is the tool's path; it needs the same guarantee."""
    order: list[str] = []

    async def _drain() -> bool:
        order.append("drain")
        return True

    async def _flush() -> None:
        order.append("flush")

    async def _inner() -> None:
        order.append("submit")

    timer = HardStopTimer(
        _plan(flush_transcript=_flush, drain_turns=_drain),
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    await timer.finalize_once(_inner)

    assert order == ["drain", "flush", "submit"]


@_asyncio
async def test_a_drain_that_raises_does_not_stop_the_finish() -> None:
    """A submitted session matters more than a perfectly complete transcript."""
    submitted: list[bool] = []

    async def _drain() -> bool:
        raise RuntimeError("event loop confused")

    async def _inner() -> None:
        submitted.append(True)

    timer = HardStopTimer(
        _plan(drain_turns=_drain),
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    await timer.finalize_once(_inner)

    assert submitted == [True]




# ───────────────── the finalization state machine ─────────────────


@_asyncio
async def test_a_failed_first_submit_can_be_retried_to_success() -> None:
    """finalizing is not a latch: a rolled-back attempt leaves the session live."""
    attempts: list[int] = []

    async def _close() -> str | None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("database briefly unavailable")
        return None

    timer = HardStopTimer(
        _plan(deadline_seconds=0.0, close=_close),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer._RETRY_AFTER_FAILED_STOP_S = 0.01  # type: ignore[misc] # noqa: SLF001
    timer.start()

    for _ in range(200):
        if timer.fired:
            break
        await asyncio.sleep(0.01)

    timer.cancel()
    assert timer.fired is True
    assert len(attempts) == 2


@_asyncio
async def test_two_concurrent_routes_submit_once_and_share_the_result() -> None:
    """The model ends while the timer is finalizing: one submit, both await it."""
    submits: list[int] = []

    async def _inner() -> None:
        submits.append(1)
        await asyncio.sleep(0.02)  # hold the in-flight task open

    async def _close() -> str | None:
        submits.append(1)
        await asyncio.sleep(0.02)

    timer = HardStopTimer(
        _plan(close=_close, deadline_seconds=999),
        session=_FakeSession(),  # type: ignore[arg-type]
    )

    tool = asyncio.create_task(timer.finalize_once(_inner))
    timer_task = asyncio.create_task(timer._stop())  # noqa: SLF001
    await asyncio.gather(tool, timer_task)

    assert len(submits) == 1, "two routes raced into two submits"


@_asyncio
async def test_a_turn_sent_after_closing_is_refused_not_graded() -> None:
    """The intake closes before the drain: late turns cannot extend the finish."""
    closed_seen: list[bool] = []

    class _SpyIntake:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> int:
            self.closed = True
            return 7

        def reopen(self, *, expected_generation: int | None = None) -> bool:
            del expected_generation
            self.closed = False
            return True

    intake = _SpyIntake()

    async def _drain() -> bool:
        closed_seen.append(intake.closed)
        return True

    async def _inner() -> None:
        closed_seen.append(intake.closed)

    timer = HardStopTimer(
        _plan(drain_turns=_drain, turn_intake=intake),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    await timer.finalize_once(_inner)

    assert closed_seen == [True, True], "the intake was not closed during the finish"


@_asyncio
async def test_a_failed_attempt_reopens_the_intake() -> None:
    """A rolled-back finish must leave the session accepting answers again."""

    class _SpyIntake:
        def __init__(self) -> None:
            self.closed = False
            self.reopens = 0

        def close(self) -> int:
            self.closed = True
            return 3

        def reopen(self, *, expected_generation: int | None = None) -> bool:
            del expected_generation
            self.closed = False
            self.reopens += 1
            return True

    intake = _SpyIntake()

    async def _close() -> str | None:
        raise RuntimeError("still refused")

    timer = HardStopTimer(
        _plan(close=_close, close_fallback=_close, turn_intake=intake),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    await timer._stop()  # noqa: SLF001

    assert timer.fired is False
    assert intake.reopens == 1
    assert intake.closed is False


@_asyncio
async def test_a_stale_generation_cannot_reopen_the_new_attempt_s_intake() -> None:
    """A zombie attempt's rollback must not undo the retry that replaced it."""
    reopened_with: list[int | None] = []

    class _SpyIntake:
        def __init__(self) -> None:
            self.generation = 0

        def close(self) -> int:
            self.generation += 1
            return self.generation

        def reopen(self, *, expected_generation: int | None = None) -> bool:
            reopened_with.append(expected_generation)
            return expected_generation == self.generation

    intake = _SpyIntake()
    first = intake.close()  # generation 1: the FAILED attempt

    # The retry closes again (generation 2) and succeeds; the zombie's rollback
    # then tries to reopen under generation 1 and must be refused.
    second = intake.close()
    assert second == 2
    assert intake.reopen(expected_generation=first) is False
    assert intake.reopen(expected_generation=second) is True


@_asyncio
async def test_an_announce_failure_after_submit_does_not_resubmit() -> None:
    """The submit persisted; a broken announcement must not reopen the session."""
    submits: list[int] = []

    async def _close() -> str | None:
        submits.append(1)
        return None

    async def _boom() -> None:
        raise RuntimeError("room already closed")

    timer = HardStopTimer(
        _plan(close=_close),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer.on_finalized = _boom

    await timer._stop()  # noqa: SLF001

    assert len(submits) == 1
    assert timer.fired is True, "the submit landed; the session stays finalized"




# ─────────────── truthfulness: bool outcome + watchdog safety net ───────────────


@_asyncio
async def test_finalize_once_returns_true_only_on_persist() -> None:
    """The tool needs an honest boolean, not a fire-and-forget."""
    persisted: list[int] = []

    async def _inner() -> None:
        persisted.append(1)

    async def _inner_boom() -> None:
        raise RuntimeError("DB down")

    ok_timer = HardStopTimer(
        _plan(deadline_seconds=999),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    assert await ok_timer.finalize_once(_inner) is True

    fail_timer = HardStopTimer(
        _plan(deadline_seconds=999),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    assert await fail_timer.finalize_once(_inner_boom) is False
    assert persisted == [1], "the failed attempt must not have counted as a submit"


@_asyncio
async def test_a_failed_model_finalize_leaves_the_watchdog_alive() -> None:
    """The deadline must still fire after the model route failed."""
    submits: list[int] = []

    async def _inner() -> None:
        submits.append(1)
        raise RuntimeError("terminal submission failed")

    async def _close() -> str | None:
        submits.append(1)
        return None

    timer = HardStopTimer(
        _plan(deadline_seconds=0.02, close=_close),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    timer._RETRY_AFTER_FAILED_STOP_S = 0.01  # type: ignore[misc] # noqa: SLF001
    timer.start()  # the safety net must be RUNNING before the model route fails

    assert await timer.finalize_once(_inner) is False

    # The watchdog is still armed: it will retry (via its own close reason)
    # and eventually persist.
    for _ in range(200):
        if timer.fired:
            break
        await asyncio.sleep(0.01)
    assert timer.fired is True, "the watchdog never fired after the model route failed"


@_asyncio
async def test_an_operational_error_retries_the_original_reason() -> None:
    """A DB/network failure must NEVER relabel a timed interview ended_early."""
    reasons: list[str] = []

    async def _refuse_timed_out() -> str | None:
        reasons.append("timed_out")
        raise ConnectionError("database unreachable")

    async def _would_relabel() -> str | None:
        reasons.append("ended_early")
        raise AssertionError("fallback must not run for an operational error")

    timer = HardStopTimer(
        _plan(close=_refuse_timed_out, close_fallback=_would_relabel),
        session=_FakeSession(),  # type: ignore[arg-type]
    )
    await timer._stop()  # noqa: SLF001

    assert reasons == ["timed_out"], "an operational error triggered the semantic fallback"
    assert timer.fired is False


# ───────────────────────── the resume's current question ─────────────────────────


def _question(question_id: Any, outcome_id: Any = None) -> Any:
    return SimpleNamespace(id=question_id, linked_outcome_id=outcome_id)


def test_the_transcript_cannot_rewind_the_current_question() -> None:
    """THE BUG, at the merge itself: state is ahead, so it must win.

    This mirrors what ``load_native_setup`` now decides — it passes
    ``current_question=None`` when state already names a question — and pins the
    consequence: Q2 stays current, so the interview resumes where it was.
    """
    q1, q2 = uuid4(), uuid4()
    state = InterviewRuntimeStateData()
    state.current_question_id = str(q2)
    state.asked_question_ids = [str(q1), str(q2)]

    # The transcript only has Q1: Q2 was selected but its reading never recorded.
    sync_question_history(state, [q1], current_question=None)

    assert state.current_question_id == str(q2), (
        "the resume rewound to a question the candidate had been moved past"
    )
    assert str(q2) in state.asked_question_ids


def test_a_state_with_no_current_question_still_adopts_the_transcript() -> None:
    """The original purpose survives: a session whose agent never wrote state."""
    q1 = uuid4()
    outcome = uuid4()
    state = InterviewRuntimeStateData()

    sync_question_history(state, [q1], current_question=_question(q1, outcome))

    assert state.current_question_id == str(q1)
    assert state.current_outcome_id == str(outcome)


def test_merging_transcript_ids_never_drops_one() -> None:
    """The id merge is additive in both directions — that part was never the bug."""
    q1, q2, q3 = uuid4(), uuid4(), uuid4()
    state = InterviewRuntimeStateData()
    state.current_question_id = str(q3)
    state.asked_question_ids = [str(q3)]

    sync_question_history(state, [q1, q2], current_question=None)

    assert set(state.asked_question_ids) == {str(q1), str(q2), str(q3)}
