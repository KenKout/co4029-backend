"""Every budget a turn spends must be written before the turn ends.

The follow-up counters and the hint ladder are the only things that force a
stubborn interview forward: ``current_question_resolved`` reads them, and the
server's auto-advance (``native_advance.advance_if_resolved``) is what moves the
question when the model will not. They live ONLY in ``interview_runtime_state`` —
no snapshot carries them — so a mutation that is not saved is a mutation that a
worker restart un-does, handing the candidate back a probe or a hint rung they
had already spent, and letting a question be worked past its budget forever.

Two write gaps, both in branches nothing else saved:

* ``fold_turn`` charges the follow-up AFTER ``grade_turn`` has already saved, and
  only the ADVANCE branch writes again (through ``publish_state``). So the
  non-advancing turn — the common case, and the one the counter exists for — held
  its charge in memory only.
* ``interview_request_hint`` advances ``hint_level`` and is not on the graded path
  at all, so nothing on a hint turn wrote state.

These tests drive the real ``fold_turn`` and the real tool against fakes, and
assert on WRITE ORDER rather than on a final value: the bug was never a wrong
number in memory, it was a correct number that never reached the database.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

_asyncio = pytest.mark.asyncio


class _Recorder:
    """Records the counter values at each save, in order."""

    def __init__(self, state: InterviewRuntimeStateData) -> None:
        self._state = state
        self.saves: list[tuple[int, int]] = []  # (follow_ups_here, hint_level)

    async def save(self) -> None:
        self.saves.append((self._state.current_question_follow_up_count, self._state.hint_level))

    @property
    def last(self) -> tuple[int, int]:
        return self.saves[-1]


def _userdata(state: InterviewRuntimeStateData, **over: Any) -> InterviewUserdata:
    base: dict[str, Any] = {
        "interview_session_id": uuid4(),
        "student_id": uuid4(),
        "state": state,
        "questions_total": 3,
        "questions_remaining": 2,
        "current_question_text": "What does a covering index buy you?",
        "max_follow_ups_per_question": 2,
        "max_hints_per_question": 3,
    }
    base.update(over)
    return InterviewUserdata(**base)


class _Selector:
    def __init__(self, remaining: int = 2) -> None:
        self._remaining = remaining

    def remaining(self) -> int:
        return self._remaining


class _Setup:
    """Stand-in for NativeSetup: only what `fold_turn` reads."""

    def __init__(self, userdata: InterviewUserdata, recorder: _Recorder) -> None:
        self.userdata = userdata
        self.selector = _Selector()
        self.save_state = recorder.save
        self._recorder = recorder

        async def _grade(**_kwargs: Any) -> None:
            # The real grader persists at the end of its own fold. That save is
            # the one that used to be mistaken for "the turn is durable".
            await recorder.save()

        self.grade_turn = _grade


class _Agent:
    """The real `fold_turn`, bound to fakes instead of an SDK session.

    Reusing the production method is the point: a hand-rolled copy of its ordering
    would pass while the shipped ordering stayed broken.
    """

    def __init__(self, setup: _Setup) -> None:
        self._setup = setup
        self.notes = 0
        self.directives = 0

    async def refresh_state_note(self, *, opening: bool = False) -> None:
        del opening
        self.notes += 1

    async def _inject_advance_directive(self) -> None:
        self.directives += 1

    def _record_shadow(self, userdata: Any, *, advanced: bool) -> None:
        del userdata, advanced

    # Bind the production implementation.
    from abridgeai.features.interviews.realtime.native_runtime import (  # noqa: PLC0415
        NativeInterviewAgent,
    )

    fold_turn = NativeInterviewAgent.fold_turn
    _persist_turn_counters = NativeInterviewAgent._persist_turn_counters  # noqa: SLF001


# ─────────────────────── the follow-up counter ───────────────────────


@_asyncio
async def test_a_probe_turn_persists_the_follow_up_it_charged() -> None:
    """THE BUG: the charge landed after the only save, on the branch that has none."""
    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    userdata = _userdata(state)
    agent = _Agent(_Setup(userdata, recorder))

    await agent.fold_turn(answer_text="a partial answer")

    assert state.current_question_follow_up_count == 1
    assert recorder.last[0] == 1, (
        "the follow-up charge never reached a save; a restart would refund it"
    )


@_asyncio
async def test_the_charge_is_saved_after_it_is_applied_not_before() -> None:
    """Order is the whole fix: grading's save predates the increment."""
    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    agent = _Agent(_Setup(_userdata(state), recorder))

    await agent.fold_turn(answer_text="a partial answer")

    follow_up_counts = [count for count, _hints in recorder.saves]
    assert follow_up_counts[0] == 0, "grading saves before the charge — unchanged"
    assert follow_up_counts[-1] == 1, "and a later save must carry the charge"


@_asyncio
async def test_repeated_probes_each_persist_their_own_charge() -> None:
    """Three probes must be three spent probes after a restart, not one."""
    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    agent = _Agent(_Setup(_userdata(state), recorder))

    for _ in range(3):
        await agent.fold_turn(answer_text="still partial")

    assert recorder.last[0] == 3


@_asyncio
async def test_a_failing_save_does_not_cost_the_reply() -> None:
    """A dead database must not break a live interview; the charge still applies."""
    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    setup = _Setup(_userdata(state), recorder)

    async def _boom() -> None:
        raise RuntimeError("database unavailable")

    setup.save_state = _boom
    agent = _Agent(setup)

    await agent.fold_turn(answer_text="a partial answer")

    assert state.current_question_follow_up_count == 1
    assert agent.notes == 1, "the state note must still be refreshed"


@_asyncio
async def test_a_setup_without_a_writer_is_tolerated() -> None:
    """A diagnostic harness wires no writer; that must not raise mid-turn."""
    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    setup = _Setup(_userdata(state), recorder)
    setup.save_state = None  # type: ignore[assignment]
    agent = _Agent(setup)

    await agent.fold_turn(answer_text="a partial answer")

    assert state.current_question_follow_up_count == 1


# ─────────────────────────── the hint ladder ───────────────────────────


@_asyncio
async def test_a_granted_hint_persists_the_ladder() -> None:
    """THE SECOND BUG: a hint turn wrote nothing at all.

    ``interview_request_hint`` is not on the graded path, so the rung it consumed
    lived in memory until some LATER graded turn happened to save. A restart in
    between gave the candidate the same rung again, past the ladder's ceiling.
    """
    from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin

    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.current_outcome_id = str(uuid4())
    recorder = _Recorder(state)
    userdata = _userdata(state)
    userdata.save_state = recorder.save

    published: list[str] = []

    async def _publish_action(*, kind: str, text: str | None = None) -> None:
        del text
        published.append(kind)

    userdata.publish_agent_action = _publish_action  # type: ignore[assignment]
    ctx = type("_Ctx", (), {"userdata": userdata})()

    tools = InterviewToolsMixin()
    result = await InterviewToolsMixin.interview_request_hint.__wrapped__(tools, ctx)  # type: ignore[attr-defined]

    assert state.hint_level == 1
    assert recorder.saves, "a granted hint saved nothing; a restart refunds the rung"
    assert recorder.last[1] == 1
    assert "Hint rung 0" in result
    assert published == ["hint"]


@_asyncio
async def test_the_ladder_is_saved_before_the_hint_is_announced() -> None:
    """Announce-then-save would tell the candidate about a rung we might lose."""
    from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin

    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    recorder = _Recorder(state)
    userdata = _userdata(state)
    order: list[str] = []

    async def _save() -> None:
        order.append("save")
        await recorder.save()

    async def _publish_action(*, kind: str, text: str | None = None) -> None:
        del kind, text
        order.append("announce")

    userdata.save_state = _save
    userdata.publish_agent_action = _publish_action  # type: ignore[assignment]
    ctx = type("_Ctx", (), {"userdata": userdata})()

    await InterviewToolsMixin.interview_request_hint.__wrapped__(  # type: ignore[attr-defined]
        InterviewToolsMixin(), ctx
    )

    assert order == ["save", "announce"]


@_asyncio
async def test_a_refused_hint_does_not_write() -> None:
    """The ladder is spent, so nothing changed — and the refusal is a plain probe."""
    from livekit.agents import ToolError

    from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin

    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    state.hint_level = 3  # at the ceiling
    recorder = _Recorder(state)
    userdata = _userdata(state)
    userdata.save_state = recorder.save
    ctx = type("_Ctx", (), {"userdata": userdata})()

    with pytest.raises(ToolError):
        await InterviewToolsMixin.interview_request_hint.__wrapped__(  # type: ignore[attr-defined]
            InterviewToolsMixin(), ctx
        )

    assert recorder.saves == []
    assert state.hint_level == 3


@_asyncio
async def test_a_failing_save_does_not_refuse_the_hint() -> None:
    """The candidate asked for help; a write problem must not deny it."""
    from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin

    state = InterviewRuntimeStateData()
    state.current_question_id = str(uuid4())
    userdata = _userdata(state)

    async def _boom() -> None:
        raise RuntimeError("database unavailable")

    userdata.save_state = _boom
    ctx = type("_Ctx", (), {"userdata": userdata})()

    result = await InterviewToolsMixin.interview_request_hint.__wrapped__(  # type: ignore[attr-defined]
        InterviewToolsMixin(), ctx
    )

    assert "Hint rung 0" in result
    assert state.hint_level == 1
