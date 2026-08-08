"""Contract tests for the agent's ``@function_tool`` surface.

``orchestrator/tools.py`` is already property-tested; this file pins the ADAPTER:
that each tool reads the userdata fields it claims to, mutates the SAME state
object the runtime will persist, and — most importantly — converts a refusal into
a ``ToolError`` whose message names what is missing.

The refusal path is the reason this file exists. If a gate silently returned a
value instead of raising, the model would advance through the whole question bank
and the coverage guarantee would be gone with no test failing.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from livekit.agents import ToolError

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS
from abridgeai.features.interviews.orchestrator.state import (
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.realtime.agent_tools import InterviewToolsMixin
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata


class _Question:
    def __init__(self, outcome_id: str, prompt_text: str) -> None:
        self.outcome_id = outcome_id
        self.prompt_text = prompt_text


class _Ctx:
    """Minimal stand-in for ``RunContext`` — the tools only touch ``userdata``."""

    def __init__(self, userdata: InterviewUserdata) -> None:
        self.userdata = userdata


def _userdata(*, points: int = 0, **kw: object) -> InterviewUserdata:
    state = InterviewRuntimeStateData()
    state.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=points)}
    state.current_outcome_id = "o1"
    data = InterviewUserdata(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        state=state,
        required_outcome_ids=["o1"],
        outcome_titles={"o1": "Explain index selection"},
        questions_remaining=3,
        select_next=lambda: _Question("o2", "What is a covering index?"),
    )
    for key, value in kw.items():
        setattr(data, key, value)
    return data


@pytest.fixture
def tools() -> InterviewToolsMixin:
    return InterviewToolsMixin()


async def _call(tool: object, ctx: _Ctx, **kw: object) -> object:
    """Invoke a ``@function_tool``-decorated bound method.

    ``function_tool`` wraps the coroutine but leaves it callable, so the bound
    method can be awaited directly. Kept behind a helper so every test calls the
    tools the same way and a future SDK change is a one-line fix here.
    """
    return await tool(ctx, **kw)  # type: ignore[operator]


# ── next_question: the coverage guarantee lives here ──────────────────────────


async def test_next_question_refuses_while_current_outcome_uncovered(
    tools: InterviewToolsMixin,
) -> None:
    ctx = _Ctx(_userdata(points=0))
    with pytest.raises(ToolError) as excinfo:
        await _call(tools.interview_next_question, ctx)
    message = str(excinfo.value).lower()
    # Actionable, not just "no": it must say what to do instead.
    assert "probe" in message or "hint" in message


async def test_next_question_grants_and_resets_per_question_counters(
    tools: InterviewToolsMixin,
) -> None:
    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS)
    assert data.state is not None
    data.state.hint_level = 2
    data.state.current_question_follow_up_count = 2
    result = await _call(tools.interview_next_question, _Ctx(data))

    assert "covering index" in str(result), "the selected question text was not returned"
    # The counters are per-question and MUST reset, or the next question starts
    # with a spent hint ladder and the candidate silently loses their scaffolding.
    assert data.state.hint_level == 0
    assert data.state.current_question_follow_up_count == 0
    assert data.state.current_outcome_id == "o2", "current outcome not advanced"


async def test_next_question_moves_the_question_the_grader_reads(
    tools: InterviewToolsMixin,
) -> None:
    """Advancing must repoint ``current_question_text``.

    ``native_runtime.on_user_turn_completed`` grades each answer against this
    field and ``interview_request_hint`` scaffolds from it. While it was written
    only at setup, every answer after the first advance was graded against
    question one and every hint scaffolded the wrong question.
    """
    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS, current_question_text="What is an index?")
    await _call(tools.interview_next_question, _Ctx(data))

    assert data.current_question_text == "What is a covering index?"


async def test_hint_scaffolds_the_question_after_an_advance(
    tools: InterviewToolsMixin,
) -> None:
    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS, current_question_text="What is an index?")
    await _call(tools.interview_next_question, _Ctx(data))
    hint = str(await _call(tools.interview_request_hint, _Ctx(data)))

    assert "covering index" in hint, "the hint scaffolded a question the candidate is no longer on"


async def test_next_question_refuses_when_the_bank_is_empty(
    tools: InterviewToolsMixin,
) -> None:
    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS)
    data.select_next = lambda: None
    with pytest.raises(ToolError):
        await _call(tools.interview_next_question, _Ctx(data))


# ── request_hint ─────────────────────────────────────────────────────────────


async def test_request_hint_escalates_then_refuses(tools: InterviewToolsMixin) -> None:
    data = _userdata(points=0)
    for _ in range(MAX_CANNOT_ANSWER_HINTS):
        await _call(tools.interview_request_hint, _Ctx(data))
    with pytest.raises(ToolError) as excinfo:
        await _call(tools.interview_request_hint, _Ctx(data))
    # The refusal must redirect, so the model does not just retry the same tool.
    assert "next_question" in str(excinfo.value).lower()


# ── end_interview ────────────────────────────────────────────────────────────


async def test_end_interview_refuses_and_names_the_missing_outcome(
    tools: InterviewToolsMixin,
) -> None:
    ctx = _Ctx(_userdata(points=0))
    with pytest.raises(ToolError) as excinfo:
        await _call(tools.interview_end_interview, ctx)
    assert "Explain index selection" in str(excinfo.value), (
        "a generic refusal invites the model to retry the identical call"
    )


async def test_end_interview_allowed_once_covered(tools: InterviewToolsMixin) -> None:
    called: list[bool] = []

    async def _finalize() -> None:
        called.append(True)

    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS)
    data.finalize_session = _finalize
    await _call(tools.interview_end_interview, _Ctx(data))
    assert called == [True], "the interview was not actually finalized"


# ── get_progress ─────────────────────────────────────────────────────────────


async def test_get_progress_returns_json_without_answer_content(
    tools: InterviewToolsMixin,
) -> None:
    import json

    raw = await _call(tools.interview_get_progress, _Ctx(_userdata(points=1)))
    payload = json.loads(str(raw))
    assert payload["required_unticked"] == ["o1"]
    lowered = str(raw).lower()
    for banned in ("the answer is", "correct answer", "rubric", "expected evidence"):
        assert banned not in lowered


def test_the_reminder_pins_the_question_and_forbids_narrated_advances() -> None:
    """The state note must name the live question and require the tool to leave it.

    Without this the model announced "Moving on to the next question:" and asked a
    new one in its own words, never calling `interview_next_question`. Server-side
    the interview stayed put, so the candidate's card, the question counter and the
    scoring all tracked a question nobody was answering.
    """
    from abridgeai.features.interviews.orchestrator.tools import build_turn_reminder

    data = InterviewRuntimeStateData()
    data.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=0)}

    note = build_turn_reminder(
        data,
        current_outcome_id="o1",
        required_outcome_ids=["o1"],
        questions_remaining=2,
        max_follow_ups_per_question=2,
        below_closing_threshold=False,
        current_question_text="What is an index?",
    )

    assert "What is an index?" in note
    assert "next_question" in note
    lowered = note.lower()
    assert "do not ask any other question" in lowered
