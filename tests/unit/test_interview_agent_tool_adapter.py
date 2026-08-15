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


class _Selector:
    """Callable stand-in for ``BankSelector``: selecting consumes the pool."""

    def __init__(self, *questions: _Question) -> None:
        self._questions = list(questions)

    def __call__(self) -> _Question | None:
        return self._questions.pop(0) if self._questions else None

    def remaining(self) -> int:
        return len(self._questions)


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
        select_next=_Selector(_Question("o2", "What is a covering index?")),
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


async def test_next_question_syncs_the_questions_remaining_counter(
    tools: InterviewToolsMixin,
) -> None:
    """The snapshot published inside the advance must renumber the header.

    ``question_number`` is derived as ``total - questions_remaining``, and only
    ``fold_turn`` recomputed that plain int — so between a tool advance and the
    next graded answer the client card showed the NEW question while the header
    still said "Question 1 of 3".
    """
    data = _userdata(points=COVERAGE_SUFFICIENT_POINTS)
    assert data.questions_remaining == 3
    await _call(tools.interview_next_question, _Ctx(data))

    assert data.questions_remaining == 0, "remaining was not resynced at the advance"


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


async def test_request_hint_refunds_the_follow_up_charge(
    tools: InterviewToolsMixin,
) -> None:
    """A granted hint does not consume the academic follow-up budget.

    `fold_turn` charges one follow-up for every turn that does not advance —
    including a TYPED hint request, which arrives as an `answer` turn when the
    candidate types "give me a hint" instead of using a hint button. Without
    this refund the follow-up budget (2) exhausts before the hint ladder (3):
    the candidate gets one rung, then the question advances with no transition
    (production session fb204f73). The refund mirrors the routed path's
    STUDENT_REQUESTED_HINT exemption in turn_state.py.
    """
    data = _userdata(points=0)
    assert data.state is not None
    data.state.current_question_follow_up_count = 2  # what fold_turn already charged
    await _call(tools.interview_request_hint, _Ctx(data))
    assert data.state.current_question_follow_up_count == 1, (
        "a granted hint request must give the follow-up back"
    )


async def test_request_hint_refund_never_goes_below_zero(
    tools: InterviewToolsMixin,
) -> None:
    """The refund clamps at zero — a hint turn must not mint budget."""
    data = _userdata(points=0)
    assert data.state is not None
    data.state.current_question_follow_up_count = 0
    await _call(tools.interview_request_hint, _Ctx(data))
    assert data.state.current_question_follow_up_count == 0


async def test_request_hint_refunds_at_most_once_per_question(
    tools: InterviewToolsMixin,
) -> None:
    """A second granted hint must not give the follow-up budget back again.

    The model calls the hint tool on its own scaffolding instinct too, and an
    unbounded refund let those calls cancel one charge every turn: the count
    ping-ponged below its threshold and the question could never auto-advance
    (production session 13e0b4c4: three follow-ups, count stuck at 1).
    """
    data = _userdata(points=0)
    assert data.state is not None
    data.state.current_question_follow_up_count = 1
    await _call(tools.interview_request_hint, _Ctx(data))
    assert data.state.current_question_follow_up_count == 0, "first refund stands"

    data.state.current_question_follow_up_count = 2
    await _call(tools.interview_request_hint, _Ctx(data))
    assert data.state.current_question_follow_up_count == 2, (
        "the second hint on one question must not refund again"
    )


async def test_request_hint_does_not_refund_when_spent(
    tools: InterviewToolsMixin,
) -> None:
    """A REFUSED hint (ladder spent) is not a hint turn — no refund.

    When the ladder is exhausted the request fails, so the follow-up charged by
    `fold_turn` stands: the refusal itself is a probe of why the candidate is
    still stuck, and refunding it would let a hint-spamming candidate escape
    the follow-up gate entirely.
    """
    data = _userdata(points=0)
    assert data.state is not None
    for _ in range(MAX_CANNOT_ANSWER_HINTS):
        await _call(tools.interview_request_hint, _Ctx(data))
    # Set the charge AFTER the granting loop: those granted hints legitimately
    # refunded. This one is the refusal — its charge must stand.
    data.state.current_question_follow_up_count = 1
    with pytest.raises(ToolError):
        await _call(tools.interview_request_hint, _Ctx(data))
    assert data.state.current_question_follow_up_count == 1, (
        "a refused hint request keeps its follow-up charge"
    )


async def test_request_hint_marks_the_next_utterance_as_a_hint(
    tools: InterviewToolsMixin,
) -> None:
    """A granted hint must reach both the transcript and the live client.

    ``_record_conversation`` persists every assistant row as kind="question"
    unless told otherwise, and the live labeler has no signal at all for a
    TYPED hint request (it arrives as an answer). The marker + the
    ``agent_action`` event are what make the badge say HINT instead of
    FOLLOW-UP — live and after a reload.
    """
    published: list[str] = []

    async def _publish(kind: str) -> None:
        published.append(kind)

    data = _userdata(points=0, publish_agent_action=_publish)
    await _call(tools.interview_request_hint, _Ctx(data))

    assert data.pending_assistant_kind == "hint"
    assert published == ["hint"]


async def test_refused_hint_sets_no_marker(
    tools: InterviewToolsMixin,
) -> None:
    from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS

    published: list[str] = []

    async def _publish(kind: str) -> None:
        published.append(kind)

    data = _userdata(points=0, publish_agent_action=_publish)
    assert data.state is not None
    data.state.hint_level = MAX_CANNOT_ANSWER_HINTS
    data.pending_assistant_kind = None
    with pytest.raises(ToolError):
        await _call(tools.interview_request_hint, _Ctx(data))

    assert data.pending_assistant_kind is None
    assert published == []


async def test_hint_refused_while_the_new_question_is_unasked(
    tools: InterviewToolsMixin,
) -> None:
    """No hint before the question it belongs to has been asked.

    The server advances on its own and the model has not spoken the new
    question yet; a hint granted in that window lands its marker on the
    question's own reading (production: the Q3 paraphrase persisted and
    badged as a hint).
    """
    published: list[str] = []

    async def _publish(kind: str) -> None:
        published.append(kind)

    data = _userdata(points=0, publish_agent_action=_publish)
    data.pending_new_question = True
    with pytest.raises(ToolError) as excinfo:
        await _call(tools.interview_request_hint, _Ctx(data))

    assert "Ask the current question first" in str(excinfo.value)
    assert data.pending_assistant_kind is None
    assert published == []


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
