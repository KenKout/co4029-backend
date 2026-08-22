"""The native (multiturn) agent path, and the flag that selects it.

Five properties are pinned here, each because losing it fails silently:

* The flag actually routes. A native agent that never runs, or a routed path that
  quietly keeps running with the flag ON, both look like "voice works".
* The four tools reach the model. Without them the LLM has no question bank, no
  hint ladder and no way to end — it would improvise an interview.
* ``on_user_turn_completed`` injects the state note and does NOT raise
  ``StopResponse``. The routed agent MUST raise it (no LLM, so the default reply
  is silence); this agent must not, or the candidate hears nothing back.
* The hard stop finalizes a session the model never ends. This is the only
  anti-deadlock layer that survives a model which simply keeps talking.
* The session is built WITH an ``llm``. Losing it does not break the room — the
  agent just stops being able to read its own conversation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

pytest.importorskip("livekit.agents", reason="requires the interview-agent extra")

from livekit.agents import ChatContext, ChatMessage, StopResponse  # noqa: E402

from abridgeai.features.interviews.orchestrator.selection import (  # noqa: E402
    CandidateQuestion,
)
from abridgeai.features.interviews.orchestrator.state import (  # noqa: E402
    InterviewRuntimeStateData,
    OutcomeCoverageState,
)
from abridgeai.features.interviews.orchestrator.tools import (  # noqa: E402
    COVERAGE_SUFFICIENT_POINTS,
)
from abridgeai.features.interviews.realtime import agent as agent_module  # noqa: E402
from abridgeai.features.interviews.realtime import native_bridge, native_runtime  # noqa: E402
from abridgeai.features.interviews.realtime.agent_session import (  # noqa: E402
    build_state_reminder,
)
from abridgeai.features.interviews.realtime.agent_userdata import (  # noqa: E402
    InterviewUserdata,
)
from abridgeai.features.interviews.services.real_time import (  # noqa: E402
    META_LANGUAGE,
    META_SESSION_ID,
    META_STUDENT_ID,
)

EXPECTED_TOOLS = {
    "interview_get_progress",
    "interview_next_question",
    "interview_request_hint",
    "interview_end_interview",
}


# ── fixtures / builders ───────────────────────────────────────────────────────


class FakeSpeechHandle:
    def __await__(self):
        async def _noop() -> None:
            return None

        return _noop().__await__()

    async def wait_for_playout(self) -> None:
        return None


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, text: str, topic: str) -> None:
        self.sent.append((topic, text))


class FakeSession:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.generated: list[dict[str, Any]] = []
        self.started_with: dict[str, Any] = {}
        self.local = FakeLocalParticipant()
        self.handlers: dict[str, list[Any]] = {}
        # The publisher reaches the room through `session.room_io.room`.
        # `AgentSession` exposes no `.room`, and a double that invents one hides a
        # dropped-egress bug the routed path already shipped once.
        self.room_io = SimpleNamespace(room=SimpleNamespace(local_participant=self.local))

    def on(self, event: str, handler: Any) -> None:
        # The runtime subscribes `conversation_item_added` to persist the
        # transcript. A double without this silently loses that wiring.
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(payload)

    def control_events(self) -> list[dict[str, Any]]:
        import json

        return [json.loads(text) for _topic, text in self.local.sent]

    def say(self, text: str, **kwargs: Any) -> FakeSpeechHandle:
        self.said.append(text)
        return FakeSpeechHandle()

    def generate_reply(self, **kwargs: Any) -> FakeSpeechHandle:
        self.generated.append(kwargs)
        return FakeSpeechHandle()

    async def start(self, agent: Any, **kwargs: Any) -> None:
        self.started_with = {"agent": agent, **kwargs}


def _state(points: int = 0) -> InterviewRuntimeStateData:
    state = InterviewRuntimeStateData()
    state.outcome_coverage = {"o1": OutcomeCoverageState(outcome_id="o1", coverage_points=points)}
    state.current_outcome_id = "o1"
    return state


def _selector(state: InterviewRuntimeStateData) -> native_bridge.BankSelector:
    return native_bridge.BankSelector(
        candidates=[
            CandidateQuestion(
                question_id="q1",
                linked_outcome_id="o1",
                question_type="open",
                difficulty="junior",
                position=0,
            ),
            CandidateQuestion(
                question_id="q2",
                linked_outcome_id="o2",
                question_type="open",
                difficulty="junior",
                position=1,
            ),
        ],
        prompts={"q1": "What is an index?", "q2": "What is a covering index?"},
        state=state,
        required_outcome_ids=("o1", "o2"),
        time_fraction_remaining=None,
    )


def _setup(*, closer: Any = None) -> native_bridge.NativeSetup:
    state = _state()
    selector = _selector(state)
    userdata = InterviewUserdata(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        state=state,
        required_outcome_ids=["o1", "o2"],
        outcome_titles={"o1": "Index selection", "o2": "Covering indexes"},
        questions_remaining=2,
        questions_total=2,
        current_question_text="What is an index?",
        select_next=selector,
    )
    return native_bridge.NativeSetup(
        userdata=userdata,
        language="en",
        input_mode="voice",
        close_session=closer or AsyncMock(return_value="Thank you. Goodbye."),
        selector=selector,
        onboarding_turns=[("assistant", "Hello, I'm Ha."), ("user", "I'm Nam.")],
    )


def _agent(setup: native_bridge.NativeSetup) -> native_runtime.NativeInterviewAgent:
    return native_runtime.NativeInterviewAgent(
        instructions="test instructions",
        chat_ctx=ChatContext.empty(),
        setup=setup,
    )


# ── the flag routes ───────────────────────────────────────────────────────────


@pytest.fixture
def job_ctx() -> SimpleNamespace:
    session_id, student_id = uuid4(), uuid4()
    metadata = (
        f'{{"{META_SESSION_ID}": "{session_id}", "{META_STUDENT_ID}": "{student_id}", '
        f'"{META_LANGUAGE}": "en"}}'
    )
    room = SimpleNamespace(on=lambda *_args, **_kw: None)
    return SimpleNamespace(
        job=SimpleNamespace(metadata=metadata),
        room=room,
        connect=AsyncMock(),
        shutdown=lambda **_kw: None,
    )


@pytest.fixture
def routed_and_native(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Stub BOTH paths' entry calls so only the routing decision is observed."""
    calls: dict[str, AsyncMock] = {
        "load_native_setup": AsyncMock(return_value=_setup()),
        "run_native_interview": AsyncMock(),
        "get_current_question_text": AsyncMock(return_value="Q1?"),
        "get_opening_text": AsyncMock(return_value=None),
        "get_room_intro_text": AsyncMock(return_value=None),
        "get_tts_voice": AsyncMock(return_value=None),
        "get_voice_persona": AsyncMock(return_value=(None, 2)),
    }
    monkeypatch.setattr(agent_module.native_bridge, "load_native_setup", calls["load_native_setup"])
    monkeypatch.setattr(
        agent_module.native_runtime, "run_native_interview", calls["run_native_interview"]
    )
    for name in (
        "get_current_question_text",
        "get_opening_text",
        "get_room_intro_text",
        "get_tts_voice",
        "get_voice_persona",
    ):
        monkeypatch.setattr(agent_module.bridge, name, calls[name])

    fake_session = FakeSession()
    monkeypatch.setattr(agent_module, "build_agent_session", lambda *_a, **_kw: fake_session)
    monkeypatch.setattr(agent_module, "BackgroundAudioPlayer", lambda **_kw: AsyncMock())
    return calls


def _with_flag(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    real = agent_module.get_settings()
    stub = SimpleNamespace(
        interview_native_agent_enabled=enabled,
        interview_voice_load_threshold=real.interview_voice_load_threshold,
        livekit_agent_name=real.livekit_agent_name,
    )
    monkeypatch.setattr(agent_module, "get_settings", lambda: stub)


async def test_flag_off_selects_the_routed_path(
    monkeypatch: pytest.MonkeyPatch,
    job_ctx: SimpleNamespace,
    routed_and_native: dict[str, AsyncMock],
) -> None:
    _with_flag(monkeypatch, enabled=False)
    await agent_module.entrypoint(job_ctx)

    routed_and_native["run_native_interview"].assert_not_awaited()
    routed_and_native["load_native_setup"].assert_not_awaited()
    # The routed path's own first DB read is the proof it ran, not merely that
    # the native one did not.
    routed_and_native["get_current_question_text"].assert_awaited_once()


async def test_flag_on_selects_the_native_path(
    monkeypatch: pytest.MonkeyPatch,
    job_ctx: SimpleNamespace,
    routed_and_native: dict[str, AsyncMock],
) -> None:
    _with_flag(monkeypatch, enabled=True)
    await agent_module.entrypoint(job_ctx)

    routed_and_native["load_native_setup"].assert_awaited_once()
    routed_and_native["run_native_interview"].assert_awaited_once()
    # Nothing from the routed path may run: the branch returns, it does not fall
    # through into building a second session on the same room.
    routed_and_native["get_current_question_text"].assert_not_awaited()
    routed_and_native["get_voice_persona"].assert_not_awaited()


# ── the tool surface ─────────────────────────────────────────────────────────


def test_native_agent_exposes_exactly_the_four_interview_tools() -> None:
    agent = _agent(_setup())
    assert {tool.info.name for tool in agent.tools} == EXPECTED_TOOLS


def test_native_agent_has_the_userdata_the_tools_read_through() -> None:
    setup = _setup()
    assert _agent(setup).userdata is setup.userdata


# ── the turn hook ────────────────────────────────────────────────────────────


async def test_on_user_turn_completed_folds_the_note_into_the_instructions() -> None:
    """The note goes to the SYSTEM instructions, never into the conversation.

    This gateway is Gemini-backed: probed live, the same note at `messages[0]`
    produced the question it names, while the note as a mid-conversation system
    message produced a generic greeting — the model was discarding it. A trailing
    system message is refused outright ("Requests ending with a model turn are not
    supported"), which is what silenced the opening.
    """
    agent = _agent(_setup())
    turn_ctx = ChatContext.empty()
    turn_ctx.add_message(role="user", content="An index speeds up lookups.")

    await agent.on_user_turn_completed(turn_ctx, SimpleNamespace(text_content="…"))

    assert not [
        item for item in turn_ctx.items if item.type == "message" and item.role == "system"
    ], "the note must not be added to the conversation"
    assert "What is an index?" in agent.instructions
    assert agent.instructions.startswith("test instructions")


async def test_on_user_turn_completed_does_not_raise_stop_response() -> None:
    """The whole point of the native path is that the LLM answers.

    ``StopResponse`` here would suppress the generation step and leave the
    candidate in silence after every answer — the exact failure the routed
    agent's opposite behaviour makes easy to reintroduce by copy-paste.
    """
    agent = _agent(_setup())
    try:
        await agent.on_user_turn_completed(ChatContext.empty(), SimpleNamespace(text_content="hi"))
    except StopResponse as exc:  # pragma: no cover - the assertion is the message
        pytest.fail(f"native agent must not suppress the reply: {exc!r}")


async def test_turn_hook_refreshes_the_question_count() -> None:
    setup = _setup()
    setup.userdata.questions_remaining = 99
    await _agent(setup).on_user_turn_completed(
        ChatContext.empty(), SimpleNamespace(text_content="x")
    )
    assert setup.userdata.questions_remaining == 2


# ── onboarding seeding ───────────────────────────────────────────────────────


def test_onboarding_turns_are_seeded_before_the_first_reply() -> None:
    from abridgeai.features.interviews.realtime.agent_context import seed_onboarding_history

    chat_ctx = ChatContext.empty()
    seed_onboarding_history(chat_ctx, _setup().onboarding_turns)
    replayed = [(item.role, item.text_content) for item in chat_ctx.items if item.type == "message"]
    assert ("user", "I'm Nam.") in replayed, "the agent would re-ask the candidate's name"


# ── the hard stop ────────────────────────────────────────────────────────────


def test_hard_stop_respects_the_time_limit_over_the_question_budget() -> None:
    # Reported: a 30-minute interview with a 2-question pool was killed after 8
    # minutes (`min(2*240s, 1800s)`), and the timed-out submission was then
    # rejected because the real limit had not elapsed. The configured limit is
    # authoritative; the per-question budget is only for untimed sessions.
    assert native_runtime.hard_stop_deadline_seconds(
        time_remaining_seconds=1800, questions_remaining=2
    ) == pytest.approx(1805.0)
    # An untimed session falls back to the question budget.
    assert native_runtime.hard_stop_deadline_seconds(
        time_remaining_seconds=None, questions_remaining=1
    ) == pytest.approx(240.0)


def test_hard_stop_never_arms_immediately() -> None:
    """A session joined at the end of its window still gets a closing exchange."""
    assert (
        native_runtime.hard_stop_deadline_seconds(time_remaining_seconds=0, questions_remaining=0)
        >= 120.0
    )


def test_untimed_session_still_gets_a_deadline() -> None:
    assert (
        native_runtime.hard_stop_deadline_seconds(
            time_remaining_seconds=None, questions_remaining=0
        )
        > 0
    )


def test_rejoin_question_text_prepends_a_language_appropriate_lead_in() -> None:
    assert native_runtime._rejoin_question_text("What is an index?", "en") == (
        "Let me repeat the question: What is an index?"
    )
    assert native_runtime._rejoin_question_text("Chỉ mục là gì?", "vi") == (
        "Để tôi nhắc lại câu hỏi: Chỉ mục là gì?"
    )


def test_rejoin_question_text_defaults_to_english_for_unknown_language() -> None:
    assert native_runtime._rejoin_question_text("Q?", "fr").startswith(
        "Let me repeat the question:"
    )


async def test_hard_stop_finalizes_when_the_model_never_ends() -> None:
    closer = AsyncMock(return_value="That concludes your interview. Goodbye.")
    session = FakeSession()
    timer = native_runtime.HardStopTimer(
        native_runtime.HardStopPlan(
            deadline_seconds=0.01,
            close=closer,
            interview_session_id=uuid4(),
        ),
        session=session,  # type: ignore[arg-type]
    )
    timer.start()
    await asyncio.sleep(0.15)

    closer.assert_awaited_once()
    assert session.said == ["That concludes your interview. Goodbye."]
    assert timer.fired is True


async def test_the_model_ending_disarms_the_timer() -> None:
    """One session, one submit — whichever route gets there first."""
    closer = AsyncMock(return_value="closing")
    tool_finalize = AsyncMock()
    timer = native_runtime.HardStopTimer(
        native_runtime.HardStopPlan(
            deadline_seconds=0.01,
            close=closer,
            interview_session_id=uuid4(),
        ),
        session=FakeSession(),  # type: ignore[arg-type]
    )
    timer.start()
    await timer.finalize_once(tool_finalize)
    await asyncio.sleep(0.15)

    tool_finalize.assert_awaited_once()
    closer.assert_not_awaited()
    await timer.finalize_once(tool_finalize)
    tool_finalize.assert_awaited_once()


# ── the selector ─────────────────────────────────────────────────────────────


def test_selector_marks_its_pick_as_asked() -> None:
    """Otherwise the highest-scoring question wins forever and is re-asked."""
    state = _state()
    selector = _selector(state)
    first = selector()
    second = selector()

    assert first is not None
    assert second is not None
    assert first.prompt_text != second.prompt_text
    assert selector() is None, "a two-question bank must run out after two picks"


def test_selector_reports_the_remaining_pool() -> None:
    selector = _selector(_state())
    assert selector.remaining() == 2
    selector()
    assert selector.remaining() == 1


# ── the composed start ───────────────────────────────────────────────────────


async def test_run_native_interview_wires_room_options_and_arms_the_stop(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup()
    tool_finalize = setup.userdata.finalize_session

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        started = fake_session.started_with
        assert started["room"] is job_ctx.room
        # Voice mode: audio both ways, and typed turns still accepted (hybrid
        # composer).
        assert started["room_input_options"].audio_enabled is True
        # Typed turns MUST NOT use the SDK default callback. It reads only
        # `ev.text` — dropping the `turn_action` / `turn_key` stream attributes —
        # and it reaches `generate_reply` without ever passing
        # `on_user_turn_completed`, which the SDK fires only from the STT path. On
        # the default a typed answer is never graded, and in a text-mode session
        # that is every answer.
        assert (
            started["room_input_options"].text_input_cb.__module__
            == "abridgeai.features.interviews.realtime.native_text_input"
        ), "typed turns must go through the attribute-aware, graded callback"
        assert started["room_output_options"].audio_enabled is True
        assert isinstance(started["agent"], native_runtime.NativeInterviewAgent)
        # The tool's finalizer is replaced by the once-only guard, so the model
        # ending and the timer firing cannot both submit the session.
        assert setup.userdata.finalize_session is not tool_finalize
        assert hard_stop.fired is False
    finally:
        hard_stop.cancel()


async def test_on_enter_asks_question_one_in_the_interviewer_s_own_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opening is generated, not read out.

    Reading the bank text verbatim made question one the ONLY question missing
    from the candidate's transcript: a verbatim reading is indistinguishable from
    the card pinned above it, so the client drops it as a duplicate, while every
    paraphrased later question is shown.

    The question must still reach the model — as the CONTENT of the instruction,
    never as a tool call, since opening with `next_question` would skip it.
    """
    setup = _setup()
    agent = _agent(setup)
    fake_session = FakeSession()
    monkeypatch.setattr(type(agent), "session", property(lambda _self: fake_session))

    await agent.on_enter()

    assert fake_session.said == [], "the bank text must not be spoken verbatim"
    assert len(fake_session.generated) == 1
    assert fake_session.generated[0]["tool_choice"] == "none"
    # The question rides in the SYSTEM instructions, which the SDK keeps at
    # `messages[0]`. Passing it as a per-call `instructions=` would append a
    # trailing system message (`agent_activity._pipeline_reply_task_impl`), and
    # this gateway rejects any request that does not end on a user turn.
    assert "instructions" not in fake_session.generated[0]
    assert "What is an index?" in agent.instructions
    assert "OPEN THE INTERVIEW NOW" in agent.instructions


async def test_on_enter_stays_silent_when_no_question_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    setup.userdata.current_question_text = None
    agent = _agent(setup)
    fake_session = FakeSession()
    monkeypatch.setattr(type(agent), "session", property(lambda _self: fake_session))

    await agent.on_enter()

    assert fake_session.generated == []
    assert fake_session.said == []


# ── the session really has an LLM ────────────────────────────────────────────


async def test_native_session_is_built_with_an_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructed for real, in an async context.

    ``AgentSession`` needs a running event loop, so this cannot be a sync test —
    a sync ``asyncio.get_event_loop()`` raises. Vietnamese is used because that
    branch reaches the OpenAI-compatible TTS gateway, which accepts
    ``NOT_GIVEN`` credentials; the Deepgram STT on that branch only needs a
    non-empty key to CONSTRUCT (no network until a stream opens), so a fake
    key is injected for environments without a real one (CI).
    """
    from pydantic import SecretStr

    from abridgeai.core.config import get_settings
    from abridgeai.features.interviews.realtime.agent_session import build_native_session

    settings = get_settings()
    if not settings.deepgram_api_key:
        monkeypatch.setattr(settings, "deepgram_api_key", SecretStr("unit-test"))

    setup = _setup()
    session = build_native_session(settings, setup.userdata, language="vi")
    try:
        assert session.llm is not None, "the native session lost its LLM; it is no longer multiturn"
        assert session.userdata is setup.userdata
    finally:
        await session.aclose()


async def test_vietnamese_tts_is_cartesia_when_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a Cartesia key the VI voice is sonic-3, not the gateway whisper."""
    from livekit.plugins import cartesia as cartesia_plugin
    from pydantic import SecretStr

    from abridgeai.core.config import get_settings
    from abridgeai.features.interviews.realtime.agent_session import build_native_session

    settings = get_settings()
    monkeypatch.setattr(settings, "deepgram_api_key", SecretStr("unit-test"))
    monkeypatch.setattr(settings, "cartesia_api_key", SecretStr("unit-test"))

    setup = _setup()
    session = build_native_session(settings, setup.userdata, language="vi")
    try:
        assert isinstance(session.tts, cartesia_plugin.TTS)
        assert session.tts._opts.language == "vi"
        assert session.tts._opts.voice == settings.cartesia_tts_voice_vi
    finally:
        await session.aclose()


async def test_vietnamese_tts_falls_back_without_cartesia_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Cartesia key => the VI session keeps the OpenAI-compatible gateway
    voice, so a keyless deployment (CI, local) still constructs."""
    from livekit.plugins import openai as openai_plugin
    from pydantic import SecretStr

    from abridgeai.core.config import get_settings
    from abridgeai.features.interviews.realtime.agent_session import build_native_session

    settings = get_settings()
    monkeypatch.setattr(settings, "deepgram_api_key", SecretStr("unit-test"))
    monkeypatch.setattr(settings, "cartesia_api_key", None)

    setup = _setup()
    session = build_native_session(settings, setup.userdata, language="vi")
    try:
        assert isinstance(session.tts, openai_plugin.TTS)
    finally:
        await session.aclose()


# ── grading is actually wired into the turn (the seam between workstreams) ─────


async def test_on_user_turn_completed_grades_the_answer() -> None:
    """Without this call, `outcome_coverage` never moves on a spoken turn.

    The state note would then read "NOT yet covered" for the whole interview, the
    advance gate would only ever open on its bounded refusal budget, and the hard
    stop would be the only thing that could end the session.
    """
    graded: list[dict[str, Any]] = []

    async def _grade(**kwargs: Any) -> None:
        graded.append(kwargs)

    setup = _setup()
    setup.grade_turn = _grade  # type: ignore[misc]
    agent = _agent(setup)

    ctx = ChatContext.empty()
    await agent.on_user_turn_completed(
        ctx, ChatMessage(role="user", content=["Operational handles transactions."])
    )

    assert len(graded) == 1, "the candidate's answer was never graded"
    assert graded[0]["answer_text"] == "Operational handles transactions."


async def test_grading_failure_does_not_break_the_turn() -> None:
    async def _boom(**kwargs: Any) -> None:
        raise RuntimeError("probe exploded")

    setup = _setup()
    setup.grade_turn = _boom  # type: ignore[misc]
    agent = _agent(setup)

    ctx = ChatContext.empty()
    # Must not raise: the LLM still has to reply to the candidate.
    await agent.on_user_turn_completed(ctx, ChatMessage(role="user", content=["An answer."]))
    # And the note must still reach the instructions, so the model is not left blind.
    assert "What is an index?" in agent.instructions, "state note missing after a grading failure"


async def test_grading_runs_before_the_note_is_built() -> None:
    """Order matters: a note built first describes last turn's coverage."""
    order: list[str] = []

    async def _grade(**kwargs: Any) -> None:
        order.append("grade")

    setup = _setup()
    setup.grade_turn = _grade  # type: ignore[misc]
    agent = _agent(setup)

    async def _spy(**_kwargs: Any) -> None:
        order.append("note")

    agent.refresh_state_note = _spy  # type: ignore[assignment,method-assign]

    await agent.on_user_turn_completed(
        ChatContext.empty(), ChatMessage(role="user", content=["A."])
    )

    assert order == ["grade", "note"], f"wrong order: {order}"


# ── shadow checker runs on the native turn, for the audit trail ───────────────


async def test_shadow_verdict_is_emitted_each_turn() -> None:
    """The native path has no ReasonCode of its own; this is where it comes from.

    Without it, a graded interview run by the conversational agent leaves no
    auditable justification for any decision, and a student appeal has nothing to
    examine.
    """
    from abridgeai.features.interviews.realtime import observability as obs

    emitted: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, **fields: Any) -> None:
        emitted.append((event, fields))

    setup = _setup()
    setup.grade_turn = AsyncMock()
    agent = _agent(setup)

    original = obs.emit
    obs.emit = _capture  # type: ignore[assignment]
    try:
        await agent.on_user_turn_completed(
            ChatContext.empty(), ChatMessage(role="user", content=["An answer."])
        )
    finally:
        obs.emit = original  # type: ignore[assignment]

    shadow = [f for name, f in emitted if name == obs.EV_SHADOW]
    assert shadow, f"no shadow event emitted; saw {[n for n, _ in emitted]}"
    assert "reason_code" in shadow[0], "the audit trail's ReasonCode is missing"


async def test_shadow_failure_does_not_break_the_turn() -> None:
    from abridgeai.features.interviews.orchestrator import shadow as shadow_mod

    setup = _setup()
    setup.grade_turn = AsyncMock()
    agent = _agent(setup)

    original = shadow_mod.shadow_check_turn

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("shadow exploded")

    native_runtime.shadow_check_turn = _boom  # type: ignore[assignment]
    try:
        # An audit record is never worth an interview.
        await agent.on_user_turn_completed(
            ChatContext.empty(), ChatMessage(role="user", content=["An answer."])
        )
    finally:
        native_runtime.shadow_check_turn = original  # type: ignore[assignment]


# ── the snapshot channel ──────────────────────────────────────────────────────


async def test_join_publishes_a_snapshot(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """A client that reloaded mid-interview learns its question without waiting.

    Nothing else tells it: the agent's words arrive as transcription, which
    carries no question identity.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=_setup()
    )
    try:
        events = fake_session.control_events()
        assert [e["status"] for e in events] == ["snapshot"]
        snapshot = events[0]["snapshot"]
        assert snapshot["current_question_text"] == "What is an index?"
        assert snapshot["outcomes_required"] == 2
        assert snapshot["is_finished"] is False
        assert events[0]["turn_key"] is None, "a snapshot is session-scoped, not turn-scoped"
    finally:
        hard_stop.cancel()


async def test_advancing_the_question_publishes_a_snapshot(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup()
    assert setup.userdata.state is not None
    setup.userdata.state.outcome_coverage["o1"].coverage_points = COVERAGE_SUFFICIENT_POINTS

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        agent = fake_session.started_with["agent"]
        await agent.interview_next_question(SimpleNamespace(userdata=setup.userdata))

        events = fake_session.control_events()
        # The advance publishes a snapshot, then the `agent_action: question`
        # marker that badges the agent's next utterance as the new question.
        assert [e["status"] for e in events] == ["snapshot", "snapshot", "agent_action"]
        assert events[2]["seq"] > events[1]["seq"] > events[0]["seq"], "seq must strictly increase"
        assert events[1]["snapshot"]["current_question_text"] == "What is a covering index?"
        assert events[2]["turn_action"] == "question"
    finally:
        hard_stop.cancel()


async def test_the_hard_stop_tells_the_client_the_interview_ended(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """Both termination routes must reach the client, not just the model's.

    The timer submits without any model turn, so if only the end tool announced
    the finish a timed-out session would leave the UI waiting forever.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup(closer=AsyncMock(return_value="Thanks, that is all."))

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        await hard_stop._stop()  # noqa: SLF001 - the deadline firing, without waiting for it

        finished = [
            e for e in fake_session.control_events() if e["snapshot"]["is_finished"] is True
        ]
        assert finished, "the client was never told the interview ended"
        assert setup.userdata.finished is True
    finally:
        hard_stop.cancel()


async def test_the_question_total_cannot_drift(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """`question_number + questions_remaining` must stay the bank size.

    `remaining()` filters on a FROZENSET of asked ids, so a repeated append never
    reduced it while a length-based asked count grew — the header walked from
    "1 of 3" to "3 of 4" mid-interview. `sync_question_history` merges the REST
    transcript's ids into the same list, so the collision is routine.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup()
    state = setup.userdata.state
    assert state is not None

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        picked = setup.selector()
        assert picked is not None
        already = list(state.asked_question_ids)
        setup.selector.state.asked_question_ids.append(already[-1])  # a REST-merge collision
        setup.userdata.questions_remaining = setup.selector.remaining()

        await userdata_publish(setup)
        snapshot = fake_session.control_events()[-1]["snapshot"]
        assert snapshot["question_number"] + snapshot["questions_remaining"] == 2, snapshot
    finally:
        hard_stop.cancel()


async def userdata_publish(setup: native_bridge.NativeSetup) -> None:
    await setup.userdata.publish_state()


async def test_the_conversation_is_persisted(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """Every committed chat item must reach `interview_session_messages`.

    Nothing on the native path wrote that table: the routed path recorded turns as
    a side effect of `take_session_step`, which this agent never calls. A finished
    interview therefore stored only the REST onboarding turns and the closing, and
    the evaluation and gap report — which read that table — had no answers to judge.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    recorded: list[dict[str, Any]] = []

    async def _capture(session_id: Any, **kwargs: Any) -> None:
        recorded.append({"session_id": session_id, **kwargs})

    monkeypatch.setattr(native_runtime, "record_turn", _capture)
    setup = _setup()
    assert setup.userdata.state is not None
    setup.userdata.state.current_question_id = "3f2a1b4c-0000-4000-8000-000000000001"

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        fake_session.emit(
            "conversation_item_added",
            SimpleNamespace(item=SimpleNamespace(role="user", text_content="My answer")),
        )
        fake_session.emit(
            "conversation_item_added",
            SimpleNamespace(item=SimpleNamespace(role="assistant", text_content="A follow-up")),
        )
        await asyncio.sleep(0)

        assert [(entry["role"], entry["text"]) for entry in recorded] == [
            ("user", "My answer"),
            ("assistant", "A follow-up"),
        ]
        # A user item is attributed to the question it was FOLDED against — the
        # snapshot taken before the server advances — not to whatever question is
        # live when the event handler runs. Reading the live value filed every
        # answer one question ahead (Q1's answer under Q2, and so on), which made
        # the evaluation grade each answer against the wrong outcome and count a
        # fully-answered session as half-answered. An assistant item is the agent
        # speaking, so it tracks the LIVE question.
        assert str(recorded[0]["session_question_id"]) == "3f2a1b4c-0000-4000-8000-000000000001"
        assert str(recorded[1]["session_question_id"]) == "3f2a1b4c-0000-4000-8000-000000000001"
    finally:
        hard_stop.cancel()


# ── the server owns the transition ────────────────────────────────────────────
#
# The model kept advancing WITHOUT calling `interview_next_question`: it said
# "Thanks, Duy. Let's look at the next scenario…" and asked a new question in its
# own words, so server-side the interview stayed on question one. The card, the
# "n of 3" counter and the FOLLOW-UP labels all described a question nobody was
# answering, and the new answer was graded against the previous outcome. Prompt
# hardening did not fix it — no `voice.tool_refused` appears for those sessions
# because the gate was never asked. `native_advance` moves the decision to the
# server; these tests pin that it is the SERVER that advances, that it does not
# advance early, and that the model calling the tool afterwards cannot skip a
# question.


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    job_ctx: SimpleNamespace,
    setup: native_bridge.NativeSetup,
) -> tuple[FakeSession, Any, Any]:
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    return fake_session, fake_session.started_with["agent"], hard_stop


def _grades_to(setup: native_bridge.NativeSetup, points: int) -> None:
    state = setup.userdata.state
    assert state is not None

    async def _grade(**_kwargs: Any) -> None:
        state.outcome_coverage["o1"].coverage_points = points

    setup.grade_turn = _grade


async def test_a_resolved_question_advances_without_the_model_calling_the_tool(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    setup = _setup()
    _grades_to(setup, COVERAGE_SUFFICIENT_POINTS)
    fake_session, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="An index speeds up lookups.")

        assert setup.userdata.current_question_text == "What is a covering index?"
        assert setup.userdata.questions_remaining == 1
        assert setup.userdata.pending_new_question is True
        # Last control event is the question marker; the snapshot sits before it.
        snapshot = fake_session.control_events()[-2]["snapshot"]
        assert snapshot["current_question_text"] == "What is a covering index?"
        assert snapshot["question_number"] == 1, "the card must move with the interviewer"
    finally:
        hard_stop.cancel()


async def test_an_unresolved_question_holds_and_charges_a_follow_up(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """Nothing on the native path was charging the follow-up budget.

    `current_question_follow_up_count` is incremented only by the routed path's
    `apply_state_updates`, so the note reported "0/2" for the whole session and the
    budget's escape hatch never fired — a candidate whose outcome never ticks stayed
    on question one indefinitely.
    """
    setup = _setup()
    _grades_to(setup, 0)
    fake_session, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="I'm not sure.")

        state = setup.userdata.state
        assert state is not None
        assert setup.userdata.current_question_text == "What is an index?"
        assert setup.userdata.pending_new_question is False
        assert state.current_question_follow_up_count == 1
        assert [e["status"] for e in fake_session.control_events()] == ["snapshot"], (
            "holding the question must not publish a new one"
        )
    finally:
        hard_stop.cancel()


async def test_a_spent_follow_up_budget_forces_the_advance(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    setup = _setup()
    _grades_to(setup, 0)
    state = setup.userdata.state
    assert state is not None
    state.current_question_follow_up_count = setup.userdata.max_follow_ups_per_question
    _fake, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="Still not sure.")

        assert setup.userdata.current_question_text == "What is a covering index?"
        assert state.current_question_follow_up_count == 0, "the budget belongs to ONE question"
    finally:
        hard_stop.cancel()


async def test_the_server_does_not_advance_at_the_buzzer(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """`resolve_next_question` stands open below the closing threshold — right for a
    model asking permission, wrong as a trigger: it would put a fresh question to
    the candidate with seconds left instead of closing."""
    setup = _setup()
    setup.userdata.below_closing_threshold = True
    _grades_to(setup, COVERAGE_SUFFICIENT_POINTS)
    _fake, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="An index speeds up lookups.")

        assert setup.userdata.current_question_text == "What is an index?"
        assert setup.userdata.pending_new_question is False
    finally:
        hard_stop.cancel()


async def test_an_empty_bank_advances_nowhere(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    setup = _setup()
    state = setup.userdata.state
    assert state is not None
    state.asked_question_ids = ["q1", "q2"]
    _grades_to(setup, COVERAGE_SUFFICIENT_POINTS)
    _fake, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="An index speeds up lookups.")

        assert setup.userdata.questions_remaining == 0
        assert setup.userdata.pending_new_question is False
    finally:
        hard_stop.cancel()


async def test_the_advance_tool_is_a_no_op_right_after_a_server_advance(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """One candidate transition must never spend two of the bank's questions."""
    setup = _setup()
    _grades_to(setup, COVERAGE_SUFFICIENT_POINTS)
    _fake, agent, hard_stop = await _run(monkeypatch, job_ctx, setup)
    try:
        await agent.fold_turn(answer_text="An index speeds up lookups.")
        state = setup.userdata.state
        assert state is not None
        asked_after_advance = list(state.asked_question_ids)

        result = await agent.interview_next_question(SimpleNamespace(userdata=setup.userdata))

        assert "What is a covering index?" in result
        assert state.asked_question_ids == asked_after_advance
        assert setup.userdata.questions_remaining == 1
        assert setup.userdata.pending_new_question is False
    finally:
        hard_stop.cancel()


def test_the_note_tells_the_model_the_server_already_moved_on() -> None:
    setup = _setup()
    setup.userdata.pending_new_question = True
    setup.userdata.current_question_text = "What is a covering index?"

    note = build_state_reminder(setup.userdata)

    assert "ALREADY moved" in note
    assert "What is a covering index?" in note
    assert "You MAY call next_question" not in note
    assert "Do NOT call next_question yet" not in note


def test_the_note_still_demands_the_tool_when_the_server_has_not_moved() -> None:
    setup = _setup()

    note = build_state_reminder(setup.userdata)

    assert "answering THIS question" in note
    assert "ALREADY moved" not in note


async def test_an_answer_is_filed_under_the_question_it_answered_not_the_next_one(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """The reported score-killer: answers shifted one question forward.

    The user answered every question, yet the evaluation paired each answer with
    the WRONG prompt (the outcome evaluations quoted Q2's answer under Q1's
    outcome). Because the advance happens DURING `fold_turn` — before the SDK
    commits the user item — reading `current_question_id` at emit time stamped
    the answer with the question the interview had moved ON to. The answer to
    question one was therefore graded against question two's outcome.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    recorded: list[dict[str, Any]] = []

    async def _capture(session_id: Any, **kwargs: Any) -> None:
        recorded.append({"session_id": session_id, **kwargs})

    monkeypatch.setattr(native_runtime, "record_turn", _capture)
    setup = _setup()
    _grades_to(setup, COVERAGE_SUFFICIENT_POINTS)
    state = setup.userdata.state
    assert state is not None
    state.current_question_id = "3f2a1b4c-0000-4000-8000-000000000001"
    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        agent = fake_session.started_with["agent"]
        await agent.fold_turn(answer_text="An index speeds up lookups.")
        # The fold advanced the interview to question two; the answer item lands
        # AFTER that advance, as in production. Emitted inside a task so the
        # handler's `asyncio.create_task` runs on an active loop.
        emit_task = asyncio.create_task(
            _emit_now(
                fake_session,
                SimpleNamespace(
                    item=SimpleNamespace(role="user", text_content="An index speeds up lookups.")
                ),
            )
        )
        await emit_task

        assert setup.userdata.current_question_text == "What is a covering index?"
        assert len(recorded) == 1
        assert str(recorded[0]["session_question_id"]) == "3f2a1b4c-0000-4000-8000-000000000001", (
            "the answer must stay with the question it answered, not the one the "
            "interview moved on to"
        )
    finally:
        hard_stop.cancel()


async def _emit_now(fake_session: FakeSession, payload: Any) -> None:
    await asyncio.sleep(0)
    fake_session.emit("conversation_item_added", payload)
    await asyncio.sleep(0)


# ── an end refusal pins the todo list into the chat context ──────────────────


async def test_end_refusal_injects_a_user_role_todo_note() -> None:
    """The refusal error is one turn old by the time the model acts on it.

    A user-role message persists for every later turn, and user content is the
    only mid-conversation channel the gateway honours (system content is merged
    at the head only). The note must list the uncovered outcomes by TITLE and
    name the questions left, so a model that just tried to quit is handed its
    checklist rather than a generic "keep going".
    """
    from livekit.agents import ToolError

    setup = _setup()
    agent = _agent(setup)
    updates: list[ChatContext] = []

    async def _update_ctx(chat_ctx: ChatContext, **_kw: Any) -> None:
        updates.append(chat_ctx)

    fake_agent = SimpleNamespace(chat_ctx=agent.chat_ctx, update_chat_ctx=_update_ctx)
    ctx = SimpleNamespace(
        userdata=setup.userdata,
        session=SimpleNamespace(current_agent=fake_agent),
    )

    with pytest.raises(ToolError):
        await agent.interview_end_interview(ctx)

    assert len(updates) == 1, "the refusal must inject exactly one note"
    note = updates[0].items[-1]
    text = note.text_content or ""
    assert getattr(note, "role", None) == "user", (
        "system content cannot be merged mid-conversation; user is the gate"
    )
    assert "Covering indexes" in text, "the note must name the unticked outcome by title"
    assert "2 question(s) left" in text
    assert "not spoken by the candidate" in text, (
        "without the marker the model may mistake the note for candidate speech"
    )


async def test_end_refusal_note_never_blocks_the_refusal_itself() -> None:
    """A note that fails to land must not cost the model the reason.

    The tool error is the load-bearing half; the injection is reinforcement.
    With no reachable session on the context, the refusal still raises.
    """
    from livekit.agents import ToolError

    setup = _setup()
    agent = _agent(setup)
    ctx = SimpleNamespace(userdata=setup.userdata)  # no .session at all

    with pytest.raises(ToolError):
        await agent.interview_end_interview(ctx)


# ── native turn handling: STT end-of-turn for English, VAD for Vietnamese ─────


def test_native_turn_handling_uses_stt_end_of_turn_for_english() -> None:
    """Flux ships its own phrase-endpointing model; the session must commit
    turns on the STT's end-of-turn signal instead of a separate detector."""
    from abridgeai.features.interviews.realtime.agent_session import _native_turn_handling

    options = _native_turn_handling("en")
    assert options.get("turn_detection") == "stt"
    assert "endpointing" in options, "endpointing bounds are shared with the routed path"


def test_native_turn_handling_keeps_vad_for_vietnamese() -> None:
    """Flux is English-only and the semantic thresholds are English-tuned; a
    wrong verdict would truncate a Vietnamese answer mid-sentence — worse than
    the VAD default."""
    from abridgeai.features.interviews.realtime.agent_session import _native_turn_handling

    options = _native_turn_handling("vi")
    assert "turn_detection" not in options
    assert "endpointing" in options


# ── the session-wide follow-up counter is charged on the native path ─────────


def test_count_follow_up_charges_both_budgets() -> None:
    from abridgeai.features.interviews.realtime.native_advance import count_follow_up

    setup = _setup()
    assert setup.userdata.state is not None
    before_total = setup.userdata.state.total_follow_up_count

    count_follow_up(setup.userdata)

    assert setup.userdata.state.current_question_follow_up_count == 1
    assert setup.userdata.state.total_follow_up_count == before_total + 1


# ── an advance pins "ask the new question" into the chat context ─────────────


async def test_fold_turn_after_an_advance_injects_the_new_question_directive(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """The model kept debating the OLD question after the server moved on.

    Production session 1d629118: the card advanced to Q2 at 1:27 while the
    interviewer probed Q1 for three more exchanges and Q2 was never asked. The
    state note alone was not enough for the model, so an advance now also pins
    a user-role directive (the one mid-conversation channel this gateway
    honours) naming the new question and forbidding the old one.
    """
    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup()
    assert setup.userdata.state is not None
    # Cover the outcome AND spend the per-question budget, so the post-grade
    # advance fires on this fold.
    setup.userdata.state.outcome_coverage["o1"].coverage_points = COVERAGE_SUFFICIENT_POINTS
    setup.userdata.state.current_question_follow_up_count = 2

    updates: list[ChatContext] = []

    async def _update_ctx(_self: Any, chat_ctx: ChatContext, **_kw: Any) -> None:
        updates.append(chat_ctx)

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        agent = fake_session.started_with["agent"]
        monkeypatch.setattr(type(agent), "update_chat_ctx", _update_ctx, raising=False)
        # Not a property on the instance: attach a stub readable chat_ctx.
        agent.__dict__.setdefault("chat_ctx", agent.chat_ctx)

        await agent.fold_turn(answer_text="A covering index covers the query.")

        directives = [
            ctx.items[-1]
            for ctx in updates
            if "ALREADY moved" in (ctx.items[-1].text_content or "")
        ]
        assert directives, "the advance must pin the ask-the-new-question directive"
        text = directives[0].text_content or ""
        assert getattr(directives[0], "role", None) == "user"
        assert "What is a covering index?" in text
        assert "not spoken by the candidate" in text
    finally:
        hard_stop.cancel()


async def test_end_is_refused_while_a_selected_question_is_still_unasked() -> None:
    """The empty-bank branch of the end gate must not fire mid-transition.

    Session e972fade: the advance selected Q3 (bank behind it now empty), the
    model called end_interview 0.8s later, the gate saw remaining=0 and
    allowed it — Q3 was never asked. A selected-but-unasked question is a
    question that remains.
    """
    from livekit.agents import ToolError

    setup = _setup()
    setup.userdata.pending_new_question = True
    setup.userdata.questions_remaining = 0
    agent = _agent(setup)
    ctx = SimpleNamespace(userdata=setup.userdata)  # no .session: note path is guarded

    with pytest.raises(ToolError, match="has NOT been asked"):
        await agent.interview_end_interview(ctx)
    assert setup.userdata.finished is False


async def test_advance_directive_replaces_the_previous_one(
    monkeypatch: pytest.MonkeyPatch, job_ctx: SimpleNamespace
) -> None:
    """Stacked directives contradicted each other ("ask Q2" vs "ask Q3")."""

    fake_session = FakeSession()
    monkeypatch.setattr(native_runtime, "build_native_session", lambda *_a, **_kw: fake_session)
    setup = _setup()
    assert setup.userdata.state is not None
    setup.userdata.state.outcome_coverage["o1"].coverage_points = COVERAGE_SUFFICIENT_POINTS
    setup.userdata.state.current_question_follow_up_count = 2

    updates: list[ChatContext] = []

    async def _update_ctx(_self: Any, chat_ctx: ChatContext, **_kw: Any) -> None:
        updates.append(chat_ctx)

    hard_stop = await native_runtime.run_native_interview(
        job_ctx, settings=SimpleNamespace(), setup=setup
    )
    try:
        agent = fake_session.started_with["agent"]
        monkeypatch.setattr(type(agent), "update_chat_ctx", _update_ctx, raising=False)

        # First fold: advances to Q2 and injects a directive naming it.
        await agent.fold_turn(answer_text="An index speeds up lookups.")
        assert len(updates) == 1
        # Seed the context with that first directive visible to the agent.
        agent.__dict__.setdefault("chat_ctx", updates[0])

        # Second fold on the (now resolved) next question: advances again.
        setup.userdata.state.outcome_coverage["o2"] = setup.userdata.state.outcome_coverage.get(
            "o2"
        ) or type(setup.userdata.state.outcome_coverage["o1"])(outcome_id="o2", coverage_points=0)
        setup.userdata.state.outcome_coverage["o2"].coverage_points = COVERAGE_SUFFICIENT_POINTS
        setup.userdata.state.current_outcome_id = "o2"
        setup.userdata.state.current_question_follow_up_count = 2
        setup.userdata.last_advance_monotonic = None  # outside the coalesce window
        setup.userdata.questions_remaining = 1

        await agent.fold_turn(answer_text="A covering index covers the query.")

        directives = [
            [
                item
                for item in ctx.items
                if item.text_content and "ALREADY moved" in item.text_content
            ]
            for ctx in updates
        ]
        last = directives[-1]
        assert len(last) == 1, "the stale directive must be replaced, not stacked"
    finally:
        hard_stop.cancel()


def test_control_note_filter_tolerates_non_message_items() -> None:
    """`update_instructions` appends AgentConfigUpdate items to the context.

    Those have no ``text_content``; a direct attribute read raised
    AttributeError and killed every directive injection after the first
    state-note refresh (session 564334c0: Q3 selected, directive never landed,
    interviewer went silent).
    """
    from types import SimpleNamespace

    from livekit.agents import ChatMessage

    from abridgeai.features.interviews.realtime.native_runtime import (
        _control_notes_removed,
    )

    ctx = ChatContext(
        items=[
            ChatMessage(role="user", content=["hello"]),
            SimpleNamespace(type="AgentConfigUpdate", instructions="new instructions"),
            ChatMessage(
                role="user",
                content=[
                    "[interview control — not spoken by the candidate] The "
                    "server has ALREADY moved..."
                ],
            ),
        ]
    )
    kept = _control_notes_removed(ctx)
    texts = [getattr(i, "text_content", None) for i in kept]
    assert "hello" in texts
    assert not any(t and "ALREADY moved" in t for t in texts)
    # The non-message item passes through untouched.
    assert any(getattr(i, "type", None) == "AgentConfigUpdate" for i in kept)
