"""Unit tests for the thinking filler and persona-aware voice construction.

The filler exists because the runtime has no pacing at all: it speaks as soon
as the brain returns and is silent until then, so a slow turn is dead air. The
research position is narrow and worth encoding in tests — a *filled pause* as a
standalone token raises perceived naturalness, while *disfluency spliced into
the answer* degrades TTS prosody. So the filler must be its own short utterance,
must not enter the transcript, and must not fire when there was no wait to fill.

Exercises ``InterviewAgent._speak_thinking_filler`` and
``build_agent_session``/``speech_rate_from_verbosity`` directly — no LiveKit
room, no live services.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import observability as obs
from abridgeai.features.interviews.realtime import session_runtime as sr


class _FakeLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


class _FakeSession:
    def __init__(self) -> None:
        self.said: list[tuple[str, dict[str, Any]]] = []

    def say(self, text: str, **kwargs: Any) -> object:
        self.said.append((text, kwargs))
        return object()


def _agent(language: str = "en") -> sr.InterviewAgent:
    agent = sr.InterviewAgent(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        first_question_text="hi",
        language=language,
    )
    # ``Agent.session`` is a property backed by the LiveKit runtime; swap in a
    # double so the filler can be exercised without a room.
    object.__setattr__(agent, "_fake_session", _FakeSession())
    type(agent).session = property(lambda self: self._fake_session)  # type: ignore[assignment]
    return agent


@pytest.fixture(autouse=True)
def _restore_session_property() -> Any:
    original = getattr(sr.InterviewAgent, "session", None)
    yield
    if original is not None:
        sr.InterviewAgent.session = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_fast_turn_hears_no_filler() -> None:
    """Cancelled before the delay elapses → nothing spoken.

    This is the property that keeps the feature from becoming a latency tax on
    turns that were already fast.
    """
    agent = _agent()
    task = asyncio.create_task(agent._speak_thinking_filler("t1"))
    await asyncio.sleep(0)  # let the task start and reach its sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert agent.session.said == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", sr._THINKING_FILLER_EN), ("vi", sr._THINKING_FILLER_VI)],
)
async def test_slow_turn_hears_the_filler_in_its_language(
    language: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sr, "_THINKING_FILLER_DELAY_S", 0.0)
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "_logger", fake, raising=False)

    agent = _agent(language)
    await agent._speak_thinking_filler("t1")

    assert [text for text, _ in agent.session.said] == [expected]


@pytest.mark.asyncio
async def test_filler_never_enters_the_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is a social signal, not interview content.

    The post-session evaluator grades the stored transcript; an acknowledgement
    token landing there would be scored as an interviewer turn.
    """
    monkeypatch.setattr(sr, "_THINKING_FILLER_DELAY_S", 0.0)
    agent = _agent()
    await agent._speak_thinking_filler("t1")

    _, kwargs = agent.session.said[0]
    assert kwargs.get("add_to_chat_ctx") is False


@pytest.mark.asyncio
async def test_filler_failure_never_breaks_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sr, "_THINKING_FILLER_DELAY_S", 0.0)
    agent = _agent()

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("tts down")

    agent.session.say = _boom  # type: ignore[method-assign]
    await agent._speak_thinking_filler("t1")  # must not raise


@pytest.mark.parametrize(
    ("verbosity", "expected"),
    [(0, 0.83), (1, 0.91), (2, 1.0), (3, 1.09), (4, 1.17)],
)
def test_speech_rate_tracks_the_frontend_wpm_curve(verbosity: int, expected: float) -> None:
    """Mirrors ``wordsPerMinuteFromTraits`` (124 + 13·verbosity) as a ratio.

    Pinned so the LiveKit voice and the browser-narration voice cannot drift
    into pacing the same persona differently.
    """
    assert sr.speech_rate_from_verbosity(verbosity) == expected


@pytest.mark.parametrize("verbosity", [-5, 99])
def test_speech_rate_clamps_out_of_range_dials(verbosity: int) -> None:
    """Speed is a hard multiplier; an extreme value is worse than a flat one."""
    assert 0.8 <= sr.speech_rate_from_verbosity(verbosity) <= 1.2


def test_neutral_persona_is_exactly_unmodified_speed() -> None:
    """The mid dial must be 1.0, so the default deployment sounds unchanged."""
    assert sr.speech_rate_from_verbosity(2) == 1.0
