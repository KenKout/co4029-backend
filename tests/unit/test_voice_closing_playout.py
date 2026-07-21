"""Unit tests for playback-aware closing (Phase 8).

The runtime must wait for the closing utterance to finish PLAYING OUT before it
tears the room down — otherwise the room can be shut mid-closing (the §5C
step-5 risk). These tests exercise ``InterviewAgent._await_closing_playout``
directly with fake SpeechHandles (no LiveKit room, no live services):

  * happy path: a handle that completes → completed=True, event emitted;
  * a handle whose playout hangs → bounded by the timeout → timed_out=True,
    and shutdown is NOT blocked forever;
  * a None handle (nothing was spoken) → no crash, event still emitted;
  * a handle whose wait raises → swallowed, event still emitted.

The event is captured by swapping the observability logger.
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


def _agent() -> sr.InterviewAgent:
    return sr.InterviewAgent(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        first_question_text="hi",
        language="en",
    )


def _playout_event(fake: _FakeLogger) -> dict[str, Any]:
    matches = [kw for name, kw in fake.calls if name == obs.EV_CLOSING_PLAYOUT]
    assert len(matches) == 1, f"expected exactly one closing_playout event, got {len(matches)}"
    return matches[0]


class _CompletingHandle:
    def __init__(self) -> None:
        self.awaited = False

    async def wait_for_playout(self) -> None:
        self.awaited = True  # returns immediately → playout finished

    def __await__(self):  # type: ignore[no-untyped-def]
        # The runtime does ``await session.say(...)``; make the fake awaitable.
        async def _noop() -> None:
            return None

        return _noop().__await__()


class _HangingHandle:
    async def wait_for_playout(self) -> None:
        await asyncio.sleep(3600)  # never completes within the timeout


class _RaisingHandle:
    async def wait_for_playout(self) -> None:
        raise RuntimeError("transport gone")


@pytest.mark.asyncio
async def test_playout_waits_for_completion(monkeypatch: Any) -> None:
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)
    agent = _agent()
    handle = _CompletingHandle()

    await agent._await_closing_playout(handle, turn_id="t1")

    assert handle.awaited is True
    ev = _playout_event(fake)
    assert ev["completed"] is True
    assert ev["timed_out"] is False


@pytest.mark.asyncio
async def test_playout_bounded_by_timeout(monkeypatch: Any) -> None:
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)
    # Shrink the ceiling so the test is fast; the real value is 30s.
    monkeypatch.setattr(sr, "_CLOSING_PLAYOUT_TIMEOUT_S", 0.05)
    agent = _agent()

    # Must return promptly (not hang) even though the handle never completes.
    await asyncio.wait_for(
        agent._await_closing_playout(_HangingHandle(), turn_id="t2"), timeout=5.0
    )

    ev = _playout_event(fake)
    assert ev["timed_out"] is True
    assert ev["completed"] is False


@pytest.mark.asyncio
async def test_playout_none_handle_is_safe(monkeypatch: Any) -> None:
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)
    agent = _agent()

    await agent._await_closing_playout(None, turn_id="t3")

    ev = _playout_event(fake)
    assert ev["completed"] is False
    assert ev["timed_out"] is False
    assert ev.get("playout_ms") is None


@pytest.mark.asyncio
async def test_playout_swallows_handle_errors(monkeypatch: Any) -> None:
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)
    agent = _agent()

    # Must not raise — a failing playout wait can't crash the job.
    await agent._await_closing_playout(_RaisingHandle(), turn_id="t4")

    ev = _playout_event(fake)
    assert ev["completed"] is False
    assert ev["timed_out"] is False


class _FakeSession:
    """Captures session.say() calls with their kwargs and returns a completing
    SpeechHandle, so we can assert the closing is forced non-interruptible."""

    def __init__(self) -> None:
        self.says: list[tuple[str, dict[str, Any]]] = []

    def say(self, text: str, **kwargs: Any) -> _CompletingHandle:
        self.says.append((text, kwargs))
        return _CompletingHandle()


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


@pytest.mark.asyncio
async def test_opening_plays_out_before_first_question(monkeypatch: Any) -> None:
    agent = sr.InterviewAgent(
        interview_session_id=uuid4(),
        student_id=uuid4(),
        opening_text="Welcome to the interview.",
        first_question_text="Tell me about dependency injection.",
        language="en",
    )
    session = _FakeSession()
    monkeypatch.setattr(type(agent), "session", property(lambda self: session))

    await agent.on_enter()

    assert [text for text, _kwargs in session.says] == [
        "Welcome to the interview.",
        "Tell me about dependency injection.",
    ]
    assert session.says[0][1]["allow_interruptions"] is False
    assert session.says[1][1]["allow_interruptions"] is False


def bridge_result(*, speak_text: str, is_finished: bool, suppress_default_closing: bool) -> Any:
    from abridgeai.features.interviews.realtime.orchestration_bridge import TurnResult

    return TurnResult(
        speak_text=speak_text,
        is_finished=is_finished,
        suppress_default_closing=suppress_default_closing,
    )


async def _run_turn_with(monkeypatch: Any, session: _FakeSession, result: Any) -> None:
    """Drive InterviewAgent.on_user_turn_completed once with a stubbed bridge
    result and a fake session, swallowing the terminal StopResponse."""
    import contextlib

    from livekit.agents import StopResponse

    agent = _agent()

    async def _fake_handle_turn(*a: Any, **k: Any) -> Any:
        return result

    monkeypatch.setattr(sr.bridge, "handle_student_turn", _fake_handle_turn)
    monkeypatch.setattr(sr, "get_job_context", lambda required=False: None)
    # Attach the fake session (property on the livekit Agent base is read-only,
    # so set the private slot the base uses).
    monkeypatch.setattr(type(agent), "session", property(lambda self: session))

    with contextlib.suppress(StopResponse):
        await agent.on_user_turn_completed(object(), _FakeMessage("an answer"))


@pytest.mark.asyncio
async def test_finished_turn_forces_non_interruptible_closing(monkeypatch: Any) -> None:
    """The real behavioural fix: on a finished turn the (adaptive) closing is
    spoken with allow_interruptions=False so it can't be cut short; a normal
    turn keeps the session default (no allow_interruptions kwarg)."""
    fake = _FakeLogger()
    monkeypatch.setattr(obs, "logger", fake)

    # Adaptive-closing case: finished + suppress_default_closing True.
    finished = bridge_result(
        speak_text="Thanks, that wraps up our interview.",
        is_finished=True,
        suppress_default_closing=True,
    )
    session = _FakeSession()
    await _run_turn_with(monkeypatch, session, finished)
    # The closing say() must be non-interruptible.
    assert session.says, "expected the closing to be spoken"
    _text, kw = session.says[0]
    assert kw.get("allow_interruptions") is False

    # Normal (non-final) turn: no forced allow_interruptions kwarg.
    normal = bridge_result(
        speak_text="Could you give an example?", is_finished=False, suppress_default_closing=False
    )
    session2 = _FakeSession()
    await _run_turn_with(monkeypatch, session2, normal)
    _t2, kw2 = session2.says[0]
    assert "allow_interruptions" not in kw2
