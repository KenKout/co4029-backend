"""Second probe: does OUR agent's `on_text_input` fire via RoomOptions?

The first probe proved the SDK carries attributes on `lk.chat`. This one proves
the wiring we actually shipped: a real `AgentSession` started with
`RoomOptions(text_input=TextInputOptions(text_input_cb=agent.on_text_input))`
routes a typed turn into `InterviewAgent.on_text_input`, our attribute validation
runs, and our control event is published.

The interview brain is stubbed (`bridge.handle_student_turn`), so this tests
transport + wiring only — grading is covered by unit tests. STT/TTS/VAD are left
out of the session so no provider credits are spent.

Usage:
    .venv/bin/python scripts/probe_agent_text_input.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livekit import api, rtc  # noqa: E402
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.voice.room_io import (  # noqa: E402
    RoomOptions,
    TextInputOptions,
)

from abridgeai.features.interviews.realtime.orchestration_bridge import (  # noqa: E402
    TurnResult,
)
from abridgeai.features.interviews.realtime.session_runtime import (  # noqa: E402
    InterviewAgent,
)

ROOM = f"probe-agent-text-{os.getpid()}"
TOPIC_CHAT = "lk.chat"
TOPIC_CONTROL = "abridge.interview.control"


def _creds() -> tuple[str, str, str]:
    from abridgeai.core.config import get_settings

    s = get_settings()
    assert s.livekit_ws_url
    assert s.livekit_api_key
    assert s.livekit_api_secret
    return (
        s.livekit_ws_url,
        s.livekit_api_key.get_secret_value(),
        s.livekit_api_secret.get_secret_value(),
    )


def _token(key: str, secret: str, identity: str) -> str:
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(minutes=5))
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=ROOM,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def main() -> int:
    url, key, secret = _creds()
    session_id, student_id = uuid4(), uuid4()

    stub_result = TurnResult(
        speak_text="Here is a hint: think about indexes.",
        is_finished=False,
        next_question_text=None,
        followup_text="Here is a hint: think about indexes.",
        state_version=77,
    )

    agent = InterviewAgent(
        interview_session_id=session_id,
        student_id=student_id,
        first_question_text=None,  # no opening speech: no TTS configured
        language="en",
    )

    # No stt/tts/vad: this probe is about the text path. `say()` would need a TTS
    # so the stub returns speak_text but we patch say to a no-op recorder.
    session: AgentSession = AgentSession()

    agent_room = rtc.Room()
    await agent_room.connect(url, _token(key, secret, "probe-agent"))
    print(f"[agent ] room connected: {ROOM}")

    said: list[str] = []
    brain = AsyncMock(return_value=stub_result)

    class _FakeHandle:
        """`say()` returns an awaitable SpeechHandle; the probe has no TTS."""

        def __await__(self):
            async def _noop():
                return None

            return _noop().__await__()

        async def wait_for_playout(self) -> None:
            return None

    def _fake_say(self, text, **kw):  # noqa: ANN001, ANN202
        said.append(text)
        return _FakeHandle()

    with (
        patch(
            "abridgeai.features.interviews.realtime.session_runtime.bridge.handle_student_turn",
            new=brain,
        ),
        patch.object(AgentSession, "say", _fake_say),
    ):
        await session.start(
            agent,
            room=agent_room,
            room_options=RoomOptions(
                text_input=TextInputOptions(text_input_cb=agent.on_text_input),
                # Audio off: nothing to publish/subscribe for a text-only probe.
                audio_input=False,
                audio_output=False,
                text_output=False,
            ),
        )
        print("[agent ] AgentSession started with RoomOptions(text_input=...)")

        client_room = rtc.Room()
        control: list[dict] = []

        def on_control(reader: rtc.TextStreamReader, identity: str) -> None:
            async def _read() -> None:
                control.append(json.loads(await reader.read_all()))

            asyncio.create_task(_read())

        client_room.register_text_stream_handler(TOPIC_CONTROL, on_control)
        await client_room.connect(url, _token(key, secret, "probe-client"))
        print("[client] room connected")
        await asyncio.sleep(1.5)

        await client_room.local_participant.send_text(
            "I am stuck on this question",
            topic=TOPIC_CHAT,
            attributes={"turn_action": "hint", "turn_key": "tk-agentprobe-1"},
        )
        print("[client] sent typed turn (turn_action=hint)")

        for _ in range(80):
            if brain.called and len(control) >= 2:
                break
            await asyncio.sleep(0.25)

        ok = True
        print("\n=== did OUR callback run? ===")
        if not brain.called:
            print("FAIL: bridge.handle_student_turn was never called")
            ok = False
        else:
            kw = brain.call_args.kwargs
            print(f"brain called with turn_action={kw.get('turn_action')!r} "
                  f"turn_id={kw.get('turn_id')!r}")
            if kw.get("turn_action") != "hint":
                print("FAIL: turn_action did not reach the brain")
                ok = False
            if kw.get("turn_id") != "tk-agentprobe-1":
                print("FAIL: client turn_key was not used as the idempotency key")
                ok = False
            if brain.call_args.args[0] != session_id:
                print("FAIL: session_id was not taken from dispatch metadata")
                ok = False
            if ok:
                print("PASS: turn_action + turn_key reached the brain; "
                      "session_id came from the agent, not the client")

        print("\n=== control events observed by the client ===")
        print(json.dumps(control, indent=2))
        statuses = [e.get("status") for e in control]
        if statuses[:2] != ["accepted", "completed"]:
            print(f"FAIL: expected accepted->completed, got {statuses}")
            ok = False
        else:
            print("PASS: accepted -> completed, in order")
        completed = next((e for e in control if e.get("status") == "completed"), None)
        if completed and completed.get("state_version") != 77:
            print(f"FAIL: state_version not the brain's: {completed.get('state_version')}")
            ok = False
        elif completed:
            print("PASS: completed carries the brain's state_version (77)")

        print(f"\nspoken: {said}")
        await client_room.disconnect()

    await session.aclose()
    await agent_room.disconnect()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'PROBE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
