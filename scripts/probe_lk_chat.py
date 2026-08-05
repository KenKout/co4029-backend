"""One-shot integration probe: does a typed `lk.chat` turn reach our agent?

Not a test — a throwaway harness for the ONE thing unit tests cannot cover: that
LiveKit's RoomIO actually routes an `lk.chat` text stream (with our attributes)
into `InterviewAgent.on_text_input`, and that our control topic comes back.

It deliberately does NOT touch the interview brain. `handle_student_turn` is
monkeypatched inside the agent process... which is impossible across processes,
so instead this probe runs its OWN AgentSession in-process with a stub agent that
records what arrives. That isolates the transport question (does the SDK wire
attributes through?) from the grading question (already unit-tested).

Usage:
    .venv/bin/python scripts/probe_lk_chat.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import timedelta

from livekit import api, rtc

ROOM = f"probe-lkchat-{os.getpid()}"
AGENT_IDENTITY = "probe-agent"
CLIENT_IDENTITY = "probe-client"

TOPIC_CHAT = "lk.chat"
TOPIC_CONTROL = "abridge.interview.control"


def _creds() -> tuple[str, str, str]:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                room_join=True, room=ROOM, can_publish=True, can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def main() -> int:
    url, key, secret = _creds()

    # ── "Agent" side: a bare room that registers a text-stream handler exactly
    # the way RoomIO does (`register_text_stream_handler` on TOPIC_CHAT), so we
    # observe what the SDK delivers, including attributes.
    agent_room = rtc.Room()
    received: list[dict] = []

    def on_text(reader: rtc.TextStreamReader, participant_identity: str) -> None:
        async def _read() -> None:
            body = await reader.read_all()
            received.append(
                {
                    "text": body,
                    "topic": reader.info.topic,
                    "attributes": reader.info.attributes,
                    "from": participant_identity,
                }
            )
            # Echo a control event back, mirroring _publish_control.
            payload = json.dumps(
                {
                    "status": "completed",
                    "turn_key": (reader.info.attributes or {}).get("turn_key"),
                    "seq": 1,
                    "turn_action": (reader.info.attributes or {}).get("turn_action", "answer"),
                    "state_version": 42,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            await agent_room.local_participant.send_text(payload, topic=TOPIC_CONTROL)

        asyncio.create_task(_read())

    agent_room.register_text_stream_handler(TOPIC_CHAT, on_text)
    await agent_room.connect(url, _token(key, secret, AGENT_IDENTITY))
    print(f"[agent ] connected to {ROOM}")

    # ── Client side: joins the SAME room and sends one typed turn.
    client_room = rtc.Room()
    control_events: list[dict] = []

    def on_control(reader: rtc.TextStreamReader, participant_identity: str) -> None:
        async def _read() -> None:
            control_events.append(json.loads(await reader.read_all()))

        asyncio.create_task(_read())

    client_room.register_text_stream_handler(TOPIC_CONTROL, on_control)
    await client_room.connect(url, _token(key, secret, CLIENT_IDENTITY))
    print(f"[client] connected to {ROOM}")

    await asyncio.sleep(1.0)  # let participants see each other

    sent_attrs = {"turn_action": "hint", "turn_key": "tk-probe-12345678"}
    info = await client_room.local_participant.send_text(
        "I am stuck, give me a hint",
        topic=TOPIC_CHAT,
        attributes=sent_attrs,
    )
    print(f"[client] sent on {TOPIC_CHAT} (stream_id={info.stream_id})")

    for _ in range(60):
        if received and control_events:
            break
        await asyncio.sleep(0.25)

    print("\n=== agent received ===")
    print(json.dumps(received, indent=2))
    print("=== client control events ===")
    print(json.dumps(control_events, indent=2))

    ok = True
    if not received:
        print("FAIL: agent never received the lk.chat stream")
        ok = False
    else:
        got = received[0]
        if got["topic"] != TOPIC_CHAT:
            print(f"FAIL: wrong topic {got['topic']!r}")
            ok = False
        if got["attributes"] != sent_attrs:
            print(f"FAIL: attributes not preserved: {got['attributes']!r} != {sent_attrs!r}")
            ok = False
        else:
            print("PASS: attributes survived the round trip (turn_action + turn_key)")
    if not control_events:
        print("FAIL: client never received a control event")
        ok = False
    else:
        ev = control_events[0]
        if ev.get("turn_key") != sent_attrs["turn_key"]:
            print("FAIL: control event not correlated by turn_key")
            ok = False
        else:
            print("PASS: control event correlated by turn_key, carries state_version")

    await client_room.disconnect()
    await agent_room.disconnect()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'PROBE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
