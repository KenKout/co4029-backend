"""Live staging pass: typed turn over lk.chat against the REAL deployed agent.

Walks a real hybrid session through onboarding (REST), mints the LiveKit join
token, joins the session room as the student, sends a typed answer on lk.chat
with the control attributes, and verifies the real interview agent (running in
pm2) grades the turn and publishes control events — including the full
InterviewSubmitAnswerResponse in the completed event's `state`.

This is the one thing unit tests cannot cover: the real LiveKit Cloud round
trip through the deployed worker.

Usage:
    uv run python scripts/probe_live_staging.py <session_id> <student_jwt>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livekit import rtc  # noqa: E402

API = os.environ.get("ABRIDGE_API", "http://localhost:8000/api/v1")
TOPIC_CHAT = "lk.chat"
TOPIC_CONTROL = "abridge.interview.control"


def _creds() -> tuple[str, str, str]:
    from abridgeai.core.config import get_settings

    s = get_settings()
    return (
        s.livekit_ws_url,
        s.livekit_api_key.get_secret_value(),
        s.livekit_api_secret.get_secret_value(),
    )


async def main() -> int:
    session_id, jwt = sys.argv[1], sys.argv[2]
    url, key, secret = _creds()

    import httpx

    headers = {"Authorization": f"Bearer {jwt}"}
    async with httpx.AsyncClient(base_url=API, headers=headers, timeout=30) as client:
        # 1. Walk onboarding through readiness.
        stages = [
            ("identity_check", "confirm_identity", "Yes, that's me"),
            ("audio_check", "audio_clear", "I can hear you clearly"),
            ("language_check", "confirm_language", "English"),
            ("preparation", "continue_setup", "I'm ready to start"),
            ("readiness", "ready", "Yes, ready"),
        ]
        first_question = None
        for stage, action, text in stages:
            r = await client.post(
                f"/interview-sessions/{session_id}/onboarding/respond",
                json={
                    "stage": stage,
                    "action": action,
                    "response_text": text,
                    "turn_key": f"staging-onboard-{stage}",
                },
            )
            print(f"[onboard {stage}] -> {r.status_code}")
            if r.status_code != 200:
                print(r.text)
                return 1
            body = r.json()
            if body.get("first_question"):
                first_question = body["first_question"]
        print(f"[onboard] done; first_question id={first_question['id'] if first_question else None}")

        # 2. Mint the LiveKit join token (agent is dispatched on room create).
        r = await client.post(f"/interview-sessions/{session_id}/realtime-token")
        print(f"[token] -> {r.status_code}")
        if r.status_code != 200:
            print(r.text)
            return 1
        token_body = r.json()
        print(f"[token] room={token_body.get('room_name')}")

    # 3. Join the room as the student and send a typed turn.
    room = rtc.Room()
    control: list[dict] = []

    def on_control(reader: rtc.TextStreamReader, identity: str) -> None:
        async def _read() -> None:
            control.append(json.loads(await reader.read_all()))

        asyncio.create_task(_read())

    room.register_text_stream_handler(TOPIC_CONTROL, on_control)
    await room.connect(url, token_body["token"])
    print(f"[client] joined {token_body['room_name']}")
    await asyncio.sleep(3.0)  # let the agent join + open the room

    turn_key = f"staging-answer-{os.getpid()}"
    await room.local_participant.send_text(
        "A database index is a sorted data structure that speeds up lookups; "
        "it trades write cost for faster reads.",
        topic=TOPIC_CHAT,
        attributes={"turn_action": "answer", "turn_key": turn_key},
    )
    print(f"[client] sent typed answer (turn_key={turn_key})")

    # 4. Wait for accepted -> completed on the control topic.
    for _ in range(240):  # up to 60s
        statuses = [e.get("status") for e in control]
        if "completed" in statuses:
            break
        await asyncio.sleep(0.25)

    print("\n=== control events received ===")
    print(json.dumps(control, indent=2))

    ok = True
    statuses = [e.get("status") for e in control]
    if statuses[:2] != ["accepted", "completed"]:
        print(f"FAIL: expected accepted->completed, got {statuses}")
        ok = False
    completed = next((e for e in control if e.get("status") == "completed"), None)
    if not completed:
        print("FAIL: no completed event")
        ok = False
    else:
        if completed.get("turn_key") != turn_key:
            print(f"FAIL: turn_key mismatch {completed.get('turn_key')!r}")
            ok = False
        state = completed.get("state") or {}
        expected_fields = [
            "next_question", "is_finished", "time_remaining_seconds",
            "ai_turn_text", "transition_text", "transition_target",
            "assistance_kind", "pending_confirmation", "interaction_state",
        ]
        missing = [f for f in expected_fields if f not in state]
        if missing:
            print(f"FAIL: completed.state missing fields: {missing}")
            ok = False
        else:
            print("PASS: completed.state carries the full response shape "
                  f"({len(state)} fields incl. next_question="
                  f"{bool(state['next_question'])}), timer={state['time_remaining_seconds']}")
        if state.get("next_question"):
            nq = state["next_question"]
            allowed = {"id", "prompt_text", "question_type"}
            leaked = set(nq) - allowed
            if leaked:
                print(f"FAIL: next_question leaked authoring columns: {leaked}")
                ok = False
            else:
                print("PASS: next_question is the projected public object "
                      f"({nq['question_type']})")

    await room.disconnect()
    print(f"\n{'STAGING PASS OK' if ok else 'STAGING PASS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
