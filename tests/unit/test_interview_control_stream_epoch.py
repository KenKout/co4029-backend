"""Additive stream_id epoch on the control protocol (plan §4, BE side).

Agent replacement (hard-stop reload, crash recovery) hands the SAME room to a
NEW ControlPublisher whose ``seq`` restarts at 1. A client tracking a global
``lastSeq`` discards every event of the new agent — the acks that would settle
its parked draft arrive numbered "1", "2" and are read as stale. The fix is an
opaque epoch: each publisher owns a ``stream_id`` (UUID) stamped under the same
lock that allocates ``seq``; old frames keep working (legacy = no tag), and the
client's ordering tracker scopes comparisons to the active stream.

These tests pin the BE half: the tag is additive wire-compatible JSON, stamped
centrally in ``_publish``, and constant for the publisher's lifetime.
"""

from __future__ import annotations

import json
from uuid import UUID

from abridgeai.features.interviews.realtime import native_control as nc
from abridgeai.features.interviews.realtime import text_protocol as tp


class _Local:
    """Records sent frames; stands in for the room's local participant."""

    def __init__(self) -> None:
        self.frames: list[str] = []

    async def send_text(self, text: str, *, topic: str) -> None:
        self.frames.append(text)


class _Room:
    def __init__(self, local: _Local) -> None:
        self.local_participant = local


class _RoomIO:
    def __init__(self, room: _Room) -> None:
        self.room = room


class _Session:
    def __init__(self, local: _Local) -> None:
        self.room_io = _RoomIO(_Room(local))


def _publisher(local: _Local) -> tuple[nc.ControlPublisher, _Local]:
    sess = _Session(local)
    pub = nc.ControlPublisher(sess, interview_session_id=UUID(int=1, version=4))
    return pub, local


async def _ack(pub: nc.ControlPublisher) -> None:
    await pub.ack(turn_key="tk-1", turn_action=tp.DEFAULT_TURN_ACTION)


def test_every_frame_carries_a_stream_id() -> None:
    local = _Local()
    pub, local = _publisher(local)

    import asyncio

    asyncio.run(_ack(pub))

    assert len(local.frames) == 1
    payload = json.loads(local.frames[0])
    sid = payload.get("stream_id")
    assert isinstance(sid, str), "stream_id must be a string tag"
    assert sid, "control frame is missing its stream epoch tag"


def test_the_stream_id_is_stable_for_the_publisher_lifetime() -> None:
    import asyncio

    local = _Local()
    pub, local = _publisher(local)
    asyncio.run(_ack(pub))
    asyncio.run(_ack(pub))

    ids = {json.loads(f)["stream_id"] for f in local.frames}
    assert len(ids) == 1, "one publisher must own exactly one epoch"


def test_two_publishers_get_distinct_epochs() -> None:
    import asyncio

    _, l1 = _publisher(_Local())
    _, l2 = _publisher(_Local())
    p1 = nc.ControlPublisher(_Session(l1), interview_session_id=UUID(int=1, version=4))
    p2 = nc.ControlPublisher(_Session(l2), interview_session_id=UUID(int=1, version=4))
    asyncio.run(_ack(p1))
    asyncio.run(_ack(p2))

    id1 = json.loads(l1.frames[0])["stream_id"]
    id2 = json.loads(l2.frames[0])["stream_id"]
    assert id1 != id2, "a replacement agent must present a NEW epoch"


def test_frames_stay_legacy_parseable_and_seq_ordering_unchanged() -> None:
    """Wire compatibility: seq still allocates; old fields unchanged."""
    import asyncio

    local = _Local()
    pub, local = _publisher(local)
    asyncio.run(_ack(pub))
    asyncio.run(_ack(pub))

    first = json.loads(local.frames[0])
    second = json.loads(local.frames[1])
    assert first["seq"] == 1, "seq must keep allocating inside the lock"
    assert second["seq"] == 2, "the second frame takes the next seq"
    assert first["status"] == tp.ControlStatus.ACCEPTED.value
    assert first["turn_key"] == "tk-1"


def test_stream_id_is_a_valid_uuid() -> None:
    import asyncio

    local = _Local()
    pub, local = _publisher(local)
    asyncio.run(_ack(pub))

    sid = json.loads(local.frames[0])["stream_id"]
    UUID(sid)  # raises ValueError on a malformed tag
    assert sid


def test_stamp_happens_under_the_same_lock_as_seq() -> None:
    """A snapshot of the source pins the stamp to the seq-allocation block."""
    import inspect

    src = inspect.getsource(nc.ControlPublisher._publish)
    assert "stream_id" in src, "stream_id must be stamped in _publish"
    lock_pos = src.index("async with self._lock")
    stamp_pos = src.index("event.stream_id")
    assert lock_pos < stamp_pos < src.index("send_text"), (
        "the epoch tag must ride the same locked allocation as seq"
    )


def test_control_event_default_has_no_stream_id_before_stamping() -> None:
    """The field is optional and stamping is centralised, not per-caller."""
    ev = tp.ControlEvent(
        status=tp.ControlStatus.ACCEPTED, turn_key=None, seq=0
    )
    payload = json.loads(ev.to_json())
    assert "stream_id" not in payload, "an unstamped event must stay legacy-shaped"
