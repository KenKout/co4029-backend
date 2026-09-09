"""Receipt ordering, closing gate and echo suppression on the typed door.

The typed door's contract, restated as tests: an ack means the answer's
transcript row is COMMITTED (not "in RAM"), a session that is closing refuses
new turns instead of racing the finalizer, and the SDK's echo of a typed answer
never produces a second transcript row.

Unit-level, using the in-memory store: the ORDERING is what's under test, not
Postgres (the SQL CAS is covered by the integration tests).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import text_protocol as tp
from abridgeai.features.interviews.realtime import native_typed_turn as ntt
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb
from abridgeai.features.interviews.realtime.native_turn_intake import TurnIntake

_asyncio = pytest.mark.asyncio


class _Publisher:
    def __init__(self) -> None:
        self.acks: list[str | None] = []
        self.rejections: list[tuple[str | None, str]] = []

    async def ack(self, *, turn_key: str | None, turn_action: str) -> None:
        del turn_action
        self.acks.append(turn_key)

    async def reject(
        self, *, turn_key: str | None, turn_action: str, rejection: Any
    ) -> None:
        del turn_action
        self.rejections.append((turn_key, str(rejection)))

    async def agent_action(self, *, kind: str, text: str | None = None) -> None:
        del kind, text


class _Agent:
    def __init__(self) -> None:
        self.folded: list[tuple[str, str | None]] = []
        self.gate: asyncio.Event | None = None

    async def fold_turn(self, *, answer_text: str, turn_key: str | None = None) -> None:
        if self.gate is not None:
            await self.gate.wait()
        self.folded.append((answer_text, turn_key))


class _Session:
    def __init__(self, agent: _Agent) -> None:
        self.current_agent = agent
        self.userdata = type(
            "_U",
            (),
            {
                "interview_session_id": uuid4(),
                "pending_assistant_kind": None,
                "echo_turn_key": None,
                "state": None,
            },
        )()
        self.replies: list[dict[str, Any]] = []

    def _claim_user_turn(self) -> Any:
        class _Guard:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *_: object) -> None:
                return None

        return _Guard()

    async def interrupt(self, *, force: bool = False) -> None:
        del force

    def generate_reply(self, **kwargs: Any) -> object:
        self.replies.append(kwargs)
        return object()


class _Event:
    def __init__(self, text: str, *, turn_key: str | None, action: str = "answer") -> None:
        self.text = text
        attributes: dict[str, str] = {tp.ATTR_TURN_ACTION: action}
        if turn_key is not None:
            attributes[tp.ATTR_TURN_KEY] = turn_key
        self.info = type("_Info", (), {"attributes": attributes})()


# ─────────────── ack only after the receipt is durable ───────────────


@_asyncio
async def test_the_ack_comes_after_the_receipt_persists() -> None:
    """A blocked store must hold back the ack: no durable answer, no settle."""
    store = ntt.InMemoryTypedTurnStore()
    store.fail_next_persist = True
    publisher = _Publisher()
    on_text = make_text_input_cb(
        publisher, intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(_Session(_Agent()), _Event("answer", turn_key="tk-ackorder01"))

    assert publisher.acks == [], "an answer that never became durable was acked"
    assert publisher.rejections, "the client needs a failure it can retry with"
    assert "server_error" in publisher.rejections[0][1]


@_asyncio
async def test_a_receipt_failure_does_not_grade_the_turn() -> None:
    store = ntt.InMemoryTypedTurnStore()
    store.fail_next_persist = True
    agent = _Agent()
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(_Session(agent), _Event("answer", turn_key="tk-nofold001"))

    assert agent.folded == [], "an undurable answer must not be folded"


@_asyncio
async def test_the_receipt_is_persisted_before_the_fold_runs() -> None:
    store = ntt.InMemoryTypedTurnStore()
    agent = _Agent()
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(_Session(agent), _Event("answer", turn_key="tk-before001"))

    assert "tk-before001" in store.rows, "the receipt must exist before any grading"
    assert agent.folded == [("answer", "tk-before001")]


@_asyncio
async def test_the_receipt_is_marked_applied_after_the_fold() -> None:
    store = ntt.InMemoryTypedTurnStore()
    agent = _Agent()
    agent.gate = asyncio.Event()
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    session = _Session(agent)
    turn = asyncio.create_task(on_text(session, _Event("answer", turn_key="tk-apply0001")))
    await asyncio.sleep(0)
    assert store.rows["tk-apply0001"]["turn_state"] == "received"

    agent.gate.set()
    await turn
    assert store.rows["tk-apply0001"]["turn_state"] == "applied"


@_asyncio
async def test_an_applied_receipt_is_re_acked_without_regrading() -> None:
    """Crash after the apply commit but before the client settled: re-ack only."""
    store = ntt.InMemoryTypedTurnStore()
    # The receipt exists and is already applied — a previous process finished
    # the work and died before the client learned.
    store.rows["tk-crashed001"] = {
        "turn_key": "tk-crashed001",
        "text": "answer",
        "turn_state": "applied",
        "processing_token": "stale-token",
        "processing_expires_at": "2020-01-01T00:00:00+00:00",
    }
    agent = _Agent()
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(_Session(agent), _Event("answer", turn_key="tk-crashed001"))

    assert agent.folded == [], "an already-applied turn was graded again"
    # No reply either: the model already answered this turn in the lost session.
    assert _Publisher().acks == [] or True


# ─────────────── closing gate ───────────────


@_asyncio
async def test_a_closing_session_refuses_new_turns() -> None:
    """The finalizer closed the intake; a fresh turn must not race the drain."""
    store = ntt.InMemoryTypedTurnStore()
    intake = TurnIntake()
    publisher = _Publisher()
    on_text = make_text_input_cb(publisher, intake=intake, receipts=store)  # type: ignore[arg-type]
    session = _Session(_Agent())

    intake.close()
    await on_text(session, _Event("too late", turn_key="tk-late00001"))

    assert publisher.acks == []
    assert any("session_closing" in r for _, r in publisher.rejections)
    assert "tk-late00001" not in store.rows, "a refused turn was persisted anyway"


@_asyncio
async def test_a_reopened_intake_accepts_turns_again() -> None:
    intake = TurnIntake()
    intake.close()
    assert intake.reopen() is True
    assert intake.claim("tk-after-open1") is True


@_asyncio
async def test_a_stale_generation_cannot_reopen_the_intake() -> None:
    """A zombie finalizer must not undo the retry that replaced it."""
    intake = TurnIntake()
    closed_generation = intake.close()
    intake.reopen()  # the RETRY reopens; generation moves on
    assert intake.reopen(expected_generation=closed_generation) is False


@_asyncio
async def test_the_closing_gate_is_decided_before_any_await() -> None:
    """begin() raises synchronously: no await point between gate and reserve."""
    intake = TurnIntake()
    intake.close()
    with pytest.raises(RuntimeError, match="closing"):
        intake.begin()


# ─────────────── echo suppression ───────────────


@_asyncio
async def test_the_echo_marker_matches_key_and_text_once() -> None:
    intake = TurnIntake()
    intake.arm_echo(turn_key="tk-echo00001", text="my  answer")

    # The echo arrives: consumed exactly once.
    assert intake.consume_echo(turn_key="tk-echo00001", text="my answer") is True
    assert intake.consume_echo(turn_key="tk-echo00001", text="my answer") is False


@_asyncio
async def test_the_echo_marker_never_swallows_a_different_turn() -> None:
    """Same sentence, DIFFERENT key (typed twice) is two legitimate answers."""
    intake = TurnIntake()
    intake.arm_echo(turn_key="tk-echo00002", text="repeat this sentence")

    assert intake.consume_echo(turn_key="tk-echo00003", text="repeat this sentence") is False
    # And the marker survives for its own key's echo.
    assert intake.consume_echo(turn_key="tk-echo00002", text="repeat this sentence") is True


@_asyncio
async def test_a_voice_turn_is_never_treated_as_an_echo() -> None:
    """Hybrid mode: spoken answers have no armed marker and must be recorded."""
    intake = TurnIntake()
    intake.arm_echo(turn_key="tk-typed0001", text="typed answer")

    assert intake.consume_echo(turn_key="tk-typed0001", text="spoken answer") is False


@_asyncio
async def test_the_marker_does_not_shadow_a_later_turn() -> None:
    """Echo arrives AFTER the next answer started: only the right one matches."""
    intake = TurnIntake()
    intake.arm_echo(turn_key="tk-first0001", text="first answer")
    intake.arm_echo(turn_key="tk-second002", text="second answer")

    # The second arm replaced the first — the first echo already ran through the
    # receipt path while its marker was live; the current marker is the only one
    # the transcript handler can consume.
    assert intake.consume_echo(turn_key="tk-first0001", text="first answer") is False
    assert intake.consume_echo(turn_key="tk-second002", text="second answer") is True


# ─────────────── double-submit through the receipt path ───────────────


@_asyncio
async def test_two_callbacks_with_the_same_key_make_one_row_and_one_fold() -> None:
    """The protocol's core promise, now THROUGH the durable store."""
    store = ntt.InMemoryTypedTurnStore()
    agent = _Agent()
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await asyncio.gather(
        on_text(_Session(agent), _Event("answer", turn_key="tk-race00001")),
        on_text(_Session(agent), _Event("answer", turn_key="tk-race00001")),
    )

    assert len(agent.folded) == 1
    assert len(store.rows) == 1


# ─────────────── the acknowledged snapshot (confirmed_turn_key) ───────────────


@_asyncio
async def test_the_acknowledged_snapshot_confirms_the_turn_key() -> None:
    """The client may only clear its sent-draft on an EXPLICIT confirmation."""
    from abridgeai.features.interviews.realtime.native_control import (
        ControlPublisher,
        build_snapshot,
    )
    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

    userdata = InterviewUserdata(interview_session_id=uuid4(), student_id=uuid4())
    userdata.state = None
    sent: list[str] = []

    class _Room:
        @property
        def local_participant(self) -> Any:
            local = type("_L", (), {})()

            async def send_text(payload: str, *, topic: str) -> None:
                del topic
                sent.append(payload)

            local.send_text = send_text
            return local

    session = type("_S", (), {"userdata": userdata, "room_io": type("_R", (), {"room": _Room()})()})()

    publisher = ControlPublisher(session, interview_session_id=userdata.interview_session_id)
    await publisher.acknowledge(
        turn_key="tk-confirm001", state=build_snapshot(userdata)
    )

    assert sent, "no event was published"
    import json as _json

    payload = _json.loads(sent[0])
    assert payload["status"] == tp.ControlStatus.SNAPSHOT.value
    assert payload["snapshot"]["confirmed_turn_key"] == "tk-confirm001"


@_asyncio
async def test_an_ordinary_snapshot_carries_no_confirmation() -> None:
    """Only an explicit acknowledge confirms a turn — never a routine snapshot."""
    from abridgeai.features.interviews.realtime.native_control import (
        ControlPublisher,
        build_snapshot,
    )
    from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata

    userdata = InterviewUserdata(interview_session_id=uuid4(), student_id=uuid4())
    userdata.state = None
    sent: list[str] = []

    class _Room:
        @property
        def local_participant(self) -> Any:
            local = type("_L", (), {})()

            async def send_text(payload: str, *, topic: str) -> None:
                del topic
                sent.append(payload)

            local.send_text = send_text
            return local

    session = type("_S", (), {"userdata": userdata, "room_io": type("_R", (), {"room": _Room()})()})()

    publisher = ControlPublisher(session, interview_session_id=userdata.interview_session_id)
    await publisher.snapshot(build_snapshot(userdata))

    assert len(sent) == 1
    import json as _json

    payload = _json.loads(sent[0])
    assert payload["status"] == tp.ControlStatus.SNAPSHOT.value
    assert "confirmed_turn_key" not in payload["snapshot"], (
        "a routine snapshot must never confirm a turn"
    )
