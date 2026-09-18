"""Concurrent DISTINCT typed turns (audit P1 #13, promoted P0 round).

Two different keys arriving together must be SERIALIZED through the
receipt→fold→settle pipeline: K2's fold has to see the state K1's fold
committed, or K2 grades/advances against a stale question and the
transcript/evaluation disagree. The echo marker is additionally a
per-key slot, not a singleton — a singleton lets K2's arm_echo overwrite
K1's marker before the SDK delivers K1's echo, double-recording K1.

Unit-level, in-memory store: the SERIALIZATION is what's under test (the
Postgres CAS lives in the integration suite).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import native_typed_turn as ntt
from abridgeai.features.interviews.realtime import text_protocol as tp
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

    async def fail(
        self, *, turn_key: str | None, turn_action: str, error_class: str
    ) -> None:
        del turn_key, turn_action, error_class

    async def agent_action(self, *, kind: str, text: str | None = None) -> None:
        del kind, text


class _Agent:
    """fold_turn with a barrier so K1 holds the pipeline mid-fold."""

    def __init__(self) -> None:
        self.folded: list[str | None] = []
        self.fold_started: list[str | None] = []
        self.gate: asyncio.Event | None = None

    async def fold_turn(self, **kwargs: Any) -> None:
        self.fold_started.append(kwargs.get("turn_key"))
        if self.gate is not None:
            await self.gate.wait()
        self.folded.append(kwargs.get("turn_key"))


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


@_asyncio
async def test_distinct_turns_serialize_k2_fold_sees_k1_committed() -> None:
    """K1 mid-fold (parked on a gate); K2 arrives. K2's fold must not run
    until K1's fold has COMPLETED — the pipeline is single-flight per
    session."""
    store = ntt.InMemoryTypedTurnStore()
    intake = TurnIntake()
    agent = _Agent()
    agent.gate = asyncio.Event()
    on_text = make_text_input_cb(
        _Publisher(), intake=intake, receipts=store  # type: ignore[arg-type]
    )
    session = _Session(agent)

    k1 = asyncio.create_task(on_text(session, _Event("answer one", turn_key="tk-serial-k01")))
    # K1 is parked inside its fold (receipt persisted, fold started).
    while agent.fold_started == []:  # noqa: ASYNC110 - spin until the fold park lands
        await asyncio.sleep(0)

    k2 = asyncio.create_task(on_text(session, _Event("answer two", turn_key="tk-serial-k02")))
    # Give K2 a real chance to misbehave: while K1 is parked MID-FOLD, K2 must
    # not even START its fold (a stale-question grade) — the pipeline is
    # single-flight from receipt through settle.
    await asyncio.sleep(0.05)
    assert agent.fold_started == ["tk-serial-k01"], (
        f"K2's fold started while K1 was mid-fold: {agent.fold_started}"
    )

    agent.gate.set()
    await asyncio.gather(k1, k2)

    # Serialized: K1's fold completes before K2's fold starts.
    assert agent.folded == ["tk-serial-k01", "tk-serial-k02"]
    assert store.rows["tk-serial-k01"]["turn_state"] == "applied"
    assert store.rows["tk-serial-k02"]["turn_state"] == "applied"


@_asyncio
async def test_echo_markers_survive_concurrent_distinct_keys() -> None:
    """arm_echo is per-key: K2's marker must not erase K1's.

    With the pipeline mutex the two turns arm their markers SEQUENTIALLY
    (K2 waits for K1's fold), so the regression this guards is the data
    structure: both slots coexist and each consumes independently — a
    singleton marker loses K1's the moment K2 arms.
    """
    store = ntt.InMemoryTypedTurnStore()
    intake = TurnIntake()
    agent = _Agent()
    on_text = make_text_input_cb(
        _Publisher(), intake=intake, receipts=store  # type: ignore[arg-type]
    )
    session = _Session(agent)

    await on_text(session, _Event("first answer", turn_key="tk-echo-k001"))
    await on_text(session, _Event("second answer", turn_key="tk-echo-k002"))

    # Both markers were live (each turn consumed its own echo inside the
    # callback) — the observable contract: each answer produced exactly one
    # transcript row (the receipt) and one consume.
    assert store.rows["tk-echo-k001"]["turn_state"] == "applied"
    assert store.rows["tk-echo-k002"]["turn_state"] == "applied"

    # Slot semantics directly: independent arm/consume per key, one-shot each.
    intake.arm_echo(turn_key="tk-slot-a", text="same text")
    intake.arm_echo(turn_key="tk-slot-b", text="same text")
    assert intake.consume_echo(turn_key="tk-slot-a", text="same text") is True
    assert intake.consume_echo(turn_key="tk-slot-b", text="same text") is True
    assert intake.consume_echo(turn_key="tk-slot-a", text="same text") is False
