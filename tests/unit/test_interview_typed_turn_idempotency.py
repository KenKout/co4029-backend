"""A resent typed turn is the SAME turn, and a finish waits for one in flight.

Two failures in the window between "the agent has your text" and "the answer is
durable". On the native path that window is a whole LLM grading call wide.

Resent turns
------------
``turn_key`` is the client's idempotency key, and the protocol explicitly invites
a retry with the same key (``ControlStatus.FAILED``: "the client keeps the draft
and may retry with the SAME turn_key"). The backend parsed it, used it to
correlate the ack, and then dropped it: it was never passed into the fold and
never persisted. So a candidate whose ack was lost to a reconnect and who resent
had the turn graded twice — coverage points applied twice, a second follow-up
charged against the question's budget, and, because the server can advance
between the two copies, the resend folded against the question AFTER the one it
answered.

Finish barrier
--------------
``TranscriptWriteBarrier`` waits for transcript writes it was handed. A turn still
inside its grading call has handed it nothing, so a finish could submit a
transcript that the candidate's last answer never reached — and the evaluator
grades that transcript.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import text_protocol as tp
from abridgeai.features.interviews.realtime.native_text_input import make_text_input_cb
from abridgeai.features.interviews.realtime.native_turn_intake import TurnIntake, TurnLedger

# Applied per-test rather than module-wide: the ledger's own tests are synchronous
# (it has no awaits), and marking those asyncio only produces a pytest warning.
_asyncio = pytest.mark.asyncio


class _Publisher:
    """Records control egress instead of touching a room."""

    def __init__(self) -> None:
        self.acks: list[str | None] = []
        self.rejections: list[str] = []

    async def ack(self, *, turn_key: str | None, turn_action: str) -> None:
        del turn_action
        self.acks.append(turn_key)

    async def reject(
        self, *, turn_key: str | None, turn_action: str, rejection: Any
    ) -> None:
        del turn_key, turn_action
        self.rejections.append(str(rejection))

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
            "_U", (), {"interview_session_id": uuid4(), "pending_assistant_kind": None}
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


# ───────────────────────── the ledger, in isolation ─────────────────────────


def test_a_key_can_only_be_claimed_once() -> None:
    ledger = TurnLedger()
    assert ledger.claim("tk-abc12345") is True
    assert ledger.claim("tk-abc12345") is False


def test_distinct_keys_are_independent() -> None:
    ledger = TurnLedger()
    assert ledger.claim("tk-first0001") is True
    assert ledger.claim("tk-second002") is True


def test_an_absent_key_is_never_a_duplicate() -> None:
    """The protocol allows omitting the key; that client opted out of idempotency."""
    ledger = TurnLedger()
    assert ledger.claim(None) is True
    assert ledger.claim(None) is True


def test_a_seeded_key_is_already_claimed() -> None:
    """Seeded from `last_turn_idempotency_key`, so a restart remembers.

    Without this the ledger starts empty after a crash or redeploy, and a client
    that reconnects to the replacement process and retries is graded again.
    """
    ledger = TurnLedger()
    ledger.seed("tk-from-a-previous-run")
    assert ledger.claim("tk-from-a-previous-run") is False


def test_the_ledger_evicts_oldest_first_under_its_cap() -> None:
    """Bounded so a looping client cannot grow it, oldest-out so a retry still hits."""
    ledger = TurnLedger(capacity=2)
    ledger.claim("tk-oldest0001")
    ledger.claim("tk-middle0002")
    ledger.claim("tk-newest0003")

    assert ledger.claim("tk-oldest0001") is True, "expected the oldest key to be evicted"
    assert ledger.claim("tk-newest0003") is False


# ─────────────────────── the typed door's dedupe ───────────────────────


@_asyncio
async def test_a_resent_turn_is_not_graded_twice() -> None:
    agent = _Agent()
    session = _Session(agent)
    publisher = _Publisher()
    on_text = make_text_input_cb(publisher, intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(session, _Event("my answer", turn_key="tk-resend0001"))
    await on_text(session, _Event("my answer", turn_key="tk-resend0001"))

    assert len(agent.folded) == 1, (
        "the same turn was graded twice: coverage points doubled and a second "
        "follow-up was charged"
    )


@_asyncio
async def test_a_resent_turn_is_re_acked_so_the_client_can_settle() -> None:
    """The client retried BECAUSE it lost the ack; silence would leave it stuck."""
    session = _Session(_Agent())
    publisher = _Publisher()
    on_text = make_text_input_cb(publisher, intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(session, _Event("my answer", turn_key="tk-resend0002"))
    await on_text(session, _Event("my answer", turn_key="tk-resend0002"))

    assert publisher.acks == ["tk-resend0002", "tk-resend0002"]


@_asyncio
async def test_a_resent_turn_does_not_produce_a_second_reply() -> None:
    """A duplicate must not make the interviewer answer the same text again."""
    session = _Session(_Agent())
    on_text = make_text_input_cb(_Publisher(), intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(session, _Event("my answer", turn_key="tk-resend0003"))
    await on_text(session, _Event("my answer", turn_key="tk-resend0003"))

    assert len(session.replies) == 1


@_asyncio
async def test_a_genuinely_new_turn_is_still_graded() -> None:
    agent = _Agent()
    session = _Session(agent)
    on_text = make_text_input_cb(_Publisher(), intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(session, _Event("first", turn_key="tk-new000001"))
    await on_text(session, _Event("second", turn_key="tk-new000002"))

    assert [text for text, _ in agent.folded] == ["first", "second"]


@_asyncio
async def test_the_client_key_reaches_the_fold_for_persistence() -> None:
    """It has to be written as `last_turn_idempotency_key` or a restart forgets."""
    agent = _Agent()
    on_text = make_text_input_cb(_Publisher(), intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(_Session(agent), _Event("answer", turn_key="tk-persist001"))

    assert agent.folded == [("answer", "tk-persist001")]


@_asyncio
async def test_a_turn_without_a_key_is_still_processed() -> None:
    agent = _Agent()
    on_text = make_text_input_cb(_Publisher(), intake=TurnIntake())  # type: ignore[arg-type]

    await on_text(_Session(agent), _Event("answer", turn_key=None))

    assert agent.folded == [("answer", None)]


@_asyncio
async def test_a_restart_still_refuses_a_key_it_persisted() -> None:
    """The seeded ledger is what covers the crash-and-retry path."""
    agent = _Agent()
    intake = TurnIntake()
    intake.seed("tk-before-crash01")
    on_text = make_text_input_cb(_Publisher(), intake=intake)  # type: ignore[arg-type]

    await on_text(_Session(agent), _Event("answer", turn_key="tk-before-crash01"))

    assert agent.folded == [], "a turn the previous process graded was graded again"


# ───────────────────── the finish barrier ─────────────────────


@_asyncio
async def test_a_finish_waits_for_a_turn_that_is_still_grading() -> None:
    """THE BUG: submit could beat the answer into the transcript.

    The transcript barrier cannot cover this — the write does not exist yet while
    the fold is still inside its grading call.
    """
    agent = _Agent()
    agent.gate = asyncio.Event()
    intake = TurnIntake()
    on_text = make_text_input_cb(_Publisher(), intake=intake)  # type: ignore[arg-type]

    turn = asyncio.create_task(
        on_text(_Session(agent), _Event("answer", turn_key="tk-inflight01"))
    )
    await asyncio.sleep(0)  # let the turn reach the gated fold
    assert intake.in_flight == 1

    drain = asyncio.create_task(intake.drain(timeout_seconds=5.0))
    await asyncio.sleep(0.05)
    assert not drain.done(), "the finish did not wait for the in-flight turn"

    agent.gate.set()
    await turn
    assert await drain is True
    assert intake.in_flight == 0


@_asyncio
async def test_a_finish_on_an_idle_session_does_not_wait() -> None:
    intake = TurnIntake()
    assert await intake.drain(timeout_seconds=0.01) is True


@_asyncio
async def test_the_drain_is_bounded_so_a_wedged_turn_cannot_hang_teardown() -> None:
    """A submitted session matters more than a complete transcript."""
    agent = _Agent()
    agent.gate = asyncio.Event()  # never set
    intake = TurnIntake()
    on_text = make_text_input_cb(_Publisher(), intake=intake)  # type: ignore[arg-type]

    turn = asyncio.create_task(
        on_text(_Session(agent), _Event("answer", turn_key="tk-wedged0001"))
    )
    await asyncio.sleep(0)

    assert await intake.drain(timeout_seconds=0.05) is False

    agent.gate.set()
    await turn


@_asyncio
async def test_a_failing_fold_still_releases_the_in_flight_slot() -> None:
    """Otherwise one bad turn makes every later finish wait out the full timeout."""

    class _Boom(_Agent):
        async def fold_turn(self, *, answer_text: str, turn_key: str | None = None) -> None:
            del answer_text, turn_key
            raise RuntimeError("gateway down")

    intake = TurnIntake()
    on_text = make_text_input_cb(_Publisher(), intake=intake)  # type: ignore[arg-type]

    await on_text(_Session(_Boom()), _Event("answer", turn_key="tk-failing001"))

    assert intake.in_flight == 0
    assert await intake.drain(timeout_seconds=0.01) is True
