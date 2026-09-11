"""Typed-turn receipt LINKAGE: bank question -> session-question, resolved once.

Gap this closes (plan §1): ``persist_receipt`` accepted ``session_question_id``
from the caller and the typed door always passed ``None`` — so every durable
typed receipt was stored UNLINKED. The post-session evaluator filters candidate
answers to rows with a ``session_question_id`` (a hard signal the answer belongs
to a real asked question), so a typed answer could be ACKED, applied, and still
invisible to grading. The bank question id was already in the receipt metadata;
what was missing was the bank→session-question resolution the transcript writer
already performs.

Contract (shared with ``native_transcript``):
* input  = the BANK question id (``InterviewQuestion.id``) captured at receipt
  time — before any fold/advance can move the interview on;
* output = the SESSION question id (``InterviewSessionQuestion.id``), created
  on demand inside the caller's transaction;
* concurrent creation of the same link resolves by constraint/reload — the
  losing caller reuses the winner's row and the answer is never lost.

Unit tests here use the in-memory store; the SQL linkage is pinned in
``tests/integration/test_interview_typed_turn_receipt.py``.
"""

from __future__ import annotations

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
        self.failures: list[tuple[str | None, str]] = []
        self.confirmations: list[str] = []

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
        del turn_action
        self.failures.append((turn_key, error_class))

    async def agent_action(self, *, kind: str, text: str | None = None) -> None:
        del kind, text

    async def acknowledge(self, *, turn_key: str, state: Any = None) -> None:
        del state
        self.confirmations.append(turn_key)


class _Agent:
    def __init__(self) -> None:
        self.folded: list[tuple[str, str | None]] = []

    async def fold_turn(self, **kwargs: Any) -> None:
        self.folded.append((kwargs.get("answer_text"), kwargs.get("turn_key")))


class _Session:
    """Mimics the production session shape: userdata.state names the bank question."""

    def __init__(self, agent: _Agent, bank_question_id: Any) -> None:
        self.current_agent = agent
        self.userdata = type(
            "_U",
            (),
            {
                "interview_session_id": uuid4(),
                "pending_assistant_kind": None,
                "echo_turn_key": None,
                # The live state object with the CURRENT bank question, exactly
                # what ``_current_bank_question_id`` reads in production.
                "state": type("_S", (), {"current_question_id": bank_question_id})(),
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


class _LinkedStore(ntt.InMemoryTypedTurnStore):
    """The in-memory store with production receipt shapes: linkage + metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.resolved: list[Any] = []

    async def persist(
        self,
        *,
        session_id: Any,
        session_question_id: Any,
        bank_question_id: Any,
        text: str,
        turn_key: str,
    ) -> tuple[dict[str, Any], bool]:
        del session_id
        # Production ``persist_receipt`` resolves the linkage itself; the store
        # only records what arrived so the test can assert the RESOLVED id (not
        # None) landed in the row.
        self.resolved.append(session_question_id)
        row, created = await super().persist(
            session_id=None,
            session_question_id=session_question_id,
            bank_question_id=bank_question_id,
            text=text,
            turn_key=turn_key,
        )
        row["session_question_id"] = session_question_id
        row["bank_question_id"] = (
            str(bank_question_id) if bank_question_id is not None else None
        )
        return row, created


# ── the typed door must resolve and persist the linkage ──────────────────────


@_asyncio
async def test_receipt_persists_with_a_linkage_input_not_none() -> None:
    """The receipt row records a linkage INPUT while a bank question is live.

    A None linkage was accepted before this change, which put every typed
    receipt outside the evaluator's candidate filter. The bank question is
    captured from the live state at receipt time and reaches the store — never
    ``None`` while a bank question is current. (The store records the input;
    the SQL resolver turns it into the session-question row, pinned by the
    integration suite.)
    """
    store = _LinkedStore()
    bank_q = uuid4()
    session = _Session(_Agent(), bank_question_id=bank_q)
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(session, _Event("Recursion terminates at the base case.", turn_key="tk-link0001"))

    assert store.resolved, "the store was never asked to persist the receipt"
    resolved = store.resolved[0]
    assert resolved is not None, (
        "the receipt was persisted UNLINKED — the evaluator filters on "
        "session_question_id, so this answer would never be graded"
    )
    row = store.rows["tk-link0001"]
    assert row["session_question_id"] == resolved
    assert row["bank_question_id"] == str(bank_q)
    assert row["turn_state"] == "applied"


@_asyncio
async def test_the_linkage_is_captured_before_the_fold_moves_the_interview() -> None:
    """The bank question must be read BEFORE ``fold_turn`` runs.

    The fold may advance to the next question; a linkage read afterwards would
    attach this answer to the NEXT question — the mis-pairing the transcript
    handler already guards against. The store receives the id captured at
    receipt time even when the fold flips the live question mid-flight.
    """
    store = _LinkedStore()
    q1 = uuid4()
    q2 = uuid4()

    class _AdvancingAgent(_Agent):
        async def fold_turn(self, *, answer_text: str, turn_key: str | None = None) -> None:
            await super().fold_turn(answer_text=answer_text, turn_key=turn_key)
            # Simulate the server advance that follows a sufficient answer.
            session.userdata.state.current_question_id = q2

    session = _Session(_AdvancingAgent(), bank_question_id=q1)
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(session, _Event("Answer to Q1.", turn_key="tk-link0002"))

    row = store.rows["tk-link0002"]
    assert row["bank_question_id"] == str(q1), (
        "the receipt captured the question the fold ADVANCED to, not the one "
        "the answer actually answered"
    )


@_asyncio
async def test_an_unlinked_turn_degrades_to_none_without_lying() -> None:
    """No live bank question (edge: onboarding/reset) → linkage stays None.

    A None linkage is honest for a turn that answers no question; the
    regression is the TYPED ANSWER path producing None while a question is
    live, which the first test pins. Here we pin that the resolver's absence
    does not crash the door and the receipt still settles.
    """
    store = _LinkedStore()
    session = _Session(_Agent(), bank_question_id=None)
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(session, _Event("Free-form remark.", turn_key="tk-link0003"))

    row = store.rows["tk-link0003"]
    assert row["session_question_id"] is None
    assert row["turn_state"] == "applied"


# ── shared resolver semantics (contract pinned here, SQL in integration) ────


@_asyncio
async def test_the_door_passes_the_captured_bank_id_as_the_linkage_input() -> None:
    """The door's linkage input equals the bank id captured BEFORE the fold.

    The unit-level contract: the typed door supplies ``session_question_id``
    (the linkage slot) from ``_current_bank_question_id`` — the same capture it
    sends as ``bank_question_id`` — so a crash between capture and resolution
    cannot orphan the answer. The store only records what arrived; the SQL
    resolution (bank→session-question row) is pinned by
    ``tests/integration/test_interview_typed_turn_receipt.py`` against real
    Postgres, where the resolver's create-on-demand actually runs.
    """
    store = _LinkedStore()
    bank_q = uuid4()
    session = _Session(_Agent(), bank_question_id=bank_q)
    on_text = make_text_input_cb(
        _Publisher(), intake=TurnIntake(), receipts=store  # type: ignore[arg-type]
    )

    await on_text(session, _Event("answer", turn_key="tk-link0004"))

    assert len(store.resolved) == 1
    assert store.resolved[0] == bank_q
