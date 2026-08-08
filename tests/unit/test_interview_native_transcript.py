"""The native path's transcript writer.

Nothing on this path wrote `interview_session_messages`: the routed path recorded
turns as a side effect of `take_session_step`, which this agent never calls. A
finished interview therefore stored only the REST onboarding turns and the closing
ceremony — and the evaluation and gap report, which read that table, had no answers
to judge.

The first attempt then failed on EVERY row: `state.current_question_id` is an
`interview_questions` id, but `session_question_id` is a foreign key to
`interview_session_questions`. Writing the bank id through violated the constraint
and the insert rolled back, so the transcript stayed empty and the only symptom was
a log line.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.realtime import native_transcript

pytestmark = pytest.mark.asyncio


class _FakeDb:
    """Records what was added; answers the resolver's two lookups."""

    def __init__(self, existing_link: Any = None) -> None:
        self.added: list[Any] = []
        self.existing_link = existing_link
        self.committed = False
        self._scalars: list[Any] = []

    async def scalar(self, statement: object) -> Any:
        # First lookup: the existing link. Second: the max sequence number.
        self._scalars.append(statement)
        return self.existing_link if len(self._scalars) == 1 else 0

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        self.committed = True


def _use(monkeypatch: pytest.MonkeyPatch, db: _FakeDb) -> None:
    class _Ctx:
        async def __aenter__(self) -> _FakeDb:
            return db

        async def __aexit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(native_transcript, "get_sessionmaker", lambda: lambda: _Ctx())


async def test_links_the_session_question_not_the_bank_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeDb()
    _use(monkeypatch, db)
    session_id, bank_question_id = uuid4(), uuid4()

    await native_transcript.record_turn(
        session_id,
        role="user",
        text="An index speeds up lookups",
        session_question_id=bank_question_id,
        kind="answer",
    )

    created = [row for row in db.added if hasattr(row, "sequence_no")]
    message = next(row for row in db.added if hasattr(row, "content_text"))
    assert created, "a session-question row must be created when none exists yet"
    assert created[0].interview_question_id == bank_question_id
    assert message.session_question_id != bank_question_id, (
        "writing the bank id here violates the foreign key and loses the turn"
    )
    assert message.role == "user"
    assert db.committed is True


async def test_reuses_an_existing_link(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = uuid4()
    db = _FakeDb(existing_link=existing)
    _use(monkeypatch, db)

    await native_transcript.record_turn(
        uuid4(),
        role="assistant",
        text="A follow-up",
        session_question_id=uuid4(),
        kind="question",
    )

    message = next(row for row in db.added if hasattr(row, "content_text"))
    assert message.session_question_id == existing
    assert message.role == "ai", "the SDK's 'assistant' maps onto the 'ai' role"
    assert not [row for row in db.added if hasattr(row, "sequence_no")]


async def test_an_unlinked_turn_still_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    _use(monkeypatch, db)

    await native_transcript.record_turn(
        uuid4(), role="user", text="hello", session_question_id=None, kind="answer"
    )

    message = next(row for row in db.added if hasattr(row, "content_text"))
    assert message.session_question_id is None
    assert db.committed is True


async def test_blank_and_unknown_roles_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    _use(monkeypatch, db)

    await native_transcript.record_turn(
        uuid4(), role="user", text="   ", session_question_id=None, kind="answer"
    )
    await native_transcript.record_turn(
        uuid4(), role="tool", text="{}", session_question_id=None, kind="answer"
    )

    assert db.added == []
