from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.quizzes.queries import overrides
from abridgeai.features.quizzes.services import audit


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def scalars(self) -> _ScalarRows:
        return self

    def all(self) -> list[object]:
        return self._rows


@pytest.mark.asyncio
async def test_audit_record_validates_registry_and_persists(monkeypatch):
    now = datetime(2026, 9, 16, tzinfo=UTC)
    db = SimpleNamespace(add=lambda row: added.append(row))
    added: list[object] = []
    flush = AsyncMock()
    monkeypatch.setattr(audit, "flush_or_conflict", flush)
    monkeypatch.setattr(audit, "utcnow", lambda: now)
    quiz_id = uuid4()
    actor_id = uuid4()

    row = await audit.record_event(
        db,
        event_name="quiz_published",
        quiz_id=quiz_id,
        actor_user_id=actor_id,
        payload={"source": "test"},
    )

    assert added == [row]
    assert row.quiz_id == quiz_id
    assert row.actor_user_id == actor_id
    assert row.payload_json == {"source": "test"}
    assert row.occurred_at == now
    flush.assert_awaited_once_with(db)

    with pytest.raises(ValueError, match="Unknown quiz audit event"):
        await audit.record_event(db, event_name="typo", quiz_id=quiz_id)


@pytest.mark.asyncio
async def test_audit_list_returns_most_recent_query_result():
    rows = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
    db = SimpleNamespace(execute=AsyncMock(return_value=_ScalarRows(rows)))

    assert await audit.list_events_for_quiz(db, uuid4(), limit=2) == rows


@pytest.mark.asyncio
async def test_override_query_crud_and_group_placeholder():
    quiz_id = uuid4()
    override_id = uuid4()
    existing = SimpleNamespace(scope="user", max_attempts=1, cooldown_hours=None)
    listed = [existing]
    db = SimpleNamespace(
        execute=AsyncMock(return_value=_ScalarRows(listed)),
        get=AsyncMock(side_effect=[existing, existing, None, existing]),
        add=lambda row: added.append(row),
        delete=AsyncMock(),
    )
    added: list[object] = []

    assert await overrides.list_overrides(db, quiz_id) == listed
    assert await overrides.get_override(db, override_id) is existing

    created = await overrides.create_override(
        db,
        quiz_id,
        {"scope": "user", "max_attempts": 4, "ignored": "not persisted"},
    )
    assert added == [created]
    assert created.quiz_id == quiz_id
    assert created.max_attempts == 4
    assert not hasattr(created, "ignored")

    updated = await overrides.update_override(
        db, override_id, {"max_attempts": 5, "cooldown_hours": 2, "ignored": 9}
    )
    assert updated is existing
    assert existing.max_attempts == 5
    assert existing.cooldown_hours == 2

    assert await overrides.update_override(db, uuid4(), {"max_attempts": 2}) is None
    assert await overrides.delete_override(db, override_id) is True
    db.delete.assert_awaited_once_with(existing)
    assert await overrides.get_group_ids_for_user(db, uuid4()) == set()


@pytest.mark.asyncio
async def test_delete_override_returns_false_when_missing():
    db = SimpleNamespace(get=AsyncMock(return_value=None), delete=AsyncMock())
    assert await overrides.delete_override(db, uuid4()) is False
    db.delete.assert_not_awaited()
