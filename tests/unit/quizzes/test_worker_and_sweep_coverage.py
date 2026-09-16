from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from abridgeai.features.quizzes.services import sweep
from abridgeai.features.quizzes.workers import generation, timing
from abridgeai.workers import audit_retention


class _SessionContext:
    def __init__(self, db: object) -> None:
        self.db = db

    async def __aenter__(self) -> object:
        return self.db

    async def __aexit__(self, *args: object) -> None:
        return None


def _sessionmaker(db: object):
    return lambda: _SessionContext(db)


@pytest.mark.asyncio
async def test_generation_worker_propagates_actor_redis_and_cleans_context(monkeypatch):
    db = object()
    actor_id = uuid4()
    run_id = uuid4()
    run = AsyncMock()
    actor = Mock()
    bind = Mock()
    clear = Mock()
    current_actor = Mock()
    monkeypatch.setattr(generation, "get_sessionmaker", lambda: _sessionmaker(db))
    monkeypatch.setattr(generation.generation_service, "run_quiz_generation", run)
    monkeypatch.setattr(generation, "set_worker_actor", actor)
    monkeypatch.setattr(generation, "bind_request_context", bind)
    monkeypatch.setattr(generation, "clear_request_context", clear)
    monkeypatch.setattr(generation, "current_actor_var", current_actor)

    redis = object()
    await generation.run_quiz_generation_task(
        {"redis": redis}, actor_id=actor_id, generation_run_id=run_id
    )

    actor.assert_called_once_with(actor_id)
    bind.assert_called_once_with(generation_run_id=str(run_id), actor_id=str(actor_id))
    run.assert_awaited_once_with(db, run_id, arq_pool=redis)
    current_actor.set.assert_called_once_with(None)
    clear.assert_called_once_with()


@pytest.mark.asyncio
async def test_generation_worker_logs_regular_failure_and_still_cleans(monkeypatch):
    db = object()
    failure = RuntimeError("pipeline failed")
    run = AsyncMock(side_effect=failure)
    logger = Mock()
    clear = Mock()
    current_actor = Mock()
    monkeypatch.setattr(generation, "get_sessionmaker", lambda: _sessionmaker(db))
    monkeypatch.setattr(generation.generation_service, "run_quiz_generation", run)
    monkeypatch.setattr(generation, "set_worker_actor", Mock())
    monkeypatch.setattr(generation, "bind_request_context", Mock())
    monkeypatch.setattr(generation, "clear_request_context", clear)
    monkeypatch.setattr(generation, "current_actor_var", current_actor)
    monkeypatch.setattr(generation, "_logger", logger)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        await generation.run_quiz_generation_task({}, uuid4(), uuid4())

    logger.exception.assert_called_once()
    current_actor.set.assert_called_once_with(None)
    clear.assert_called_once_with()


@pytest.mark.asyncio
async def test_timing_worker_commits_logs_nonzero_result_and_cleans(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock())
    result = {"submitted": 2, "expired": 1, "skipped": 3}
    worker = AsyncMock(return_value=result)
    logger = Mock()
    clear = Mock()
    monkeypatch.setattr(timing, "get_sessionmaker", lambda: _sessionmaker(db))
    monkeypatch.setattr(timing, "sweep_overdue", worker)
    monkeypatch.setattr(timing, "bind_request_context", Mock())
    monkeypatch.setattr(timing, "clear_request_context", clear)
    monkeypatch.setattr(timing, "_logger", logger)

    assert await timing.sweep_overdue_attempts_task({"ignored": True}) == result

    worker.assert_awaited_once_with(db)
    db.commit.assert_awaited_once_with()
    logger.info.assert_called_once_with("swept_overdue_quiz_attempts", **result)
    clear.assert_called_once_with()


@pytest.mark.asyncio
async def test_timing_worker_does_not_log_empty_sweep(monkeypatch):
    db = SimpleNamespace(commit=AsyncMock())
    logger = Mock()
    monkeypatch.setattr(timing, "get_sessionmaker", lambda: _sessionmaker(db))
    monkeypatch.setattr(
        timing,
        "sweep_overdue",
        AsyncMock(return_value={"submitted": 0, "expired": 0, "skipped": 4}),
    )
    monkeypatch.setattr(timing, "bind_request_context", Mock())
    monkeypatch.setattr(timing, "clear_request_context", Mock())
    monkeypatch.setattr(timing, "_logger", logger)

    await timing.sweep_overdue_attempts_task({})

    logger.info.assert_not_called()


class _Rows:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[object, object]]:
        return self._rows


@pytest.mark.asyncio
async def test_list_overdue_candidates_projects_result_rows():
    pair_a = (SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4()))
    pair_b = (SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4()))
    db = SimpleNamespace(execute=AsyncMock(return_value=_Rows([pair_a, pair_b])))

    assert await sweep.list_overdue_candidates(db, limit=7) == [pair_a, pair_b]
    db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_counts_skipped_expired_and_submitted(monkeypatch):
    now = datetime(2026, 9, 16, 8, 0, tzinfo=UTC)
    skipped_attempt = SimpleNamespace(started_at=now)
    expired_attempt = SimpleNamespace(started_at=now)
    submitted_attempt = SimpleNamespace(started_at=now)
    skipped_quiz = SimpleNamespace(
        overdue_handling="autosubmit", grace_period_seconds=None, due_at=None
    )
    expired_quiz = SimpleNamespace(
        overdue_handling="autoabandon", grace_period_seconds=10, due_at=None
    )
    submitted_quiz = SimpleNamespace(
        overdue_handling="graceperiod", grace_period_seconds=30, due_at=now
    )
    monkeypatch.setattr(
        sweep,
        "list_overdue_candidates",
        AsyncMock(
            return_value=[
                (skipped_attempt, skipped_quiz),
                (expired_attempt, expired_quiz),
                (submitted_attempt, submitted_quiz),
            ]
        ),
    )
    monkeypatch.setattr(sweep._timing, "resolve_effective_timing", Mock(return_value=object()))
    monkeypatch.setattr(sweep._timing, "is_overdue", Mock(side_effect=[False, True, True]))
    expire = AsyncMock()
    finalize = AsyncMock()
    monkeypatch.setattr(sweep, "_expire_attempt", expire)
    monkeypatch.setattr(sweep, "_finalize_attempt", finalize)
    db = object()

    assert await sweep.sweep_overdue(db, now=now) == {
        "submitted": 1,
        "expired": 1,
        "skipped": 1,
    }
    expire.assert_awaited_once_with(db, expired_attempt, now=now)
    finalize.assert_awaited_once_with(db, submitted_attempt, submitted_quiz, now=now)
    assert sweep._timing.is_overdue.call_args_list[2].kwargs["grace_period_seconds"] == 30


@pytest.mark.asyncio
async def test_audit_retention_worker_sums_and_logs(monkeypatch):
    factory = object()
    prune = AsyncMock(return_value={"http_audit_log": 4, "auth_events": 2})
    logger = Mock()
    monkeypatch.setattr(audit_retention, "get_sessionmaker", lambda: factory)
    monkeypatch.setattr(audit_retention, "prune_audit_logs", prune)
    monkeypatch.setattr(audit_retention, "logger", logger)

    assert await audit_retention.prune_audit_logs_task({"ignored": True}) == 6
    prune.assert_awaited_once_with(factory)
    logger.info.assert_called_once_with(
        "audit.retention_sweep",
        total_deleted=6,
        http_audit_log=4,
        auth_events=2,
    )
