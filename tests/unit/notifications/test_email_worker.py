"""The email-channel worker, and the state it must not leave behind.

The task itself is thin -- bind an actor, call one service, commit. What
the tests below are actually about is the ``finally`` block, because an ARQ
worker process handles jobs back to back in the same interpreter. The actor
lives in a ``ContextVar`` that the audit layer reads when it stamps
``created_by`` / ``updated_by``. A job that exits without clearing it hands
its actor to whatever job runs next in that worker, and the resulting audit
rows name a user who had nothing to do with them -- correct-looking,
attributable, and wrong.

That failure is invisible in the job that causes it and shows up in an
unrelated one, so it is worth a test on the path that makes it likeliest:
the one where the task raises.

The service is mocked here; ``deliver_email_for_notification`` has its own
coverage in the notifications suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from abridgeai.core.audit import current_actor_var
from abridgeai.features.notifications.workers import email as email_worker
from abridgeai.features.notifications.workers.email import send_email_notification_task


class _FakeSession:
    def __init__(self) -> None:
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        self.rolled_back += 1

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    db = _FakeSession()
    monkeypatch.setattr(email_worker, "get_sessionmaker", lambda: lambda: db)
    return db


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Records the actor/log-context calls the task is required to make."""
    calls: dict[str, Any] = {"bound": [], "cleared": 0, "actor_during_task": "unset"}
    monkeypatch.setattr(
        email_worker, "bind_request_context", lambda **fields: calls["bound"].append(fields)
    )
    monkeypatch.setattr(
        email_worker,
        "clear_request_context",
        lambda: calls.__setitem__("cleared", calls["cleared"] + 1),
    )
    return calls


@pytest.fixture(autouse=True)
def _clean_actor() -> Any:
    """Guard the suite itself against the leak these tests are about."""
    token = current_actor_var.set(None)
    yield
    current_actor_var.reset(token)


async def test_the_notification_is_delivered_and_committed(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    deliver = AsyncMock(return_value=True)
    monkeypatch.setattr(email_worker.email_service, "deliver_email_for_notification", deliver)
    actor, notification = uuid4(), uuid4()

    await send_email_notification_task({}, actor, notification)

    assert deliver.await_args.kwargs["notification_id"] == notification
    assert session.committed == 1
    assert session.rolled_back == 0


async def test_the_actor_is_bound_before_the_service_runs(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit columns are stamped from the ContextVar during the write, so
    binding after the service call would be binding after the rows exist.

    The recipient doubles as the actor: they own the notification row.
    """
    actor = uuid4()
    seen: dict[str, Any] = {}

    async def _deliver(_db: object, *, notification_id: object) -> bool:
        seen["actor"] = current_actor_var.get()
        return True

    monkeypatch.setattr(email_worker.email_service, "deliver_email_for_notification", _deliver)

    await send_email_notification_task({}, actor, uuid4())

    assert seen["actor"] == actor


async def test_the_log_context_carries_both_ids(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker interleaves jobs, so a log line without its ids cannot be
    attributed to the notification that produced it."""
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(return_value=True),
    )
    actor, notification = uuid4(), uuid4()

    await send_email_notification_task({}, actor, notification)

    assert context["bound"] == [
        {"notification_id": str(notification), "actor_id": str(actor)}
    ]


async def test_a_failure_rolls_back_logs_and_re_raises(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-raising is what hands the job back to ARQ's retry policy.

    Swallowing it would mark the job successful and the notification would
    never be delivered, with the failure visible nowhere.
    """
    logger = Mock()
    monkeypatch.setattr(email_worker, "_logger", logger)
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(side_effect=RuntimeError("smtp exploded")),
    )

    with pytest.raises(RuntimeError, match="smtp exploded"):
        await send_email_notification_task({}, uuid4(), uuid4())

    assert session.rolled_back == 1
    assert session.committed == 0
    assert logger.exception.call_args.args[0] == "email_notification_task_failed"


async def test_the_actor_is_cleared_even_when_the_task_fails(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leak this module's ``finally`` exists to prevent.

    A worker runs jobs back to back in one process. An actor left set would
    be inherited by the next job on that worker, whose audit rows would then
    name this notification's recipient.
    """
    monkeypatch.setattr(email_worker, "_logger", Mock())
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        await send_email_notification_task({}, uuid4(), uuid4())

    assert current_actor_var.get() is None
    assert context["cleared"] == 1


async def test_the_actor_is_cleared_after_a_successful_task(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(return_value=True),
    )

    await send_email_notification_task({}, uuid4(), uuid4())

    assert current_actor_var.get() is None
    assert context["cleared"] == 1


async def test_a_shutdown_signal_is_not_recorded_as_a_task_failure(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``KeyboardInterrupt`` and ``SystemExit`` are the worker being stopped,
    not the job going wrong. Logging them as failures would fill the error
    budget every time the process is restarted, and they propagate either
    way -- but the context still has to be cleaned up on the way out.
    """
    logger = Mock()
    monkeypatch.setattr(email_worker, "_logger", logger)
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(side_effect=KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        await send_email_notification_task({}, uuid4(), uuid4())

    logger.exception.assert_not_called()
    assert current_actor_var.get() is None
    assert context["cleared"] == 1


async def test_a_delivery_that_declines_is_still_committed(
    session: _FakeSession, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service returns ``False`` for a missing or already-terminal row.

    That is a decision, not an error: the job is done and must not be
    retried, so the task commits and returns normally.
    """
    monkeypatch.setattr(
        email_worker.email_service,
        "deliver_email_for_notification",
        AsyncMock(return_value=False),
    )

    await send_email_notification_task({}, uuid4(), uuid4())

    assert session.committed == 1
    assert session.rolled_back == 0
