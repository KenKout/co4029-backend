"""The nightly sweep that repairs drifted course-completion rows.

Every synchronous writer of ``course_enrollments.status`` -- lesson
progress, quiz grading, interview evaluation -- deliberately swallows its
own exceptions, because a student's mark-complete must not fail over a
bookkeeping side-effect. The cost of that safety is that a lost write is
silent and permanent, and the column is not cosmetic: career-path stage
unlock reads it as ``satisfied``, so drift either hands out a stage nobody
earned or withholds one that was.

This task is the backstop. Its own logic is four lines, and the part worth
pinning is the line that decides the log level: a repair is reported at
**warning**, not info, because in steady state there should be nothing to
repair. A non-zero count is not "the sweep did its job", it is evidence
that one of the synchronous call sites is dropping writes. Flattening both
branches to ``info`` would leave that signal in a log nobody greps.

The service and the session are mocked; ``resync_stale_course_completions``
has its own coverage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from abridgeai.features.enrollments.workers import completion_drift
from abridgeai.features.enrollments.workers.completion_drift import (
    resync_course_completions_task,
)


class _FakeSession:
    def __init__(self) -> None:
        self.committed = 0

    async def commit(self) -> None:
        self.committed += 1

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    db = _FakeSession()
    monkeypatch.setattr(completion_drift, "get_sessionmaker", lambda: lambda: db)
    return db


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> Mock:
    recorder = Mock()
    monkeypatch.setattr(completion_drift, "logger", recorder)
    return recorder


def _resync(monkeypatch: pytest.MonkeyPatch, scanned: int, fixed: int) -> AsyncMock:
    stub = AsyncMock(return_value=(scanned, fixed))
    monkeypatch.setattr(
        completion_drift.completion_service, "resync_stale_course_completions", stub
    )
    return stub


async def test_a_clean_sweep_is_reported_at_info(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """Nothing to repair is the expected nightly outcome.

    It still gets a line, so the absence of one means the cron did not run
    rather than that it found nothing.
    """
    _resync(monkeypatch, scanned=120, fixed=0)

    fixed = await resync_course_completions_task({})

    assert fixed == 0
    logger.info.assert_called_once()
    logger.warning.assert_not_called()
    assert logger.info.call_args.args[0] == "enrollments.completion_drift_sweep"
    assert logger.info.call_args.kwargs == {"scanned": 120, "fixed": 0}


async def test_a_repair_is_escalated_to_warning(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """The signal this task exists to raise.

    A row needing repair means a synchronous writer failed earlier and said
    nothing. The sweep fixing it is the symptom being treated, not the
    problem being solved, so it is logged at a level someone alerts on.
    """
    _resync(monkeypatch, scanned=120, fixed=3)

    fixed = await resync_course_completions_task({})

    assert fixed == 3
    logger.warning.assert_called_once()
    logger.info.assert_not_called()
    assert logger.warning.call_args.kwargs == {"scanned": 120, "fixed": 3}


async def test_the_counts_are_both_reported(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """``fixed`` alone cannot be read: three repairs out of five enrolments
    is a broken writer, three out of fifty thousand is a stray row."""
    _resync(monkeypatch, scanned=48_000, fixed=3)

    await resync_course_completions_task({})

    assert set(logger.warning.call_args.kwargs) == {"scanned", "fixed"}


async def test_the_task_owns_the_commit(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """The service recomputes and flushes per row but leaves the
    transaction to its caller, so a sweep that never commits would repair
    nothing and still report success."""
    _resync(monkeypatch, scanned=10, fixed=2)

    await resync_course_completions_task({})

    assert session.committed == 1


async def test_the_repaired_count_is_returned_to_arq(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """ARQ records a task's return value, which is what makes the trend
    visible across nights rather than only in the log line."""
    _resync(monkeypatch, scanned=5, fixed=5)
    assert await resync_course_completions_task({}) == 5


async def test_the_context_argument_is_not_required_to_carry_anything(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """A cron task is invoked by the worker with its own ctx; this one opens
    its own session rather than reading anything out of it."""
    _resync(monkeypatch, scanned=1, fixed=0)
    ctx: dict[str, Any] = {"redis": SimpleNamespace(), "job_id": "abc"}

    assert await resync_course_completions_task(ctx) == 0


async def test_a_failing_sweep_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession, logger: Mock
) -> None:
    """Unlike the synchronous call sites this task backstops, the sweep may
    fail loudly: nothing of the student's is riding on it, and a silent
    backstop that stopped running would leave the drift it exists to catch
    accumulating unseen.
    """
    monkeypatch.setattr(
        completion_drift.completion_service,
        "resync_stale_course_completions",
        AsyncMock(side_effect=RuntimeError("query blew up")),
    )

    with pytest.raises(RuntimeError, match="query blew up"):
        await resync_course_completions_task({})

    assert session.committed == 0
