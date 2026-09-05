"""A published verdict is final: no later job may overwrite or re-grade it.

Two evaluation jobs can be in flight for the same session. The recovery sweep
enqueues under a per-attempt job ID (so ARQ cannot dedupe it) while the original
job may still be running: ``WorkerSettings.job_timeout`` is 20 minutes and the
sweep's grace is 15, so the windows overlap by design.

When that happens the two jobs race, and the loser used to win. The failure
branch of ``evaluate_and_generate_report`` re-read the row and stamped
``evaluation_failure`` (plus ``status='failed'`` on the final ARQ attempt)
without looking at ``pass_verdict``. A session that had *already been graded*
by the sibling job was therefore relabelled as a grader failure — and because
the recovery query filters on ``pass_verdict IS NULL``, nothing ever repaired
it. The student's history showed a permanent error for a graded interview.

Two invariants pinned here:

* an infrastructure failure never overwrites a published verdict
  (``pass_verdict=False`` is published too — it is a real judgement);
* a session that already carries a verdict is not graded a second time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.models import InterviewSession
from abridgeai.features.interviews.services import evaluation as evaluation_service


class _FakeDB:
    """Minimal AsyncSession stand-in: ``get`` dispatches on the model class."""

    def __init__(self, rows: dict[type, Any]) -> None:
        self._rows = rows
        self.commits = 0
        self.rollbacks = 0

    async def get(self, model: type, _pk: Any) -> Any:
        return self._rows.get(model)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


def _session(*, pass_verdict: bool | None, status: str = "completed") -> Any:
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        interview_config_id=uuid4(),
        assessment_started_at="2026-09-05T00:00:00+00:00",
        status=status,
        pass_verdict=pass_verdict,
        internal_summary_json={"total_score": 80.0},
    )


@pytest.mark.asyncio
async def test_a_stale_failure_never_overwrites_a_published_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sibling job graded it. This job's crash must not relabel it failed."""
    graded = _session(pass_verdict=True)
    db = _FakeDB({InterviewSession: graded})

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        # What THIS job read when it started: not yet graded.
        return _session(pass_verdict=None)

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LLM provider 503")

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _boom)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            graded.id,
            is_final_attempt=True,
        )

    assert graded.status == "completed", (
        "a graded session must not be relabelled as a grader failure by a stale job"
    )
    assert "evaluation_failure" not in graded.internal_summary_json, (
        "the failure note would make a graded interview read as broken"
    )


@pytest.mark.asyncio
async def test_a_published_failing_verdict_is_also_protected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pass_verdict=False`` is a real judgement, not an absence of one."""
    graded = _session(pass_verdict=False)
    db = _FakeDB({InterviewSession: graded})

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return _session(pass_verdict=None)

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _boom)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            graded.id,
            is_final_attempt=True,
        )

    assert graded.status == "completed"
    assert "evaluation_failure" not in graded.internal_summary_json


@pytest.mark.asyncio
async def test_a_still_failing_ungraded_session_is_stamped_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not disarm the real failure path for ungraded rows."""
    ungraded = _session(pass_verdict=None)
    db = _FakeDB({InterviewSession: ungraded})

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return ungraded

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("still broken")

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _boom)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            ungraded.id,
            is_final_attempt=True,
        )

    assert ungraded.status == "failed"
    assert ungraded.internal_summary_json["evaluation_failure"]["message"] == "still broken"


@pytest.mark.asyncio
async def test_an_already_graded_session_is_not_graded_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate job returns early instead of re-running the judge.

    Re-grading would spend LLM budget, write a second gap report, and could
    flip a verdict the student has already been shown.
    """
    graded = _session(pass_verdict=True)
    db = _FakeDB({InterviewSession: graded})
    stage_calls: list[str] = []

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return graded

    async def _outcomes(*_args: Any, **_kwargs: Any) -> Any:
        stage_calls.append("list_outcomes_for_config")
        return []

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _outcomes)

    await evaluation_service.evaluate_and_generate_report(
        db,  # type: ignore[arg-type]
        graded.id,
    )

    assert stage_calls == [], "the judge must not run again for a published verdict"
    assert graded.pass_verdict is True
