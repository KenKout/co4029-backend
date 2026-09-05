"""A published verdict is final, and only the claim owner may write an outcome.

Two evaluation jobs can be in flight for the same session. The recovery sweep
enqueues under a per-attempt job ID (so ARQ cannot dedupe it) while the original
job may still be running: ``WorkerSettings.job_timeout`` is 20 minutes and the
sweep's grace is 15, so the windows overlap by design.

When that happened the loser used to win. The failure branch of
``evaluate_and_generate_report`` re-read the row and stamped
``evaluation_failure`` (plus ``status='failed'`` on the final ARQ attempt)
without asking whether it was still the session's owner. A session that had
*already been graded* by the sibling job was therefore relabelled as a grader
failure — and because the recovery query filters on ``pass_verdict IS NULL``,
nothing ever repaired it. The student's history showed a permanent error for a
graded interview.

The guard is now ownership, not a ``pass_verdict`` read: the job claims the
session before judging and re-checks that its lease is still its own before
publishing OR stamping a failure. That closes the window a value check leaves
open — a full grading pass sits between reading ``pass_verdict`` and writing.

Invariants pinned here:

* an unclaimable session (already graded, or claimed by a live job) is never
  judged — the stages do not even run;
* a job that lost its lease writes neither a verdict nor a failure;
* the ordinary failure path still records the trail for the rightful owner, and
  releases the claim so the next retry can start immediately.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

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


def _session(*, pass_verdict: bool | None = None, status: str = "completed") -> Any:
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        interview_config_id=uuid4(),
        assessment_started_at="2026-09-05T00:00:00+00:00",
        status=status,
        pass_verdict=pass_verdict,
        internal_summary_json={"total_score": 80.0},
        evaluation_claim_token=None,
        evaluation_claim_expires_at=None,
    )


def _patch_session_read(monkeypatch: pytest.MonkeyPatch, row: Any) -> None:
    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return row

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)


def _patch_claim(
    monkeypatch: pytest.MonkeyPatch,
    *,
    claimed: bool,
    still_owner: bool = True,
) -> dict[str, Any]:
    """Stub the claim helpers; records what the service asked for."""
    calls: dict[str, Any] = {"claim": 0, "owns": 0, "token": None}

    async def _claim(_db: Any, _sid: Any, *, token: UUID, now: Any, lease_expires_at: Any) -> bool:
        calls["claim"] += 1
        calls["token"] = token
        return claimed

    async def _owns(_db: Any, _sid: Any, *, token: UUID, now: Any) -> bool:
        calls["owns"] += 1
        assert token == calls["token"], "ownership must be checked with the token we claimed"
        return still_owner

    monkeypatch.setattr(
        evaluation_service.sessions_queries, "claim_session_evaluation", _claim
    )
    monkeypatch.setattr(
        evaluation_service.sessions_queries, "holds_session_evaluation_claim", _owns
    )
    return calls


@pytest.mark.asyncio
async def test_an_unclaimable_session_is_never_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already graded, or owned by a live job: the stages must not run at all."""
    graded = _session(pass_verdict=True)
    db = _FakeDB({InterviewSession: graded})
    stage_calls: list[str] = []

    async def _outcomes(*_args: Any, **_kwargs: Any) -> Any:
        stage_calls.append("list_outcomes_for_config")
        return []

    _patch_session_read(monkeypatch, graded)
    calls = _patch_claim(monkeypatch, claimed=False)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _outcomes)

    await evaluation_service.evaluate_and_generate_report(db, graded.id)  # type: ignore[arg-type]

    assert calls["claim"] == 1
    assert stage_calls == [], "the judge must not run without the claim"
    assert graded.pass_verdict is True
    assert graded.status == "completed"


@pytest.mark.asyncio
async def test_a_stale_owner_does_not_stamp_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lease lost mid-run: the crash trail belongs to whoever owns it now."""
    row = _session(pass_verdict=None)
    db = _FakeDB({InterviewSession: row})

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("LLM provider 503")

    _patch_session_read(monkeypatch, row)
    _patch_claim(monkeypatch, claimed=True, still_owner=False)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _boom)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            row.id,
            is_final_attempt=True,
        )

    assert row.status == "completed", (
        "a superseded job must not relabel a session it no longer owns"
    )
    assert "evaluation_failure" not in row.internal_summary_json


@pytest.mark.asyncio
async def test_the_rightful_owner_still_records_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not disarm the real failure path."""
    row = _session(pass_verdict=None)
    row.evaluation_claim_token = uuid4()
    row.evaluation_claim_expires_at = "2026-09-05T01:00:00+00:00"
    db = _FakeDB({InterviewSession: row})

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("still broken")

    _patch_session_read(monkeypatch, row)
    _patch_claim(monkeypatch, claimed=True, still_owner=True)
    monkeypatch.setattr(evaluation_service.authoring_queries, "list_outcomes_for_config", _boom)

    with pytest.raises(RuntimeError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            row.id,
            is_final_attempt=True,
        )

    assert row.status == "failed"
    assert row.internal_summary_json["evaluation_failure"]["message"] == "still broken"
    assert row.evaluation_claim_token is None, (
        "a failed pass must release its lease so the next retry can claim at once"
    )
    assert row.evaluation_claim_expires_at is None


@pytest.mark.asyncio
async def test_a_session_that_never_reached_the_assessment_is_not_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ungradeable refusal stays AHEAD of the claim — don't burn a lease."""
    row = _session(pass_verdict=None)
    row.assessment_started_at = None
    db = _FakeDB({InterviewSession: row})

    _patch_session_read(monkeypatch, row)
    calls = _patch_claim(monkeypatch, claimed=True)

    await evaluation_service.evaluate_and_generate_report(db, row.id)  # type: ignore[arg-type]

    assert calls["claim"] == 0, "an ungradeable session must not be claimed at all"
