"""A shutdown cancellation is not an evaluation failure.

The worker can be stopped at any moment — a deploy, an OOM guard, a restart.
The cancellation arrives as ``asyncio.CancelledError`` somewhere inside the
grading pipeline, i.e. AFTER the claim was taken. Treating that like an
exception used to stamp ``evaluation_failure`` (and ``status='failed'`` on the
final attempt) for work that never failed: the student saw an error badge for
a verdict that was never attempted, and the failure trail had to be explained
away by every later operator.

Pinned here:

* a CancelledError after the claim is re-raised, not converted;
* no ``evaluation_failure`` note and no ``status='failed'`` are written;
* the claim IS released (scoped to our token) so the recovery sweep can
  re-drive immediately instead of waiting out the lease;
* the release happens even when the failing session is mid-transaction — the
  cleanup runs on its own session, bounded and shielded.

The hard-kill case (no cancellation delivered at all) is deliberately NOT
covered here: the claim lease plus the sweep's ARQ reconciliation protect the
budget in that world, and there is no code left to test.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

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


def _session() -> Any:
    return SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        interview_config_id=uuid4(),
        assessment_started_at="2026-09-05T00:00:00+00:00",
        status="completed",
        pass_verdict=None,
        internal_summary_json={},
        evaluation_claim_token=None,
        evaluation_claim_expires_at=None,
    )


def _patch_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    calls: dict[str, Any] = {"claim": 0, "token": None}

    async def _claim(_db: Any, _sid: Any, *, token: UUID, now: Any, lease_expires_at: Any) -> bool:
        del now, lease_expires_at
        calls["claim"] += 1
        calls["token"] = token
        return True

    monkeypatch.setattr(evaluation_service.sessions_queries, "claim_session_evaluation", _claim)
    return calls


@pytest.mark.asyncio
async def test_a_cancellation_never_stamps_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE BUG: shutdown used to relabel a healthy session as failed."""
    row = _session()
    db = _FakeDB({})
    released: list[UUID] = []

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return row

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    _patch_claim(monkeypatch)

    async def _release(_db: Any, _sid: Any, *, token: UUID) -> None:
        released.append(token)

    monkeypatch.setattr(
        evaluation_service.sessions_queries,
        "release_session_evaluation_claim",
        _release,
    )

    class _FakeSessionCtx:
        async def __aenter__(self) -> Any:
            return db

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(
        evaluation_service,
        "get_sessionmaker",
        lambda: (lambda *a, **k: _FakeSessionCtx()),
        raising=False,
    )

    async def _cancel_mid_pipeline(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        evaluation_service.authoring_queries,
        "list_outcomes_for_config",
        _cancel_mid_pipeline,
    )

    with pytest.raises(asyncio.CancelledError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            row.id,
            is_final_attempt=True,
        )

    assert row.status == "completed", (
        "shutdown cancellation must not relabel the session as failed"
    )
    assert "evaluation_failure" not in row.internal_summary_json
    assert released, "the claim must be released so the sweep can re-drive at once"
    assert db.rollbacks >= 1, "the interrupted transaction is rolled back first"


@pytest.mark.asyncio
async def test_a_failing_cleanup_does_not_mask_the_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the release itself blows up, the cancellation still propagates."""
    row = _session()
    db = _FakeDB({})

    async def _get_session(_db: Any, _session_id: Any) -> Any:
        return row

    monkeypatch.setattr(evaluation_service.sessions_queries, "get_session", _get_session)
    _patch_claim(monkeypatch)

    async def _release_fails(_db: Any, _sid: Any, *, token: UUID) -> None:
        del token
        raise RuntimeError("cleanup connection refused")

    monkeypatch.setattr(
        evaluation_service.sessions_queries,
        "release_session_evaluation_claim",
        _release_fails,
    )

    class _FakeSessionCtx:
        async def __aenter__(self) -> Any:
            return db

        async def __aexit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(
        evaluation_service,
        "get_sessionmaker",
        lambda: (lambda *a, **k: _FakeSessionCtx()),
        raising=False,
    )

    async def _cancel_mid_pipeline(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        evaluation_service.authoring_queries,
        "list_outcomes_for_config",
        _cancel_mid_pipeline,
    )

    with pytest.raises(asyncio.CancelledError):
        await evaluation_service.evaluate_and_generate_report(
            db,  # type: ignore[arg-type]
            row.id,
            is_final_attempt=True,
        )

    assert row.status == "completed"
    assert "evaluation_failure" not in row.internal_summary_json
