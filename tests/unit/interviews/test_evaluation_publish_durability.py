"""A committed verdict must survive a failing side effect.

``evaluate_and_generate_report`` ends with a course-completion sync: an interview
is a gradeable curriculum unit, so a passing verdict can promote the enrollment and
unlock the next career-path stage. That call is best-effort by design — the nightly
drift sweeper repairs a miss — and its exception is swallowed.

Swallowing it was only safe once the call moved OUT of the verdict's transaction.
``sync_course_completion`` flushes through ``flush_or_conflict``, which ROLLS BACK
the session before re-raising. Inside the verdict's transaction that rollback
discarded the uncommitted verdict, gap report and outcome rows; the swallow then
let execution reach ``db.commit()``, which committed nothing. The job logged
success, the interview silently reverted to ungraded, and — because the claim was
cleared in the same discarded transaction — the row still held its 30-minute lease,
so the recovery sweep could not re-drive it either.

Pinned here: the verdict is committed BEFORE the sync runs, and a sync that blows
up (rollback included) neither loses the verdict nor fails the job.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.services import evaluation as evaluation_service


class _RecordingDB:
    """Tracks commit/rollback ordering and snapshots the row at each commit."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self.events: list[str] = []
        self.committed_verdicts: list[Any] = []

    async def commit(self) -> None:
        self.events.append("commit")
        self.committed_verdicts.append(self._session.pass_verdict)

    async def rollback(self) -> None:
        self.events.append("rollback")
        # Emulate what a real rollback does to the pending write.
        self._session.pass_verdict = None

    async def get(self, _model: type, _pk: Any) -> Any:
        return None


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


@pytest.mark.asyncio
async def test_a_sync_that_rolls_back_cannot_take_the_verdict_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shape of the bug, in the order the real code now runs it.

    ``flush_or_conflict`` rolls the session back before re-raising. Because the
    verdict is committed FIRST and the sync runs in its own transaction, that
    rollback has nothing of the evaluation left to discard.
    """
    row = _session()
    db = _RecordingDB(row)

    # Publish, as the service does before returning from its try block.
    row.pass_verdict = True
    await db.commit()
    assert db.committed_verdicts == [True]

    async def _sync_that_rolls_back(_db: Any, *, course_id: Any, student_id: Any) -> None:
        await _db.rollback()  # what flush_or_conflict does on IntegrityError
        raise RuntimeError("duplicate completion award")

    monkeypatch.setattr(
        "abridgeai.features.enrollments.api.public.sync_course_completion",
        _sync_that_rolls_back,
    )

    await evaluation_service._sync_course_completion(
        db,  # type: ignore[arg-type]
        student_id=row.student_id,
        course_id=uuid4(),
    )

    assert db.committed_verdicts == [True], (
        "the verdict commit must already have happened before the sync could fail"
    )
    assert db.events == ["commit", "rollback", "rollback"]


@pytest.mark.asyncio
async def test_a_failing_completion_sync_does_not_raise_into_the_job() -> None:
    """The evaluation succeeded. A drift-repairable side effect must not fail it."""
    row = _session()
    row.pass_verdict = True
    db = _RecordingDB(row)

    async def _boom(_db: Any, *, course_id: Any, student_id: Any) -> None:
        raise RuntimeError("boom")

    import abridgeai.features.enrollments.api.public as enrollments_public

    original = enrollments_public.sync_course_completion
    enrollments_public.sync_course_completion = _boom  # type: ignore[assignment]
    try:
        await evaluation_service._sync_course_completion(
            db,  # type: ignore[arg-type]
            student_id=row.student_id,
            course_id=uuid4(),
        )
    finally:
        enrollments_public.sync_course_completion = original  # type: ignore[assignment]

    assert db.events == ["rollback"], "a failed side effect must not commit half state"


@pytest.mark.asyncio
async def test_a_successful_completion_sync_commits_its_own_transaction() -> None:
    """It no longer rides someone else's commit, so it has to own one."""
    row = _session()
    row.pass_verdict = True
    db = _RecordingDB(row)

    async def _ok(_db: Any, *, course_id: Any, student_id: Any) -> str:
        return "completed"

    import abridgeai.features.enrollments.api.public as enrollments_public

    original = enrollments_public.sync_course_completion
    enrollments_public.sync_course_completion = _ok  # type: ignore[assignment]
    try:
        await evaluation_service._sync_course_completion(
            db,  # type: ignore[arg-type]
            student_id=row.student_id,
            course_id=uuid4(),
        )
    finally:
        enrollments_public.sync_course_completion = original  # type: ignore[assignment]

    assert db.events == ["commit"]


def test_the_completion_sync_is_called_after_the_commit_in_the_source() -> None:
    """Structural pin: the call must sit outside the try/except that commits.

    An ordering bug here is invisible until a conflict actually fires in
    production, so pin the shape rather than only the behaviour.
    """
    import inspect

    source = inspect.getsource(evaluation_service.evaluate_and_generate_report)
    sync_at = source.index("_sync_course_completion(")
    except_at = source.index("    except Exception as exc:")
    assert sync_at > except_at, (
        "_sync_course_completion moved back inside the verdict's transaction — a "
        "rollback in it would discard the verdict and commit an empty transaction"
    )
