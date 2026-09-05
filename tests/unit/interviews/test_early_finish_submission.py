"""Unit coverage for explicitly ending an interview before all questions."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.interviews.services import taking as taking_service


def _session(*, assessment_started_at: datetime | None) -> SimpleNamespace:
    """A live assessment session.

    ``assessment_started_at`` is the field that decides gradeability: None means
    the candidate never got past onboarding (identity / audio / readiness), so
    there is nothing to judge.
    """
    return SimpleNamespace(
        id=uuid4(),
        interview_config_id=uuid4(),
        student_id=uuid4(),
        status="in_progress",
        interview_language="en",
        ended_at=None,
        assessment_started_at=assessment_started_at,
    )


def _db(*, user_message_count: int) -> SimpleNamespace:
    count_result = MagicMock()
    count_result.scalar_one.return_value = user_message_count
    return SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(time_limit_minutes=30)),
        execute=AsyncMock(return_value=count_result),
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )


async def _submit(session, db, arq, *, reason: str, terminalized: bool = True):
    """Run ``submit_session`` against the fake DB.

    Terminalization is an atomic conditional UPDATE (so two concurrent finishers
    cannot both relabel the row — see
    ``tests/integration/test_interview_finish_race.py``), which means the ORM
    instance is no longer what carries the new status; production re-reads it via
    ``db.refresh``. This stub applies the decided status to the in-memory session
    so these tests keep asserting the DECISION, which is what they are about.

    ``terminalized=False`` simulates losing that race: another caller already
    terminalized the session.
    """
    actor = SimpleNamespace(user_id=session.student_id)

    async def _terminalize(_db, _sid, *, status: str, ended_at):
        if terminalized:
            session.status = status
            session.ended_at = ended_at
        return terminalized

    with (
        patch.object(taking_service, "_require_session", AsyncMock(return_value=session)),
        patch.object(taking_service, "_assert_owns_session"),
        patch.object(taking_service, "ensure_ceremony_message", AsyncMock()),
        patch.object(
            taking_service.sessions_queries,
            "terminalize_in_progress_session",
            AsyncMock(side_effect=_terminalize),
        ),
    ):
        return await taking_service.submit_session(
            db,
            session.id,
            actor,  # type: ignore[arg-type]  -- SimpleNamespace stands in for CurrentUser
            arq_pool=arq,
            reason=reason,  # type: ignore[arg-type]  -- str stands in for FinishReason
        )


@pytest.mark.asyncio
async def test_early_finish_without_answers_is_completed_and_enqueued() -> None:
    """Reaching the assessment and skipping every question IS an assessment.

    The candidate saw the questions and chose not to answer, so the run is
    graded and the unanswered questions score zero. This is deliberate — do not
    "fix" it by requiring answers.
    """
    session = _session(assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC))
    db = _db(user_message_count=0)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    result = await _submit(session, db, arq, reason="ended_early")

    assert result.status == "completed"
    assert result.ended_at is not None
    arq.enqueue_job.assert_awaited_once_with(
        "evaluate_interview_session_task",
        session.student_id,
        session.id,
        _job_id=f"interview-evaluation:{session.id}",
    )


@pytest.mark.asyncio
async def test_quitting_during_onboarding_is_abandoned_and_never_graded() -> None:
    """A run that never reached the assessment has nothing to grade.

    Regression: this path used to be ``completed`` and enqueued, because the
    gate was ``user_message_count > 0 or reason != "timed_out"`` — the ``or``
    made any non-timeout submit gradeable regardless of content. In production
    that produced 14 sessions with outcome verdicts and a ``pass_verdict``
    derived from identity-check / audio-check chatter alone (9 of them with zero
    student messages), each having consumed one of the student's attempts.
    """
    session = _session(assessment_started_at=None)
    db = _db(user_message_count=0)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    result = await _submit(session, db, arq, reason="ended_early")

    assert result.status == "abandoned"
    assert result.ended_at is not None
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_onboarding_quit_is_abandoned_even_with_onboarding_chatter() -> None:
    """Onboarding replies are not answers.

    ``user_message_count`` only counts question-linked turns, so it is 0 here
    even though the transcript holds "Yes, that's me." / "The audio is clear.".
    The status must not depend on that chatter.
    """
    session = _session(assessment_started_at=None)
    db = _db(user_message_count=0)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    result = await _submit(session, db, arq, reason="natural")

    assert result.status == "abandoned"
    arq.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_answered_run_is_still_graded() -> None:
    """The normal path is untouched: answers present → completed and graded."""
    session = _session(assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC))
    db = _db(user_message_count=3)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    result = await _submit(session, db, arq, reason="natural")

    assert result.status == "completed"
    arq.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_timed_out_with_answers_is_timed_out_and_graded() -> None:
    """Deadline finish with answers stays ``timed_out`` and is graded."""
    session = _session(assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC))
    db = _db(user_message_count=2)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    with patch.object(
        taking_service, "utcnow", return_value=datetime(2026, 7, 24, 23, 0, tzinfo=UTC)
    ):
        result = await _submit(session, db, arq, reason="timed_out")

    assert result.status == "timed_out"
    arq.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_losing_the_terminalize_race_does_not_enqueue() -> None:
    """Another caller already ended this session, so it owns the enqueue.

    Without this, both the student's finish and the agent's hard-stop timer
    enqueued, and only ARQ's session-scoped job ID stopped a double grade — a
    property of the job-ID choice rather than of this function.
    """
    session = _session(assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC))
    db = _db(user_message_count=2)
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    await _submit(session, db, arq, reason="natural", terminalized=False)

    arq.enqueue_job.assert_not_awaited()
