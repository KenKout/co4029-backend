"""Unit coverage for explicitly ending an interview before all questions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.interviews.services import taking as taking_service


@pytest.mark.asyncio
async def test_early_finish_without_answers_is_completed_and_enqueued() -> None:
    session = SimpleNamespace(
        id=uuid4(),
        interview_config_id=uuid4(),
        student_id=uuid4(),
        status="in_progress",
        session_mode="assessment",
        interview_language="en",
        ended_at=None,
    )
    actor = SimpleNamespace(user_id=session.student_id)
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    db = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(time_limit_minutes=30)),
        execute=AsyncMock(return_value=count_result),
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    with (
        patch.object(taking_service, "_require_session", AsyncMock(return_value=session)),
        patch.object(taking_service, "_assert_owns_session"),
        patch.object(taking_service, "ensure_ceremony_message", AsyncMock()),
    ):
        result = await taking_service.submit_session(
            db,
            session.id,
            actor,
            arq_pool=arq,
            reason="ended_early",
        )

    assert result.status == "completed"
    assert result.ended_at is not None
    arq.enqueue_job.assert_awaited_once_with(
        "evaluate_interview_session_task",
        actor.user_id,
        session.id,
        _job_id=f"interview-evaluation:{session.id}",
    )
