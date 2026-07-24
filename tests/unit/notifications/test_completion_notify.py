"""Unit tests for the material + quiz completion-notify helpers (T7.6).

These dispatchers are best-effort wrappers around ``send_notification`` that
the material-ingest worker and quiz-generation service call at their
success/failure transitions. The tests mock both the ``send_notification``
surface and the DB session so they stay pure (no Postgres):

* success + failure paths dispatch with the right category / entity / deep-link
* the material helper resolves version → lesson → course for the ``action_url``
* a missing recipient / target is skipped silently
* any internal error is swallowed (a notification must never break the job)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from abridgeai.features.materials.services import completion_notify as material_notify
from abridgeai.features.quizzes.services import completion_notify as quiz_notify


def _material_ctx_session(row: dict | None):
    """Fake AsyncSession whose execute().mappings().first() returns ``row``."""
    session = MagicMock()
    mappings = MagicMock()
    mappings.first.return_value = row
    result = MagicMock()
    result.mappings.return_value = mappings
    session.execute = AsyncMock(return_value=result)
    return session


# --------------------------------------------------------------------------- #
# Material processing                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_material_success_dispatches_ready_notification(monkeypatch) -> None:
    material_id, lesson_id, course_id = uuid4(), uuid4(), uuid4()
    version_id, recipient = uuid4(), uuid4()
    session = _material_ctx_session(
        {
            "material_id": material_id,
            "material_title": "Chapter 1",
            "lesson_id": lesson_id,
            "course_id": course_id,
        }
    )
    sent = AsyncMock()
    monkeypatch.setattr(material_notify, "send_notification", sent)

    await material_notify.notify_material_processing_outcome(
        session,
        recipient_user_id=recipient,
        material_version_id=version_id,
        succeeded=True,
        arq_pool=object(),
    )

    sent.assert_awaited_once()
    kwargs = sent.await_args.kwargs
    assert kwargs["recipient_user_id"] == recipient
    assert kwargs["notification_type"] == "material_processing"
    assert kwargs["entity_type"] == "material"
    assert kwargs["entity_id"] == material_id
    assert kwargs["action_url"] == f"/teacher/courses/{course_id}/lessons/{lesson_id}"
    assert "ready" in kwargs["title"].lower()


@pytest.mark.asyncio
async def test_material_failure_dispatches_failed_notification(monkeypatch) -> None:
    session = _material_ctx_session(
        {
            "material_id": uuid4(),
            "material_title": "Broken PDF",
            "lesson_id": uuid4(),
            "course_id": uuid4(),
        }
    )
    sent = AsyncMock()
    monkeypatch.setattr(material_notify, "send_notification", sent)

    await material_notify.notify_material_processing_outcome(
        session,
        recipient_user_id=uuid4(),
        material_version_id=uuid4(),
        succeeded=False,
        error_message="OCR timeout",
    )

    sent.assert_awaited_once()
    kwargs = sent.await_args.kwargs
    assert "fail" in kwargs["title"].lower()
    assert "OCR timeout" in kwargs["body"]


@pytest.mark.asyncio
async def test_material_missing_context_skips(monkeypatch) -> None:
    session = _material_ctx_session(None)  # version resolved to nothing
    sent = AsyncMock()
    monkeypatch.setattr(material_notify, "send_notification", sent)

    await material_notify.notify_material_processing_outcome(
        session,
        recipient_user_id=uuid4(),
        material_version_id=uuid4(),
        succeeded=True,
    )

    sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_material_notify_swallows_errors(monkeypatch) -> None:
    # execute() blows up — the helper must not propagate.
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    sent = AsyncMock()
    monkeypatch.setattr(material_notify, "send_notification", sent)

    await material_notify.notify_material_processing_outcome(
        session,
        recipient_user_id=uuid4(),
        material_version_id=uuid4(),
        succeeded=True,
    )

    sent.assert_not_awaited()  # never got far enough, but no exception raised


# --------------------------------------------------------------------------- #
# Quiz generation                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_quiz_success_dispatches_ready_notification(monkeypatch) -> None:
    quiz_id, course_id, recipient = uuid4(), uuid4(), uuid4()
    sent = AsyncMock()
    monkeypatch.setattr(quiz_notify, "send_notification", sent)

    await quiz_notify.notify_quiz_generation_outcome(
        MagicMock(),
        recipient_user_id=recipient,
        course_id=course_id,
        quiz_id=quiz_id,
        quiz_title="Week 3 Quiz",
        succeeded=True,
        arq_pool=object(),
    )

    sent.assert_awaited_once()
    kwargs = sent.await_args.kwargs
    assert kwargs["recipient_user_id"] == recipient
    assert kwargs["notification_type"] == "quiz_generation"
    assert kwargs["entity_type"] == "quiz"
    assert kwargs["entity_id"] == quiz_id
    assert kwargs["action_url"] == f"/teacher/courses/{course_id}/quizzes/{quiz_id}"
    assert "ready" in kwargs["title"].lower()


@pytest.mark.asyncio
async def test_quiz_failure_dispatches_failed_notification(monkeypatch) -> None:
    sent = AsyncMock()
    monkeypatch.setattr(quiz_notify, "send_notification", sent)

    await quiz_notify.notify_quiz_generation_outcome(
        MagicMock(),
        recipient_user_id=uuid4(),
        course_id=uuid4(),
        quiz_id=uuid4(),
        quiz_title="Week 3 Quiz",
        succeeded=False,
        error_message="LLM rate limit",
    )

    sent.assert_awaited_once()
    kwargs = sent.await_args.kwargs
    assert "fail" in kwargs["title"].lower()
    assert "LLM rate limit" in kwargs["body"]


@pytest.mark.asyncio
async def test_quiz_missing_recipient_skips(monkeypatch) -> None:
    sent = AsyncMock()
    monkeypatch.setattr(quiz_notify, "send_notification", sent)

    await quiz_notify.notify_quiz_generation_outcome(
        MagicMock(),
        recipient_user_id=None,  # system-initiated run
        course_id=uuid4(),
        quiz_id=uuid4(),
        quiz_title="Week 3 Quiz",
        succeeded=True,
    )

    sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_quiz_notify_swallows_errors(monkeypatch) -> None:
    sent = AsyncMock(side_effect=RuntimeError("dispatch exploded"))
    monkeypatch.setattr(quiz_notify, "send_notification", sent)

    # Should NOT raise even though send_notification blows up.
    await quiz_notify.notify_quiz_generation_outcome(
        MagicMock(),
        recipient_user_id=uuid4(),
        course_id=uuid4(),
        quiz_id=uuid4(),
        quiz_title="Week 3 Quiz",
        succeeded=True,
    )
