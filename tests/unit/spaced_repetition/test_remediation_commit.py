"""The remediation notification has to survive the request that wrote it.

``record_card_review`` hands a ``CardFailedEvent`` back to the router instead
of firing it, so that a rolled-back review can never produce a notification for
a review that did not happen. The router's side of that bargain is a three-step
sequence: commit the review, dispatch the remediation, commit **again**.

The second commit is the one that was missing, and nothing caught it, because
the dispatcher and everything under it flush without committing — by design,
``send_notification`` leaves the transaction to its caller — while the
request-scoped session from ``get_db`` is closed rather than committed at the
end of the request. A flushed-but-uncommitted notification row is therefore
rolled back on the way out: the student fails a card, the remediation is
assembled correctly, and the row it produced quietly disappears. Nothing logs,
nothing raises, and the existing service-level tests pass because they commit
themselves.

Both endpoints that grade a card are covered — answering inside a quiz, and
answering a card in the review loop — because both carry the same events and
both had the same omission.

No database: the session is a recorder and what is asserted is the order of
calls on it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.api._dto import GradeReviewResultDTO
from abridgeai.features.quizzes.routers import learner as quiz_router
from abridgeai.features.spaced_repetition.routers import learner as sr_router
from abridgeai.features.spaced_repetition.schemas.review import ReviewSubmitRequest
from abridgeai.features.spaced_repetition.services._events import CardFailedEvent
from abridgeai.features.spaced_repetition.services.review import CardReviewResult

STUDENT_ID = uuid4()
QUESTION_ID = uuid4()
QUIZ_ID = uuid4()
ATTEMPT_ID = uuid4()


class _RecordingSession:
    """Stands in for the request session, remembering what happened to it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def commit(self) -> None:
        self.calls.append("commit")

    async def rollback(self) -> None:
        self.calls.append("rollback")


def _failed_review() -> CardReviewResult:
    """A q==0 review — the only grade that queues a remediation event."""
    return CardReviewResult(
        q=0,
        ef_before=2.5,
        ef_after=2.5,
        interval_before=1,
        interval_after=1,
        repetition_count_after=0,
        due_at=datetime.now(tz=UTC),
        interval_seconds=86400,
        last_q=0,
        passing=False,
        retry_available_at=datetime.now(tz=UTC),
        calibration_active=True,
        pending_events=[
            CardFailedEvent(
                student_id=STUDENT_ID,
                question_id=QUESTION_ID,
                quiz_attempt_id=ATTEMPT_ID,
                quiz_id=QUIZ_ID,
                timestamp=datetime.now(tz=UTC),
            )
        ],
    )


def _user() -> CurrentUser:
    return CurrentUser(user_id=STUDENT_ID, session_id=uuid4())


def _patch_dispatch(
    monkeypatch: pytest.MonkeyPatch, module: Any, session: _RecordingSession, *, fail: bool = False
) -> None:
    """Record the dispatch on the session's timeline, optionally blowing up."""

    async def _dispatch(db: Any, **kwargs: Any) -> None:
        session.calls.append("dispatch")
        if fail:
            raise RuntimeError("knowledge graph unreachable")

    monkeypatch.setattr(module, "dispatch_remediation_for_card_failure", _dispatch)


@pytest.fixture
def session() -> _RecordingSession:
    return _RecordingSession()


@pytest.fixture
def review_endpoint(monkeypatch: pytest.MonkeyPatch, session: _RecordingSession) -> None:
    """Stub everything around the review endpoint except its transaction work."""

    async def _grade(db: Any, **kwargs: Any) -> GradeReviewResultDTO:
        return GradeReviewResultDTO(is_correct=False)

    async def _record(db: Any, **kwargs: Any) -> CardReviewResult:
        return _failed_review()

    async def _remaining(db: Any, student_id: Any) -> int:
        return 3

    monkeypatch.setattr(sr_router, "grade_review_answer", _grade)
    monkeypatch.setattr(sr_router, "record_card_review", _record)
    monkeypatch.setattr(sr_router, "get_due_card_count", _remaining)


@pytest.fixture
def answer_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub everything around the quiz-answer endpoint except its transaction work."""

    class _Answer:
        id = uuid4()
        attempt_id = ATTEMPT_ID
        question_id = QUESTION_ID
        selected_option_id = None

    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    async def _answer_attempt(db: Any, *args: Any, **kwargs: Any) -> tuple[Any, CardReviewResult]:
        return _Answer(), _failed_review()

    monkeypatch.setattr(quiz_router, "_require_live_owned_attempt", _noop)
    monkeypatch.setattr(quiz_router, "_require_session_owner", _noop)
    monkeypatch.setattr(quiz_router.taking_service, "answer_attempt", _answer_attempt)


class TestReviewLoop:
    async def test_the_notification_is_committed_after_it_is_dispatched(
        self, session: _RecordingSession, review_endpoint: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dispatch(monkeypatch, sr_router, session)

        await sr_router.submit_review(QUESTION_ID, ReviewSubmitRequest(), _user(), session)

        # Review commit, then dispatch, then the commit that makes the
        # notification durable. Without the last one the row is rolled back
        # when the request session closes.
        assert session.calls == ["commit", "dispatch", "commit"]

    async def test_a_failed_dispatch_leaves_no_open_transaction(
        self, session: _RecordingSession, review_endpoint: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dispatch(monkeypatch, sr_router, session, fail=True)

        result = await sr_router.submit_review(QUESTION_ID, ReviewSubmitRequest(), _user(), session)

        assert session.calls == ["commit", "dispatch", "rollback"]
        assert result.question_id == QUESTION_ID  # the review itself still stands


class TestInQuizAnswer:
    async def test_the_notification_is_committed_after_it_is_dispatched(
        self, session: _RecordingSession, answer_endpoint: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dispatch(monkeypatch, quiz_router, session)

        await quiz_router.record_answer(ATTEMPT_ID, object(), _user(), session)

        assert session.calls == ["commit", "dispatch", "commit"]

    async def test_a_failed_dispatch_leaves_no_open_transaction(
        self, session: _RecordingSession, answer_endpoint: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_dispatch(monkeypatch, quiz_router, session, fail=True)

        answer = await quiz_router.record_answer(ATTEMPT_ID, object(), _user(), session)

        assert session.calls == ["commit", "dispatch", "rollback"]
        assert answer.question_id == QUESTION_ID  # the answer itself still stands
