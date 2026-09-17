"""Telling people what happened to a path-change request.

A dean's moves on a request are invisible to the student who filed it
unless something says so, and a request sitting in a dean's queue is
invisible to the dean unless something says so. These four functions are
that something, and they are all the same shape, which is what makes the
shared properties worth testing once across the set rather than four times
by hand.

Two of those properties carry real weight.

**Failure is swallowed.** Every one of these is called from inside the
transaction that records the decision. A dean's approval is a change to a
student's academic record; losing it because an inbox row could not be
written would be far worse than the student not being told. So a notify
failure is logged and absorbed, and the decision commits regardless.

**The link differs by audience on purpose.** A dean is sent to the
specific program's review queue, because their inbox is organised by
program and the badge on a program card is where they notice. A student is
sent to their own request record -- deliberately not to the career path
they were refused, which is the screen they least need after a rejection.

Everything below the module under test is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from abridgeai.features.learning_programs import notify


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture every ``send_notification`` call as its kwargs."""
    calls: list[dict[str, Any]] = []

    async def _send(_db: object, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(notify.notifications_api, "send_notification", _send)
    return calls


@pytest.fixture
def locale(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    stub = AsyncMock(return_value="en")
    monkeypatch.setattr(notify, "get_user_locale", stub)
    return stub


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> Mock:
    recorder = Mock()
    monkeypatch.setattr(notify, "_logger", recorder)
    return recorder


def _student_call(name: str) -> tuple[Any, dict[str, Any]]:
    """One of the three student-facing notifiers, with its required kwargs."""
    common = {
        "student_user_id": uuid4(),
        "request_id": uuid4(),
        "program_name": "BSc Software Engineering",
        "target_path_name": "Data Engineering",
    }
    if name == "rejected":
        return notify.notify_path_change_rejected, {
            **common,
            "reason_code": "prerequisites_unmet",
            "reason_detail": None,
            "note": None,
        }
    fn = {
        "in_progress": notify.notify_path_change_in_progress,
        "approved": notify.notify_path_change_approved,
    }[name]
    return fn, common


STUDENT_EVENTS = ["in_progress", "rejected", "approved"]


class TestEveryNotificationCarriesTheSameHandle:
    """All four ride one category and one entity type.

    The category is what the student's preference screen toggles: one
    switch for "decisions about my path change" rather than three. The
    entity pair is how the SPA links an inbox row back to the request it is
    about, so a row missing it is a dead end.
    """

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_a_student_notification_names_its_request(
        self, event: str, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        fn, kwargs = _student_call(event)

        await fn(object(), **kwargs)

        assert len(sent) == 1
        assert sent[0]["notification_type"] == "path_change_review"
        assert sent[0]["entity_type"] == "path_change_request"
        assert sent[0]["entity_id"] == kwargs["request_id"]
        assert sent[0]["recipient_user_id"] == kwargs["student_user_id"]

    async def test_the_dean_notification_names_the_same_request(
        self, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        request_id, dean_id = uuid4(), uuid4()

        await notify.notify_dean_path_change_requested(
            object(),
            dean_user_id=dean_id,
            request_id=request_id,
            program_id=uuid4(),
            program_name="BSc Software Engineering",
            student_label="Nguyen Van A",
            target_path_name="Data Engineering",
        )

        assert sent[0]["notification_type"] == "path_change_review"
        assert sent[0]["entity_id"] == request_id
        assert sent[0]["recipient_user_id"] == dean_id


class TestWhereEachAudienceIsSent:
    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_a_student_lands_on_their_own_request_record(
        self, event: str, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        """Not a deep link to the path they were refused.

        After a rejection the useful screen is their request and its
        reason; the path itself is the one thing they cannot act on.
        """
        fn, kwargs = _student_call(event)

        await fn(object(), **kwargs)

        assert sent[0]["action_url"] == "/learning-programs"

    async def test_a_dean_lands_on_that_programs_review_queue(
        self, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        """A dean owning several programs needs the one the badge is on."""
        program_id = uuid4()

        await notify.notify_dean_path_change_requested(
            object(),
            dean_user_id=uuid4(),
            request_id=uuid4(),
            program_id=program_id,
            program_name="BSc Software Engineering",
            student_label="Nguyen Van A",
            target_path_name="Data Engineering",
        )

        assert sent[0]["action_url"] == (
            f"/management/learning-programs/{program_id}?tab=requests"
        )


class TestCopyIsRenderedInTheRecipientsLanguage:
    """The locale is resolved per recipient, not per request.

    A Vietnamese student's rejection reason and the English-speaking dean's
    queue alert come out of the same review action, so reading the locale
    once for "the request" would put one of them in the wrong language.
    """

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_the_student_is_the_one_looked_up(
        self, event: str, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        fn, kwargs = _student_call(event)

        await fn(object(), **kwargs)

        assert locale.await_args.args[1] == kwargs["student_user_id"]

    async def test_the_dean_is_the_one_looked_up_for_their_own_alert(
        self, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        dean_id = uuid4()

        await notify.notify_dean_path_change_requested(
            object(),
            dean_user_id=dean_id,
            request_id=uuid4(),
            program_id=uuid4(),
            program_name="P",
            student_label="S",
            target_path_name="T",
        )

        assert locale.await_args.args[1] == dean_id

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_the_resolved_locale_reaches_the_copy_builders(
        self, event: str, monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]]
    ) -> None:
        """Resolving the locale and then not threading it through would
        produce English copy for a Vietnamese reader with nothing failing.
        """
        monkeypatch.setattr(notify, "get_user_locale", AsyncMock(return_value="vi"))
        seen: list[str] = []

        def _record(**kwargs: Any) -> str:
            seen.append(kwargs["locale"])
            return "copy"

        for builder in (
            "path_change_in_progress_title",
            "path_change_in_progress_body",
            "path_change_rejected_title",
            "path_change_rejected_body",
            "path_change_approved_title",
            "path_change_approved_body",
        ):
            monkeypatch.setattr(notify.notifications_api, builder, _record)

        fn, kwargs = _student_call(event)
        await fn(object(), **kwargs)

        assert seen == ["vi", "vi"], "both the title and the body are localized"


class TestTheKindChangesTheWording:
    """A drop request and a change request ride the same three moments.

    Only ``kind`` separates them, and it has to reach the copy builders: a
    student who asked to *leave* a path must not be told they "stay on
    their current path".
    """

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    @pytest.mark.parametrize("kind", ["change", "drop"])
    async def test_the_kind_reaches_the_body_builder(
        self,
        event: str,
        kind: str,
        monkeypatch: pytest.MonkeyPatch,
        sent: list[dict[str, Any]],
        locale: AsyncMock,
    ) -> None:
        seen: list[str] = []

        def _record(**kwargs: Any) -> str:
            if "kind" in kwargs:
                seen.append(kwargs["kind"])
            return "copy"

        for builder in (
            "path_change_in_progress_body",
            "path_change_rejected_body",
            "path_change_approved_body",
        ):
            monkeypatch.setattr(notify.notifications_api, builder, _record)

        fn, kwargs = _student_call(event)
        await fn(object(), **kwargs, kind=kind)

        assert seen == [kind]

    async def test_change_is_the_default_kind(
        self, monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        seen: list[str] = []

        def _record(**kwargs: Any) -> str:
            if "kind" in kwargs:
                seen.append(kwargs["kind"])
            return "copy"

        monkeypatch.setattr(notify.notifications_api, "path_change_approved_body", _record)

        fn, kwargs = _student_call("approved")
        await fn(object(), **kwargs)

        assert seen == ["change"]


class TestTheRejectionReasonTravels:
    async def test_the_structured_reason_and_the_deans_words_both_reach_the_body(
        self, monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]], locale: AsyncMock
    ) -> None:
        """"Rejected" with no reason is what makes a student re-file the
        same request, so the reason is the point of the notification rather
        than a detail of it. The free-text note carries the dean's own
        words when they picked ``other``.
        """
        captured: dict[str, Any] = {}

        def _record(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "copy"

        monkeypatch.setattr(notify.notifications_api, "path_change_rejected_body", _record)

        await notify.notify_path_change_rejected(
            object(),
            student_user_id=uuid4(),
            request_id=uuid4(),
            program_name="BSc Software Engineering",
            target_path_name="Data Engineering",
            reason_code="other",
            reason_detail="Capacity reached for this intake",
            note="Try again next semester.",
        )

        assert captured["reason_code"] == "other"
        assert captured["reason_detail"] == "Capacity reached for this intake"
        assert captured["note"] == "Try again next semester."


class TestAFailedNotificationNeverBreaksTheDecision:
    """The property that makes these safe to call inside the review
    transaction.

    The decision has already been written by the time these run. Letting an
    exception out would roll back an academic record change because an
    inbox row could not be written.
    """

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_a_dispatch_failure_is_absorbed(
        self,
        event: str,
        monkeypatch: pytest.MonkeyPatch,
        locale: AsyncMock,
        logger: Mock,
    ) -> None:
        monkeypatch.setattr(
            notify.notifications_api,
            "send_notification",
            AsyncMock(side_effect=RuntimeError("inbox write failed")),
        )

        fn, kwargs = _student_call(event)

        assert await fn(object(), **kwargs) is None
        logger.exception.assert_called_once()

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_a_locale_lookup_failure_is_absorbed_too(
        self,
        event: str,
        monkeypatch: pytest.MonkeyPatch,
        sent: list[dict[str, Any]],
        logger: Mock,
    ) -> None:
        """The lookup happens first, so it is the earliest thing that can
        fail -- and it is a database read on the same connection the review
        is using."""
        monkeypatch.setattr(
            notify, "get_user_locale", AsyncMock(side_effect=RuntimeError("db is gone"))
        )

        fn, kwargs = _student_call(event)

        assert await fn(object(), **kwargs) is None
        assert sent == []
        logger.exception.assert_called_once()

    async def test_a_failed_dean_alert_is_absorbed(
        self, monkeypatch: pytest.MonkeyPatch, locale: AsyncMock, logger: Mock
    ) -> None:
        monkeypatch.setattr(
            notify.notifications_api,
            "send_notification",
            AsyncMock(side_effect=RuntimeError("inbox write failed")),
        )

        result = await notify.notify_dean_path_change_requested(
            object(),
            dean_user_id=uuid4(),
            request_id=uuid4(),
            program_id=uuid4(),
            program_name="P",
            student_label="S",
            target_path_name="T",
        )

        assert result is None
        logger.exception.assert_called_once()

    @pytest.mark.parametrize("event", STUDENT_EVENTS)
    async def test_the_failure_log_names_the_request(
        self,
        event: str,
        monkeypatch: pytest.MonkeyPatch,
        locale: AsyncMock,
        logger: Mock,
    ) -> None:
        """A swallowed failure is only recoverable if the log says which
        student was never told about which request."""
        monkeypatch.setattr(
            notify.notifications_api,
            "send_notification",
            AsyncMock(side_effect=RuntimeError("boom")),
        )

        fn, kwargs = _student_call(event)
        await fn(object(), **kwargs)

        fields = logger.exception.call_args.kwargs
        assert fields["request_id"] == str(kwargs["request_id"])
        assert fields["student_user_id"] == str(kwargs["student_user_id"])


def test_only_the_four_notifiers_are_exported() -> None:
    """The module is imported by the review service; its constants and the
    logger are internals."""
    assert notify.__all__ == [
        "notify_dean_path_change_requested",
        "notify_path_change_approved",
        "notify_path_change_in_progress",
        "notify_path_change_rejected",
    ]


def test_all_four_share_one_preference_category() -> None:
    """One toggle for "decisions about my path change", not three."""
    assert notify._CATEGORY == "path_change_review"


def test_the_notifiers_are_passed_the_callers_session() -> None:
    """``send_notification`` does not commit; the rows join the review's own
    transaction, so a rolled-back decision takes its notification with it.
    """
    import inspect

    for name in notify.__all__:
        params = list(inspect.signature(getattr(notify, name)).parameters)
        assert params[0] == "db", f"{name} must take the caller's session first"


def test_arq_pool_is_threaded_through_for_the_email_channel() -> None:
    """Without it the email-channel job is never enqueued and the
    notification stays in-app only."""
    import inspect

    for name in notify.__all__:
        params = inspect.signature(getattr(notify, name)).parameters
        assert "arq_pool" in params
        assert params["arq_pool"].default is None
