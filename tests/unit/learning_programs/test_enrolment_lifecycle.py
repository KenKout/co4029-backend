"""Leaving a program, and the two states a request can reach without a
decision.

Withdrawing a student is the largest cleanup in the feature: the enrolment
closes, every active path attempt is snapshotted and cancelled, the
entitlements those attempts granted are revoked, and any request still
waiting for a dean is cancelled with a reason that says why. Missing any
one of those leaves the student holding access to content they are no
longer enrolled in, or a request in a queue nobody can act on.

The subtlest part is the access release. A student can be on the same
career path through two programs at once, so leaving one of them must not
revoke access the other still grants -- the release only fires when no
other active attempt holds that path.

Acknowledging and cancelling are the two non-decisions. Acknowledging is
idempotent, and deliberately notifies only on the edge: two deans opening
the same queue is normal, and the student should not receive the same
"someone is looking at this" message twice.

Queries, notifications and the career-paths API are mocked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.features.learning_programs import services


def _attempt(path_id: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        career_path_id=path_id or uuid4(),
        status="active",
        exit_snapshot=None,
        ended_at=None,
        updated_by=None,
    )


@pytest.fixture
def withdrawal(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    program = SimpleNamespace(id=uuid4(), organization_id=uuid4(), name="BSc SE")
    enrollment = SimpleNamespace(
        id=uuid4(),
        learning_program_id=program.id,
        status="active",
        withdrawn_at=None,
        withdrawal_reason=None,
        updated_by=None,
    )

    monkeypatch.setattr(services, "_require_operator", AsyncMock())
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_enrollment_out", AsyncMock(return_value="closed"))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(
        services.queries, "get_program_enrollment", AsyncMock(return_value=enrollment)
    )
    monkeypatch.setattr(services.queries, "list_active_attempts", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        services.queries, "build_exit_snapshot", AsyncMock(return_value={"percent": 40})
    )
    monkeypatch.setattr(services.queries, "revoke_path_entitlements", AsyncMock())
    monkeypatch.setattr(
        services.queries, "count_other_active_path_attempts", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(services.queries, "get_pending_request", AsyncMock(return_value=None))
    monkeypatch.setattr(services.career_paths_api, "release_program_path_access", AsyncMock())

    return {
        "db": SimpleNamespace(),
        "program": program,
        "enrollment": enrollment,
        "student_id": uuid4(),
        "actor": SimpleNamespace(user_id=uuid4()),
    }


async def _withdraw(world: dict[str, Any], reason: str = "left the faculty"):
    return await services.withdraw_student(
        world["db"],
        program_id=world["program"].id,
        student_id=world["student_id"],
        reason=reason,
        actor=world["actor"],
    )


class TestWithdrawingAStudent:
    async def test_an_unknown_program_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=None))
        with pytest.raises(NotFoundError, match="learning_program_not_found"):
            await _withdraw(withdrawal)

    async def test_a_student_who_is_not_enrolled_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services.queries, "get_program_enrollment", AsyncMock(return_value=None)
        )
        with pytest.raises(NotFoundError, match="program_enrollment_not_found"):
            await _withdraw(withdrawal)

    async def test_a_completed_program_cannot_be_withdrawn_from(
        self, withdrawal: dict[str, Any]
    ) -> None:
        """Completion is an academic record. Withdrawing afterwards would
        erase an award the student earned rather than ending something in
        progress -- if it was granted in error, that is a correction, not a
        withdrawal.
        """
        withdrawal["enrollment"].status = "completed"

        with pytest.raises(ConflictError, match="completed_program_cannot_be_withdrawn"):
            await _withdraw(withdrawal)

    async def test_the_reason_is_recorded_with_the_withdrawal(
        self, withdrawal: dict[str, Any]
    ) -> None:
        """A withdrawn row with no reason leaves the next person reading the
        record unable to tell an administrative cleanup from a student who
        left.
        """
        await _withdraw(withdrawal, reason="transferred to another faculty")

        enrollment = withdrawal["enrollment"]
        assert enrollment.status == "withdrawn"
        assert enrollment.withdrawal_reason == "transferred to another faculty"
        assert enrollment.withdrawn_at is not None
        assert enrollment.updated_by == withdrawal["actor"].user_id

    async def test_every_active_attempt_is_snapshotted_before_it_is_closed(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        """The snapshot is the only record of how far they got.

        Progress is derived from live rows, and those stop being reachable
        once the attempt is cancelled, so capturing it afterwards would
        capture nothing.
        """
        first, second = _attempt(), _attempt()
        monkeypatch.setattr(
            services.queries, "list_active_attempts", AsyncMock(return_value=[first, second])
        )

        await _withdraw(withdrawal)

        for attempt in (first, second):
            assert attempt.exit_snapshot == {"percent": 40}
            assert attempt.status == "cancelled"
            assert attempt.ended_at is not None

    async def test_entitlements_are_revoked_for_each_attempt(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        """Left in place, the student keeps reaching course content through
        a program they are no longer on."""
        attempt = _attempt()
        revoke = AsyncMock()
        monkeypatch.setattr(
            services.queries, "list_active_attempts", AsyncMock(return_value=[attempt])
        )
        monkeypatch.setattr(services.queries, "revoke_path_entitlements", revoke)

        await _withdraw(withdrawal)

        revoke.assert_awaited_once()
        assert revoke.await_args.kwargs["attempt_id"] == attempt.id

    async def test_path_access_is_released_when_nothing_else_grants_it(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        attempt = _attempt()
        release = AsyncMock()
        monkeypatch.setattr(
            services.queries, "list_active_attempts", AsyncMock(return_value=[attempt])
        )
        monkeypatch.setattr(
            services.queries, "count_other_active_path_attempts", AsyncMock(return_value=0)
        )
        monkeypatch.setattr(services.career_paths_api, "release_program_path_access", release)

        await _withdraw(withdrawal)

        release.assert_awaited_once()
        assert release.await_args.kwargs["career_path_id"] == attempt.career_path_id

    async def test_path_access_survives_when_another_programme_still_grants_it(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        """The reason the release is conditional rather than automatic.

        A student can reach the same career path through two programs.
        Leaving one of them must not revoke the access the other still
        gives them, which would lock them out of a path they are actively
        walking.
        """
        attempt = _attempt()
        release = AsyncMock()
        monkeypatch.setattr(
            services.queries, "list_active_attempts", AsyncMock(return_value=[attempt])
        )
        monkeypatch.setattr(
            services.queries, "count_other_active_path_attempts", AsyncMock(return_value=1)
        )
        monkeypatch.setattr(services.career_paths_api, "release_program_path_access", release)

        await _withdraw(withdrawal)

        release.assert_not_awaited()

    async def test_the_other_attempt_check_excludes_the_one_being_closed(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        """Counting itself would make the answer always "something else
        holds it", and access would never be released for anyone."""
        attempt = _attempt()
        count = AsyncMock(return_value=0)
        monkeypatch.setattr(
            services.queries, "list_active_attempts", AsyncMock(return_value=[attempt])
        )
        monkeypatch.setattr(services.queries, "count_other_active_path_attempts", count)

        await _withdraw(withdrawal)

        assert count.await_args.kwargs["excluding_attempt_id"] == attempt.id

    async def test_a_waiting_request_is_cancelled_with_a_reason(
        self, monkeypatch: pytest.MonkeyPatch, withdrawal: dict[str, Any]
    ) -> None:
        """Left open, it sits in a dean's queue against an enrolment that no
        longer exists -- undecidable, and impossible to explain.
        """
        pending = SimpleNamespace(
            status="pending", reviewed_at=None, decision_reason=None, updated_by=None
        )
        monkeypatch.setattr(
            services.queries, "get_pending_request", AsyncMock(return_value=pending)
        )

        await _withdraw(withdrawal)

        assert pending.status == "cancelled"
        assert pending.decision_reason == "program_enrollment_withdrawn", (
            "the reason distinguishes this from a student cancelling their own"
        )
        assert pending.reviewed_at is not None

    async def test_a_withdrawal_with_nothing_outstanding_is_still_clean(
        self, withdrawal: dict[str, Any]
    ) -> None:
        """No attempts, no request: the common case for someone who was
        enrolled and never started."""
        result = await _withdraw(withdrawal)

        assert result == "closed"
        assert withdrawal["enrollment"].status == "withdrawn"


@pytest.fixture
def request_world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    student_id = uuid4()
    enrollment = SimpleNamespace(
        id=uuid4(), student_id=student_id, learning_program_id=uuid4()
    )
    request = SimpleNamespace(
        id=uuid4(),
        program_enrollment_id=enrollment.id,
        from_attempt_id=uuid4(),
        target_career_path_id=uuid4(),
        target_career_path_version_id=uuid4(),
        kind="change",
        status="pending",
        reason="r",
        reviewed_by=None,
        reviewed_at=None,
        decision_reason=None,
        decision_reason_code=None,
        decision_note=None,
        new_attempt_id=None,
        in_progress_at=None,
        in_progress_by=None,
        updated_by=None,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    program = SimpleNamespace(id=enrollment.learning_program_id, name="BSc SE")

    monkeypatch.setattr(services, "_require_owner_dean", AsyncMock())
    monkeypatch.setattr(services, "_subject_path_name", AsyncMock(return_value="Data Eng"))
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services.notify, "notify_path_change_in_progress", AsyncMock())
    monkeypatch.setattr(
        services.queries, "get_change_request", AsyncMock(return_value=request)
    )
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))

    return {
        "db": SimpleNamespace(),
        "request": request,
        "enrollment": enrollment,
        "student_id": student_id,
        "actor": SimpleNamespace(user_id=uuid4()),
    }


class TestAcknowledgingARequest:
    """Recording that a dean has picked the request up.

    Deliberately not a decision: nothing about the enrolment, the attempt or
    the switch budget moves, and every approval-time recheck still runs
    later. It exists to answer the question a silent queue provokes.
    """

    async def test_acknowledging_marks_and_attributes_it(
        self, request_world: dict[str, Any]
    ) -> None:
        await services.mark_change_request_in_progress(
            request_world["db"],
            request_id=request_world["request"].id,
            actor=request_world["actor"],
        )

        request = request_world["request"]
        assert request.status == "in_progress"
        assert request.in_progress_by == request_world["actor"].user_id
        assert request.in_progress_at is not None

    async def test_the_student_is_told_once_on_the_edge(
        self, request_world: dict[str, Any]
    ) -> None:
        await services.mark_change_request_in_progress(
            request_world["db"],
            request_id=request_world["request"].id,
            actor=request_world["actor"],
        )

        services.notify.notify_path_change_in_progress.assert_awaited_once()
        call = services.notify.notify_path_change_in_progress.await_args.kwargs
        assert call["student_user_id"] == request_world["student_id"]
        assert call["kind"] == "change"

    async def test_re_acknowledging_is_a_silent_no_op(
        self, request_world: dict[str, Any]
    ) -> None:
        """Two deans opening the same queue is normal.

        The second one must not see a 409, and the student must not receive
        a second "someone is looking at this" message for the same request.
        """
        request_world["request"].status = "in_progress"
        stamped = request_world["request"].in_progress_by

        result = await services.mark_change_request_in_progress(
            request_world["db"],
            request_id=request_world["request"].id,
            actor=request_world["actor"],
        )

        assert result.status == "in_progress"
        assert request_world["request"].in_progress_by == stamped, (
            "the first dean keeps the acknowledgement"
        )
        services.notify.notify_path_change_in_progress.assert_not_awaited()

    @pytest.mark.parametrize("status", ["approved", "rejected", "cancelled", "invalidated"])
    async def test_a_decided_request_cannot_be_acknowledged(
        self, request_world: dict[str, Any], status: str
    ) -> None:
        """Acknowledging after a decision would put a closed request back in
        the queue's "being looked at" state."""
        request_world["request"].status = status

        with pytest.raises(
            ConflictError, match="only_pending_requests_can_be_marked_in_progress"
        ):
            await services.mark_change_request_in_progress(
                request_world["db"],
                request_id=request_world["request"].id,
                actor=request_world["actor"],
            )

    async def test_a_dean_cannot_acknowledge_their_own_request(
        self, request_world: dict[str, Any]
    ) -> None:
        """The same self-dealing guard the decision path applies.

        Acknowledging is not a decision, but taking ownership of your own
        request is how it stops being visible to anyone who could refuse it.
        """
        request_world["actor"].user_id = request_world["student_id"]

        with pytest.raises(ForbiddenError, match="self_approval_is_not_allowed"):
            await services.mark_change_request_in_progress(
                request_world["db"],
                request_id=request_world["request"].id,
                actor=request_world["actor"],
            )

    @pytest.mark.parametrize("missing", ["request", "enrollment", "program"])
    async def test_a_broken_chain_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, request_world: dict[str, Any], missing: str
    ) -> None:
        getter = {
            "request": "get_change_request",
            "enrollment": "get_enrollment",
            "program": "get_program",
        }[missing]
        monkeypatch.setattr(services.queries, getter, AsyncMock(return_value=None))

        with pytest.raises(NotFoundError):
            await services.mark_change_request_in_progress(
                request_world["db"],
                request_id=request_world["request"].id,
                actor=request_world["actor"],
            )


class TestAStudentCancellingTheirOwnRequest:
    async def test_a_pending_request_can_be_withdrawn(
        self, request_world: dict[str, Any]
    ) -> None:
        result = await services.cancel_change_request(
            request_world["db"],
            request_id=request_world["request"].id,
            student_id=request_world["student_id"],
        )

        assert result.status == "cancelled"
        assert request_world["request"].updated_by == request_world["student_id"]
        assert request_world["request"].reviewed_at is not None

    async def test_an_acknowledged_request_can_still_be_withdrawn(
        self, request_world: dict[str, Any]
    ) -> None:
        """A student who changed their mind should not have to wait for a
        decision just because a dean opened the request -- and a rejection
        costs no switch budget, so there is nothing to game by cancelling
        late.
        """
        request_world["request"].status = "in_progress"

        result = await services.cancel_change_request(
            request_world["db"],
            request_id=request_world["request"].id,
            student_id=request_world["student_id"],
        )

        assert result.status == "cancelled"

    @pytest.mark.parametrize("status", ["approved", "rejected", "cancelled", "invalidated"])
    async def test_a_decided_request_cannot_be_withdrawn(
        self, request_world: dict[str, Any], status: str
    ) -> None:
        request_world["request"].status = status

        with pytest.raises(ConflictError, match="only_open_requests_can_be_cancelled"):
            await services.cancel_change_request(
                request_world["db"],
                request_id=request_world["request"].id,
                student_id=request_world["student_id"],
            )

    async def test_another_students_request_is_indistinguishable_from_a_missing_one(
        self, request_world: dict[str, Any]
    ) -> None:
        """Both answer not-found rather than forbidden.

        Splitting them would let anyone confirm which request ids exist by
        watching which error came back, and a request id leaks that a
        particular student asked to change path.
        """
        with pytest.raises(NotFoundError, match="path_change_request_not_found"):
            await services.cancel_change_request(
                request_world["db"],
                request_id=request_world["request"].id,
                student_id=uuid4(),
            )

    async def test_an_unknown_request_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, request_world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services.queries, "get_change_request", AsyncMock(return_value=None)
        )

        with pytest.raises(NotFoundError, match="path_change_request_not_found"):
            await services.cancel_change_request(
                request_world["db"], request_id=uuid4(), student_id=request_world["student_id"]
            )

    async def test_cancelling_does_not_require_a_dean(
        self, monkeypatch: pytest.MonkeyPatch, request_world: dict[str, Any]
    ) -> None:
        """It is the student's own request; requiring an owning dean would
        make a student unable to withdraw it."""
        guard = AsyncMock(side_effect=AssertionError("the dean guard must not run here"))
        monkeypatch.setattr(services, "_require_owner_dean", guard)

        await services.cancel_change_request(
            request_world["db"],
            request_id=request_world["request"].id,
            student_id=request_world["student_id"],
        )

        guard.assert_not_awaited()
