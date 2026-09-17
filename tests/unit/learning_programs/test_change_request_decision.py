"""A Faculty Dean deciding a student's path-change request.

This is the one function in the feature that both moves a student between
career paths and tells them why when it will not. Its own docstring calls
approval "one atomic invariant set", and the guards are not
interchangeable: some protect the student, some protect the dean, and one
protects the decision from the time that passed while it sat in a queue.

The rejection half is where most of the care is, because a rejection is
the outcome the student has to act on. A fixed reason code keeps the
vocabulary consistent across deans; ``other`` is the escape hatch, and it
has to carry words, or the student receives a rejection whose stated reason
is literally "other" and re-files the same request.

Approval is guarded against three things that can change between filing
and deciding: the student leaving the program, the student switching paths
by another route, and the target path being archived out from under the
request.

Everything below the service is mocked. The happy-path switch mechanics
(entitlement transfer, exit snapshots, switch accounting) are covered by
``tests/integration/test_learning_programs.py``; what is here is the
decision logic that sits in front of them.
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


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """An open ``change`` request on an active enrolment, ready to decide."""
    student_id, dean_id = uuid4(), uuid4()
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=student_id,
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="active",
    )
    attempt = SimpleNamespace(
        id=uuid4(),
        program_enrollment_id=enrollment.id,
        career_path_id=uuid4(),
        status="active",
    )
    request = SimpleNamespace(
        id=uuid4(),
        program_enrollment_id=enrollment.id,
        from_attempt_id=attempt.id,
        target_career_path_id=uuid4(),
        target_career_path_version_id=uuid4(),
        kind="change",
        status="pending",
        reason="The data track fits my internship.",
        reviewed_by=None,
        reviewed_at=None,
        decision_reason=None,
        decision_reason_code=None,
        decision_note=None,
        new_attempt_id=None,
        updated_by=None,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    program = SimpleNamespace(id=enrollment.learning_program_id, name="BSc Software Engineering")

    monkeypatch.setattr(services, "_require_owner_dean", AsyncMock())
    monkeypatch.setattr(services, "_subject_path_name", AsyncMock(return_value="Data Engineering"))
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services.notify, "notify_path_change_rejected", AsyncMock())
    monkeypatch.setattr(services.notify, "notify_path_change_approved", AsyncMock())

    monkeypatch.setattr(
        services.queries, "get_change_request", AsyncMock(return_value=request)
    )
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(services.queries, "get_attempt", AsyncMock(return_value=attempt))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[attempt]))

    return {
        "db": SimpleNamespace(add=lambda _row: None),
        "request": request,
        "enrollment": enrollment,
        "attempt": attempt,
        "program": program,
        "student_id": student_id,
        "actor": SimpleNamespace(user_id=dean_id),
    }


async def _decide(world: dict[str, Any], **overrides: Any):
    kwargs: dict[str, Any] = {
        "request_id": world["request"].id,
        "approve": False,
        "decision_reason": None,
        "decision_reason_code": "advising_required",
        "actor": world["actor"],
    }
    kwargs.update(overrides)
    return await services.decide_change_request(world["db"], **kwargs)


class TestWhoMayDecide:
    async def test_a_dean_cannot_decide_their_own_request(
        self, world: dict[str, Any]
    ) -> None:
        """A dean can also be enrolled as a student somewhere.

        Ownership of the program is not permission to approve your own
        path change; without this, the one person who can grant the request
        and the one person who benefits are the same.
        """
        world["actor"].user_id = world["student_id"]

        with pytest.raises(ForbiddenError, match="self_approval_is_not_allowed"):
            await _decide(world, approve=True)

    async def test_the_self_check_applies_to_rejection_too(
        self, world: dict[str, Any]
    ) -> None:
        """Rejecting your own request is less obviously harmful and still
        not a decision you get to make about yourself."""
        world["actor"].user_id = world["student_id"]

        with pytest.raises(ForbiddenError, match="self_approval_is_not_allowed"):
            await _decide(world)

    async def test_only_an_owning_dean_may_decide(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services,
            "_require_owner_dean",
            AsyncMock(side_effect=ForbiddenError("not_an_owning_dean")),
        )

        with pytest.raises(ForbiddenError, match="not_an_owning_dean"):
            await _decide(world)


class TestWhatCanBeDecided:
    @pytest.mark.parametrize("missing", ["request", "enrollment", "program"])
    async def test_a_missing_link_in_the_chain_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any], missing: str
    ) -> None:
        getter = {
            "request": "get_change_request",
            "enrollment": "get_enrollment",
            "program": "get_program",
        }[missing]
        monkeypatch.setattr(services.queries, getter, AsyncMock(return_value=None))

        with pytest.raises(NotFoundError):
            await _decide(world)

    @pytest.mark.parametrize("status", ["pending", "in_progress"])
    async def test_both_open_statuses_are_decidable(
        self, world: dict[str, Any], status: str
    ) -> None:
        """A dean can decide straight from the queue, or acknowledge first
        and decide after checking the record. Acknowledging must not spend
        the request."""
        world["request"].status = status

        result = await _decide(world)

        assert result.status == "rejected"

    @pytest.mark.parametrize("status", ["approved", "rejected", "cancelled", "invalidated"])
    async def test_a_terminal_request_cannot_be_decided_twice(
        self, world: dict[str, Any], status: str
    ) -> None:
        """Two deans can have the queue open at once. The second one's
        decision must not overwrite a record the student has already been
        told about.
        """
        world["request"].status = status

        with pytest.raises(ConflictError, match="request_is_not_open"):
            await _decide(world)


class TestRejectingWithAReason:
    async def test_a_rejection_must_carry_a_reason_code(
        self, world: dict[str, Any]
    ) -> None:
        """A rejection with no reason is what makes a student re-file the
        same request, and the dean re-read the same record."""
        with pytest.raises(ConflictError, match="rejection_reason_code_is_required"):
            await _decide(world, decision_reason_code=None)

    async def test_the_code_must_come_from_the_fixed_vocabulary(
        self, world: dict[str, Any]
    ) -> None:
        """The codes are rendered as localized copy, so an unknown one would
        reach the student as a missing translation key."""
        with pytest.raises(ConflictError, match="unknown_rejection_reason_code"):
            await _decide(world, decision_reason_code="because_i_said_so")

    async def test_other_must_carry_the_words_the_list_could_not_express(
        self, world: dict[str, Any]
    ) -> None:
        """``other`` is the escape hatch from the fixed list.

        Allowing it empty would send the student a rejection whose stated
        reason is the word "other", which is worse than the fixed codes it
        was added to escape.
        """
        with pytest.raises(
            ConflictError, match="rejection_reason_is_required_when_code_is_other"
        ):
            await _decide(world, decision_reason_code="other", decision_reason=None)

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    async def test_whitespace_does_not_count_as_a_reason(
        self, world: dict[str, Any], blank: str
    ) -> None:
        with pytest.raises(
            ConflictError, match="rejection_reason_is_required_when_code_is_other"
        ):
            await _decide(world, decision_reason_code="other", decision_reason=blank)

    async def test_an_other_rejection_keeps_the_deans_words_as_the_reason(
        self, world: dict[str, Any]
    ) -> None:
        await _decide(
            world,
            decision_reason_code="other",
            decision_reason="  Capacity reached for this intake  ",
            decision_note="Try again next semester.",
        )

        request = world["request"]
        assert request.decision_reason_code == "other"
        assert request.decision_reason == "Capacity reached for this intake", "trimmed"
        assert request.decision_note == "Try again next semester."

    async def test_a_fixed_code_rejection_stores_no_free_text_reason(
        self, world: dict[str, Any]
    ) -> None:
        """With a code from the list, the reason *is* the code.

        Keeping a second free-text copy beside it would give the SPA two
        things to render and no rule for which wins.
        """
        await _decide(
            world,
            decision_reason_code="progress_loss_too_high",
            decision_reason="You would lose two completed stages.",
        )

        request = world["request"]
        assert request.decision_reason_code == "progress_loss_too_high"
        assert request.decision_reason is None

    async def test_free_text_on_a_fixed_code_becomes_the_note(
        self, world: dict[str, Any]
    ) -> None:
        """The dean typed something useful into the wrong box.

        Discarding it would lose the only part of the rejection written for
        this particular student, so it is moved to the note rather than
        dropped.
        """
        await _decide(
            world,
            decision_reason_code="advising_required",
            decision_reason="Speak to your advisor in week 3.",
        )

        assert world["request"].decision_note == "Speak to your advisor in week 3."

    async def test_an_explicit_note_is_not_overwritten_by_the_reason_box(
        self, world: dict[str, Any]
    ) -> None:
        """When the dean filled in both, the note is what they meant as the
        note."""
        await _decide(
            world,
            decision_reason_code="advising_required",
            decision_reason="ignored free text",
            decision_note="See me in office hours.",
        )

        assert world["request"].decision_note == "See me in office hours."

    async def test_the_rejection_is_attributed_and_timestamped(
        self, world: dict[str, Any]
    ) -> None:
        """A decision on a student's record has to say who made it."""
        await _decide(world)

        request = world["request"]
        assert request.status == "rejected"
        assert request.reviewed_by == world["actor"].user_id
        assert request.reviewed_at is not None
        assert request.updated_by == world["actor"].user_id

    async def test_the_student_is_told_with_the_reason_that_was_stored(
        self, world: dict[str, Any]
    ) -> None:
        """The notification and the record must agree; a student comparing
        their inbox with the request page should not see two accounts of
        the same decision.
        """
        await _decide(
            world,
            decision_reason_code="other",
            decision_reason="Capacity reached",
            decision_note="Try next intake.",
        )

        call = services.notify.notify_path_change_rejected.await_args.kwargs
        assert call["student_user_id"] == world["student_id"]
        assert call["request_id"] == world["request"].id
        assert call["reason_code"] == "other"
        assert call["reason_detail"] == "Capacity reached"
        assert call["note"] == "Try next intake."
        assert call["program_name"] == "BSc Software Engineering"
        assert call["kind"] == "change"

    async def test_a_drop_request_is_rejected_as_a_drop(
        self, world: dict[str, Any]
    ) -> None:
        """``kind`` reaches the copy builder, so a student who asked to
        leave is not told they stay on their current path."""
        world["request"].kind = "drop"

        await _decide(world)

        assert services.notify.notify_path_change_rejected.await_args.kwargs["kind"] == "drop"

    async def test_rejecting_does_not_require_an_active_enrolment(
        self, world: dict[str, Any]
    ) -> None:
        """A student who withdrew while their request sat in the queue can
        still be told the answer; only approval needs somewhere to apply
        the change to.
        """
        world["enrollment"].status = "withdrawn"

        result = await _decide(world)

        assert result.status == "rejected"


class TestApprovingAgainstAMovingTarget:
    """A request is decided minutes or days after it was filed, and the
    student is still using the system in between.
    """

    async def test_a_student_who_left_the_program_cannot_be_switched(
        self, world: dict[str, Any]
    ) -> None:
        world["enrollment"].status = "withdrawn"

        with pytest.raises(ConflictError, match="program_is_not_active"):
            await _decide(world, approve=True)

    async def test_a_request_whose_attempt_ended_is_stale(
        self, world: dict[str, Any]
    ) -> None:
        """The student already left the path they were asking to leave.

        Approving would switch them out of something they are no longer on,
        which is why the check is against the attempt the request was filed
        against rather than against whatever is active now.
        """
        world["attempt"].status = "switched_out"

        with pytest.raises(ConflictError, match="active_path_changed_since_request"):
            await _decide(world, approve=True)

    async def test_a_request_whose_attempt_vanished_is_stale(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(services.queries, "get_attempt", AsyncMock(return_value=None))

        with pytest.raises(ConflictError, match="active_path_changed_since_request"):
            await _decide(world, approve=True)

    async def test_an_attempt_belonging_to_another_enrolment_is_refused(
        self, world: dict[str, Any]
    ) -> None:
        """The id is carried on the request, so it is worth confirming it
        still points inside this enrolment before acting on it."""
        world["attempt"].program_enrollment_id = uuid4()

        with pytest.raises(ConflictError, match="active_path_changed_since_request"):
            await _decide(world, approve=True)

    async def test_a_path_the_student_already_holds_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """They picked up the target path by another route while waiting.

        Approving would leave them on the same path twice, which the unique
        index would reject anyway -- but as a 500 rather than as a sentence
        explaining what happened.
        """
        other = SimpleNamespace(
            id=uuid4(),
            career_path_id=world["request"].target_career_path_id,
            status="active",
        )
        monkeypatch.setattr(
            services.queries,
            "list_attempts",
            AsyncMock(return_value=[world["attempt"], other]),
        )

        with pytest.raises(ConflictError, match="path_already_selected"):
            await _decide(world, approve=True)

    async def test_a_completed_attempt_on_the_target_path_also_blocks(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Finished still counts as held: switching onto a path they have
        already completed is not a change worth making."""
        done = SimpleNamespace(
            id=uuid4(),
            career_path_id=world["request"].target_career_path_id,
            status="completed",
        )
        monkeypatch.setattr(
            services.queries,
            "list_attempts",
            AsyncMock(return_value=[world["attempt"], done]),
        )

        with pytest.raises(ConflictError, match="path_already_selected"):
            await _decide(world, approve=True)

    async def test_an_abandoned_attempt_on_the_target_path_does_not_block(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Only active and completed attempts count as holding a path, so a
        student who dropped it earlier may be switched back onto it.
        """
        monkeypatch.setattr(
            services.queries,
            "get_version",
            AsyncMock(return_value=SimpleNamespace(max_path_switches=3)),
        )
        monkeypatch.setattr(services.queries, "count_approved_switches", AsyncMock(return_value=0))
        monkeypatch.setattr(services.queries, "list_version_paths", AsyncMock(return_value=[]))
        old = SimpleNamespace(
            id=uuid4(),
            career_path_id=world["request"].target_career_path_id,
            status="dropped",
        )
        monkeypatch.setattr(
            services.queries,
            "list_attempts",
            AsyncMock(return_value=[world["attempt"], old]),
        )

        result = await _decide(world, approve=True)

        assert result.status == "invalidated", (
            "it got past the duplicate guard and stopped at the archived-target "
            "check, which is the next one in the chain"
        )

    async def test_a_target_path_removed_from_the_version_invalidates_the_request(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Invalidated, not rejected. The dean did not refuse anything --
        the thing being asked for stopped existing while the request waited,
        so the record should not read as a decision against the student.
        """
        monkeypatch.setattr(
            services.queries,
            "get_version",
            AsyncMock(return_value=SimpleNamespace(max_path_switches=3)),
        )
        monkeypatch.setattr(services.queries, "count_approved_switches", AsyncMock(return_value=0))
        monkeypatch.setattr(services.queries, "list_version_paths", AsyncMock(return_value=[]))

        result = await _decide(world, approve=True)

        assert result.status == "invalidated"
        assert world["request"].decision_reason == "target_path_archived"
        services.notify.notify_path_change_approved.assert_not_awaited()

    async def test_an_archived_target_path_invalidates_too(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services.queries,
            "get_version",
            AsyncMock(return_value=SimpleNamespace(max_path_switches=3)),
        )
        monkeypatch.setattr(services.queries, "count_approved_switches", AsyncMock(return_value=0))
        monkeypatch.setattr(
            services.queries,
            "list_version_paths",
            AsyncMock(
                return_value=[
                    {
                        "career_path_id": world["request"].target_career_path_id,
                        "status": "archived",
                    }
                ]
            ),
        )

        result = await _decide(world, approve=True)

        assert result.status == "invalidated"

    async def test_the_switch_budget_is_enforced(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The version caps how many times a student may move.

        It is checked at approval rather than at filing, because the budget
        can be spent by another request between the two.
        """
        monkeypatch.setattr(
            services.queries,
            "get_version",
            AsyncMock(return_value=SimpleNamespace(max_path_switches=2)),
        )
        monkeypatch.setattr(services.queries, "count_approved_switches", AsyncMock(return_value=2))

        with pytest.raises(ConflictError, match="path_switch_limit_reached"):
            await _decide(world, approve=True)

    async def test_a_missing_pinned_version_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(services.queries, "get_version", AsyncMock(return_value=None))

        with pytest.raises(NotFoundError, match="program_version_not_found"):
            await _decide(world, approve=True)


class TestADropTakesItsOwnRoute:
    async def test_a_drop_approval_is_delegated(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Dropping is not a switch with no destination: there is no new
        attempt, no entitlement transfer, and the student must be left with
        at least one path. It gets its own routine rather than a pile of
        conditionals in this one.
        """
        world["request"].kind = "drop"
        drop = AsyncMock(return_value="dropped")
        monkeypatch.setattr(services, "_approve_path_drop", drop)

        result = await _decide(world, approve=True, decision_note="  ok  ")

        assert result == "dropped"
        kwargs = drop.await_args.kwargs
        assert kwargs["request"] is world["request"]
        assert kwargs["enrollment"] is world["enrollment"]
        assert kwargs["attempt"] is world["attempt"]
        assert kwargs["actor_id"] == world["actor"].user_id

    async def test_a_drop_still_passes_the_staleness_guards(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The guards run before the fork, so a drop cannot skip them."""
        world["request"].kind = "drop"
        world["attempt"].status = "switched_out"
        monkeypatch.setattr(services, "_approve_path_drop", AsyncMock())

        with pytest.raises(ConflictError, match="active_path_changed_since_request"):
            await _decide(world, approve=True)
