"""What the learning-program endpoints do besides calling a service.

Almost every endpoint here is three lines -- call, commit, map the error --
and the interesting part is that the three lines have to be *the same three
lines* twenty times over. Two of them fail silently when they are not:

**The commit.** These endpoints own their transaction. An endpoint that
returns the service's result without committing answers ``200`` with a
populated body and then discards the write when the session closes. The
manager sees the student enrolled, the roster does not. Nothing logs.

**The error map.** An endpoint that forgets to catch turns a refusal the
service worded carefully into a ``500``, which the SPA renders as "something
went wrong" -- so "you have used all three path switches this year" becomes
an unexplained failure.

Both are checked here as a table across every endpoint rather than one test
per handler, because the risk is a *new* endpoint being added without them.

The third theme is narrower and worse: the learner endpoints take the
student from the access token and never from the request body. A handler
that read an id out of the payload would let any student act on another
student's enrolment, and the service below has no way to tell.

The error-map helper itself and the CSV reader are covered in
``test_router_csv_and_errors``; the services are mocked here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.features.learning_programs import routers

_PROGRAM = uuid4()
_REQUEST = uuid4()
_ENROLMENT = uuid4()
_STUDENT = uuid4()
_ORG = uuid4()


def _actor() -> SimpleNamespace:
    return SimpleNamespace(user_id=uuid4())


def _db() -> SimpleNamespace:
    return SimpleNamespace(commit=AsyncMock())


def _payload(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


# (label, endpoint, service it delegates to, positional args after db/actor).
# ``args`` is a callable so each case gets fresh objects.
_WRITES: list[tuple[str, str, str, Any]] = [
    ("create", "create_program", "create_program", lambda a, d: (_payload(), a, d)),
    (
        "update",
        "update_program",
        "update_program",
        lambda a, d: (_PROGRAM, _payload(), a, d),
    ),
    ("publish", "publish_program", "publish_program", lambda a, d: (_PROGRAM, a, d)),
    ("archive", "archive_program", "archive_program", lambda a, d: (_PROGRAM, a, d)),
    (
        "enroll",
        "enroll_students",
        "enroll_students",
        lambda a, d: (_PROGRAM, _payload(student_ids=[_STUDENT]), a, d),
    ),
    (
        "withdraw",
        "withdraw_student",
        "withdraw_student",
        lambda a, d: (_PROGRAM, _STUDENT, _payload(reason="left the faculty"), a, d),
    ),
    (
        "approve",
        "approve_change_request",
        "decide_change_request",
        lambda a, d: (_REQUEST, _payload(reason="ok", note=None), a, d),
    ),
    (
        "in-progress",
        "mark_change_request_in_progress",
        "mark_change_request_in_progress",
        lambda a, d: (_REQUEST, a, d),
    ),
    (
        "reject",
        "reject_change_request",
        "decide_change_request",
        lambda a, d: (
            _REQUEST,
            _payload(reason="no", reason_code="capacity", note=None),
            a,
            d,
        ),
    ),
    (
        "select path",
        "select_path",
        "select_path",
        lambda a, d: (_ENROLMENT, _payload(career_path_id=uuid4()), a, d),
    ),
    (
        "request change",
        "request_path_change",
        "request_path_change",
        lambda a, d: (
            _ENROLMENT,
            _payload(target_career_path_id=uuid4(), from_attempt_id=None, reason="r"),
            a,
            d,
        ),
    ),
    (
        "request drop",
        "request_path_drop",
        "request_path_drop",
        lambda a, d: (
            _ENROLMENT,
            _payload(from_attempt_id=uuid4(), reason="too heavy"),
            a,
            d,
        ),
    ),
    (
        "cancel",
        "cancel_change_request",
        "cancel_change_request",
        lambda a, d: (_REQUEST, a, d),
    ),
]

_READS: list[tuple[str, str, str, Any]] = [
    ("options", "get_authoring_options", "get_authoring_options", lambda a, d: (a, d)),
    ("get program", "get_program", "get_program_for_operator", lambda a, d: (_PROGRAM, a, d)),
    (
        "list versions",
        "list_program_versions",
        "list_program_versions",
        lambda a, d: (_PROGRAM, a, d),
    ),
    (
        "get version",
        "get_program_version",
        "get_program_version",
        lambda a, d: (_PROGRAM, uuid4(), a, d),
    ),
    ("roster", "list_roster", "list_roster", lambda a, d: (_PROGRAM, a, d)),
    (
        "change requests",
        "list_change_requests",
        "list_change_requests",
        lambda a, d: (_PROGRAM, a, d),
    ),
    ("my programs", "list_my_programs", "list_my_enrollments", lambda a, d: (a, d)),
]


class TestEveryWriteCommitsItsOwnWork:
    """The endpoint owns the transaction; the service never commits.

    That split is deliberate -- it lets a service compose into another one
    without half-committing -- but it means the commit is a line a new
    endpoint can simply be missing, with no test failing and no log line.
    """

    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _WRITES)
    async def test_a_successful_write_is_committed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        monkeypatch.setattr(routers.services, service, AsyncMock(return_value="saved"))
        actor, db = _actor(), _db()

        result = await getattr(routers, endpoint)(*args(actor, db))

        assert result == "saved", label
        db.commit.assert_awaited_once_with()

    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _WRITES)
    async def test_a_refused_write_is_not_committed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        """Committing after a refusal is the dangerous half.

        The service raises partway through, so the session holds whatever it
        had already written -- committing that would persist a fragment of a
        rejected operation.
        """
        monkeypatch.setattr(
            routers.services, service, AsyncMock(side_effect=ConflictError("nope"))
        )
        actor, db = _actor(), _db()

        with pytest.raises(HTTPException):
            await getattr(routers, endpoint)(*args(actor, db))

        assert not db.commit.await_count, label


class TestEveryWriteTranslatesTheServicesRefusals:
    """Each of the three has a distinct meaning to the client.

    Letting one escape produces a 500, which the SPA shows as a generic
    failure -- so a sentence the service wrote for the user is replaced by
    "something went wrong".
    """

    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _WRITES)
    async def test_a_missing_row_is_a_404(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        monkeypatch.setattr(
            routers.services, service, AsyncMock(side_effect=NotFoundError("gone"))
        )

        with pytest.raises(HTTPException) as raised:
            await getattr(routers, endpoint)(*args(_actor(), _db()))

        assert raised.value.status_code == 404, label

    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _WRITES)
    async def test_a_refusal_is_a_403(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        monkeypatch.setattr(
            routers.services, service, AsyncMock(side_effect=ForbiddenError("not yours"))
        )

        with pytest.raises(HTTPException) as raised:
            await getattr(routers, endpoint)(*args(_actor(), _db()))

        assert raised.value.status_code == 403, label

    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _WRITES)
    async def test_a_rule_violation_is_a_409(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        monkeypatch.setattr(
            routers.services,
            service,
            AsyncMock(side_effect=ConflictError("switch_budget_exhausted")),
        )

        with pytest.raises(HTTPException) as raised:
            await getattr(routers, endpoint)(*args(_actor(), _db()))

        assert raised.value.status_code == 409, label
        assert raised.value.detail["error"] == "switch_budget_exhausted"


class TestReadsAreReads:
    @pytest.mark.parametrize(("label", "endpoint", "service", "args"), _READS)
    async def test_a_read_returns_the_services_answer_uncommitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        """A commit on a read path is not harmful so much as a sign the
        endpoint was copied from a write and not finished."""
        monkeypatch.setattr(routers.services, service, AsyncMock(return_value="answer"))
        actor, db = _actor(), _db()

        assert await getattr(routers, endpoint)(*args(actor, db)) == "answer", label

        assert not db.commit.await_count

    @pytest.mark.parametrize(
        ("label", "endpoint", "service", "args"),
        [case for case in _READS if case[0] not in {"options", "my programs"}],
    )
    async def test_a_read_of_something_absent_is_a_404(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        """The two excluded endpoints have nothing to look up: one reports
        the caller's own options, the other their own enrolments, and an
        empty list is the right answer for both.
        """
        monkeypatch.setattr(
            routers.services, service, AsyncMock(side_effect=NotFoundError("gone"))
        )

        with pytest.raises(HTTPException) as raised:
            await getattr(routers, endpoint)(*args(_actor(), _db()))

        assert raised.value.status_code == 404, label

    @pytest.mark.parametrize(
        ("label", "endpoint", "service", "args"),
        [case for case in _READS if case[0] not in {"options", "my programs"}],
    )
    async def test_a_read_of_someone_elses_program_is_a_403(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        """The permission dependency grants ``learning_program.read`` across
        the deployment; which programs this operator may see is the
        service's call, and it needs to survive the trip out.
        """
        monkeypatch.setattr(
            routers.services, service, AsyncMock(side_effect=ForbiddenError("other faculty"))
        )

        with pytest.raises(HTTPException) as raised:
            await getattr(routers, endpoint)(*args(_actor(), _db()))

        assert raised.value.status_code == 403, label


class TestListingProgramsWithoutSayingWhich:
    """``organization_id`` is optional, and the fallback is the whole
    endpoint.

    The SPA calls this with no query string on first paint, so whatever this
    resolves to is the list a manager sees when they open the page.
    """

    async def test_an_explicit_organization_is_used_as_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lookup = AsyncMock()
        monkeypatch.setattr(routers.access_control_api, "get_user_primary_org", lookup)
        service = AsyncMock(return_value=["a program"])
        monkeypatch.setattr(routers.services, "list_programs", service)

        result = await routers.list_programs(_actor(), _db(), organization_id=_ORG)

        assert result == ["a program"]
        assert service.await_args.kwargs["organization_id"] == _ORG
        lookup.assert_not_awaited()

    async def test_no_organization_falls_back_to_the_callers_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            routers.access_control_api,
            "get_user_primary_org",
            AsyncMock(return_value=SimpleNamespace(id=_ORG)),
        )
        service = AsyncMock(return_value=[])
        monkeypatch.setattr(routers.services, "list_programs", service)

        await routers.list_programs(_actor(), _db())

        assert service.await_args.kwargs["organization_id"] == _ORG

    async def test_the_fallback_looks_the_organization_up_by_the_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not by anything the caller sent. This is the only place the
        endpoint decides whose data to read, so the input has to be the
        authenticated identity.
        """
        lookup = AsyncMock(return_value=SimpleNamespace(id=_ORG))
        monkeypatch.setattr(routers.access_control_api, "get_user_primary_org", lookup)
        monkeypatch.setattr(routers.services, "list_programs", AsyncMock(return_value=[]))
        actor = _actor()

        await routers.list_programs(actor, _db())

        assert lookup.await_args.args[1] == actor.user_id

    async def test_a_caller_in_no_organization_gets_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not an error, and emphatically not every program in the
        deployment.

        A newly provisioned account with the permission but no org
        membership is the case this exists for; listing with
        ``organization_id=None`` would be an unscoped query.
        """
        monkeypatch.setattr(
            routers.access_control_api, "get_user_primary_org", AsyncMock(return_value=None)
        )
        service = AsyncMock()
        monkeypatch.setattr(routers.services, "list_programs", service)

        assert await routers.list_programs(_actor(), _db()) == []

        service.assert_not_awaited()


class TestTheStudentIsTakenFromTheToken:
    """Every learner endpoint anchors on ``actor.user_id``.

    The enrolment and request ids in these URLs are not secret and not
    guessable-proof; ownership is checked in the service against the id the
    router passes down. If that id came from the body instead, the check
    would compare the payload against itself.
    """

    async def test_selecting_a_path_acts_as_the_authenticated_student(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, "select_path", service)
        actor = _actor()

        await routers.select_path(
            _ENROLMENT, _payload(career_path_id=uuid4()), actor, _db()
        )

        assert service.await_args.kwargs["student_id"] == actor.user_id

    async def test_requesting_a_change_acts_as_the_authenticated_student(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, "request_path_change", service)
        actor = _actor()

        await routers.request_path_change(
            _ENROLMENT,
            _payload(target_career_path_id=uuid4(), from_attempt_id=None, reason="r"),
            actor,
            _db(),
        )

        assert service.await_args.kwargs["student_id"] == actor.user_id

    async def test_requesting_a_drop_acts_as_the_authenticated_student(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, "request_path_drop", service)
        actor = _actor()

        await routers.request_path_drop(
            _ENROLMENT, _payload(from_attempt_id=uuid4(), reason="r"), actor, _db()
        )

        assert service.await_args.kwargs["student_id"] == actor.user_id

    async def test_cancelling_acts_as_the_authenticated_student(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request id is all this endpoint receives, so without the token
        the service could not tell whose request it is being asked to
        withdraw.
        """
        service = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, "cancel_change_request", service)
        actor = _actor()

        await routers.cancel_change_request(_REQUEST, actor, _db())

        assert service.await_args.kwargs["student_id"] == actor.user_id

    async def test_listing_my_programs_reads_only_the_callers_enrolments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = AsyncMock(return_value=[])
        monkeypatch.setattr(routers.services, "list_my_enrollments", service)
        actor = _actor()

        await routers.list_my_programs(actor, _db())

        assert service.await_args.args[1] == actor.user_id


class TestDecidingARequest:
    """Approve and reject share one service call and differ by a flag.

    Sharing the call is what keeps the two decisions symmetrical -- same
    notification, same transition, same audit row -- but it also means the
    flag is the entire difference, and a wrong one is a rejection recorded
    as an approval.
    """

    async def test_approving_sets_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = AsyncMock(return_value="decided")
        monkeypatch.setattr(routers.services, "decide_change_request", service)

        await routers.approve_change_request(
            _REQUEST, _payload(reason="looks fine", note="n"), _actor(), _db()
        )

        assert service.await_args.kwargs["approve"] is True

    async def test_rejecting_clears_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = AsyncMock(return_value="decided")
        monkeypatch.setattr(routers.services, "decide_change_request", service)

        await routers.reject_change_request(
            _REQUEST,
            _payload(reason="capacity", reason_code="no_capacity", note="n"),
            _actor(),
            _db(),
        )

        assert service.await_args.kwargs["approve"] is False

    async def test_only_a_rejection_carries_a_reason_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An approval needs no justification; a rejection is shown to the
        student and filtered in reporting, so its code is mandatory. The
        approve path does not send the argument at all rather than sending
        ``None``, which is what keeps the service's own required-ness check
        meaningful.
        """
        service = AsyncMock(return_value="decided")
        monkeypatch.setattr(routers.services, "decide_change_request", service)

        await routers.approve_change_request(
            _REQUEST, _payload(reason="fine", note=None), _actor(), _db()
        )
        approving = service.await_args.kwargs

        await routers.reject_change_request(
            _REQUEST,
            _payload(reason="capacity", reason_code="no_capacity", note=None),
            _actor(),
            _db(),
        )
        rejecting = service.await_args.kwargs

        assert "decision_reason_code" not in approving
        assert rejecting["decision_reason_code"] == "no_capacity"

    async def test_the_deans_note_and_reason_are_both_passed_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = AsyncMock(return_value="decided")
        monkeypatch.setattr(routers.services, "decide_change_request", service)

        await routers.approve_change_request(
            _REQUEST,
            _payload(reason="meets the criteria", note="spoke to them"),
            _actor(),
            _db(),
        )

        assert service.await_args.kwargs["decision_reason"] == "meets the criteria"
        assert service.await_args.kwargs["decision_note"] == "spoke to them"

    async def test_acknowledging_decides_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acknowledging is not a verdict.

        It reaches a different service entirely, so it must not be able to
        move the request to approved or rejected.
        """
        decide = AsyncMock()
        monkeypatch.setattr(routers.services, "decide_change_request", decide)
        monkeypatch.setattr(
            routers.services, "mark_change_request_in_progress", AsyncMock(return_value="ack")
        )

        result = await routers.mark_change_request_in_progress(_REQUEST, _actor(), _db())

        assert result == "ack"

        decide.assert_not_awaited()


class TestTheEmailDispatchHandle:
    """``arq_pool`` rides along on every endpoint that notifies somebody.

    It is ``None`` until the app factory overrides the dependency, and the
    notification path accepts ``None`` by writing the in-app row without
    enqueuing email. So a missing pool costs the email, never the decision.
    """

    async def test_an_unconfigured_deployment_has_no_pool(self) -> None:
        assert await routers.get_arq_pool() is None

    @pytest.mark.parametrize(
        ("label", "endpoint", "service", "args"),
        [case for case in _WRITES if case[0] in {"approve", "reject", "in-progress"}],
    )
    async def test_the_pool_reaches_the_review_endpoints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        mock = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, service, mock)
        pool = object()

        await getattr(routers, endpoint)(*args(_actor(), _db()), pool)

        assert mock.await_args.kwargs["arq_pool"] is pool, label

    @pytest.mark.parametrize(
        ("label", "endpoint", "service", "args"),
        [case for case in _WRITES if case[0] in {"request change", "request drop"}],
    )
    async def test_the_pool_reaches_the_student_request_endpoints(
        self,
        monkeypatch: pytest.MonkeyPatch,
        label: str,
        endpoint: str,
        service: str,
        args: Any,
    ) -> None:
        """These are what put a request in the dean's queue, so the email
        they trigger is the one that makes anyone look at it."""
        mock = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, service, mock)
        pool = object()

        await getattr(routers, endpoint)(*args(_actor(), _db()), pool)

        assert mock.await_args.kwargs["arq_pool"] is pool, label

    async def test_a_missing_pool_is_passed_through_rather_than_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The service branches on it. Omitting the argument would make a
        deployment without a worker take a different code path from one
        whose pool is simply absent.
        """
        service = AsyncMock(return_value="ok")
        monkeypatch.setattr(routers.services, "decide_change_request", service)

        await routers.approve_change_request(
            _REQUEST, _payload(reason="r", note=None), _actor(), _db()
        )

        assert service.await_args.kwargs["arq_pool"] is None
