"""The manager's view of one person, and the invite that creates them.

``get_user_overview`` is pure assembly: it holds no query of its own and
composes five sibling features' public APIs into the page a manager opens
when they click a name. That makes it cheap to get subtly wrong, and two of
its decisions are the kind nobody notices until someone asks why the page
disagrees with itself.

**Dropped enrolments are kept.** They used to be skipped, which meant a
student who left a course simply vanished from this page and the manager
had no way to see it had ever happened. The row status already distinguishes
them, so showing the row costs nothing and restores the history.

**Last-active is the latest of everything, not the login.** A student who
signed in in September and has been working since would otherwise show a
September timestamp on a page whose whole purpose is telling a manager
whether to follow up.

``create_user_account`` is the admin invite, and it carries a privilege
guard worth more than the rest of the file: the ``admin`` role is
global-scope and holds every permission, so it cannot be granted through an
org-scoped invite.

Every cross-feature read is mocked; this module owns none of them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.features.identity.schemas import UserRead
from abridgeai.features.identity.services import admin as admin_service

_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _user_read(user_id: UUID, *, last_login_at: datetime | None = None) -> UserRead:
    return UserRead.model_validate(
        {
            "id": user_id,
            "primary_email": "student@test.local",
            "status": "active",
            "last_login_at": last_login_at,
            "created_at": _NOW,
            "updated_at": _NOW,
            "profile": None,
        }
    )


def _course(course_id: UUID, *, title: str = "Databases") -> SimpleNamespace:
    return SimpleNamespace(id=course_id, title=title, slug="databases", status="published")


def _enrollment(course_id: UUID, *, status: str = "active") -> SimpleNamespace:
    return SimpleNamespace(course_id=course_id, status=status, enrolled_at=_NOW)


def _progress(**over: Any) -> dict[str, Any]:
    base = {
        "completion_percent": 40.0,
        "completed_lessons": 2,
        "total_lessons": 5,
        "last_activity_at": None,
    }
    base.update(over)
    return base


@pytest.fixture
def overview(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A student whose sibling features report nothing until a test says so."""
    user_id = uuid4()

    monkeypatch.setattr(
        admin_service.user_queries, "get_user", AsyncMock(return_value=SimpleNamespace(id=user_id))
    )
    monkeypatch.setattr(admin_service.user_queries, "get_profile", AsyncMock(return_value=None))
    monkeypatch.setattr(
        admin_service.access_control_api,
        "get_role_codes_for_users",
        AsyncMock(return_value={user_id: ["student"]}),
    )
    monkeypatch.setattr(
        admin_service.access_control_api, "get_primary_orgs_for_users", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        admin_service.access_control_api,
        "get_user_membership_codes",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        admin_service, "_serialize_search_row", AsyncMock(return_value=_user_read(user_id))
    )

    import abridgeai.features.career_paths.api.public as career_paths_api
    import abridgeai.features.courses.api.public as courses_api
    import abridgeai.features.enrollments.api.public as enrollments_api
    import abridgeai.features.learning_programs.api.public as learning_programs_api
    import abridgeai.features.progress.api.public as progress_api

    monkeypatch.setattr(
        enrollments_api, "list_user_course_enrollments", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(courses_api, "get_course_by_id", AsyncMock(return_value=None))
    monkeypatch.setattr(courses_api, "list_courses_for_teacher", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        progress_api, "get_course_progress_for_user", AsyncMock(return_value=_progress())
    )
    monkeypatch.setattr(
        career_paths_api, "list_user_career_enrollments", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        career_paths_api, "get_path_course_progress_for_user", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        learning_programs_api, "list_student_program_enrollments", AsyncMock(return_value=[])
    )

    return {
        "db": SimpleNamespace(),
        "user_id": user_id,
        "courses_api": courses_api,
        "enrollments_api": enrollments_api,
        "progress_api": progress_api,
        "career_paths_api": career_paths_api,
        "learning_programs_api": learning_programs_api,
    }


class TestWhichSectionsAreAssembled:
    async def test_an_unknown_user_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            admin_service.user_queries, "get_user", AsyncMock(return_value=None)
        )
        with pytest.raises(NotFoundError):
            await admin_service.get_user_overview(overview["db"], user_id=uuid4())

    @pytest.mark.parametrize("role", ["manager", "hod", "admin"])
    async def test_a_non_learner_gets_identity_only(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any], role: str
    ) -> None:
        """A manager has no learning surface, and filling the page with
        empty "0 of 0 courses" panels would imply they should have one.
        """
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_role_codes_for_users",
            AsyncMock(return_value={overview["user_id"]: [role]}),
        )
        enrollments = AsyncMock(return_value=[])
        monkeypatch.setattr(
            overview["enrollments_api"], "list_user_course_enrollments", enrollments
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.courses == []
        assert result.assigned_courses == []
        enrollments.assert_not_awaited()

    async def test_a_teacher_gets_the_courses_they_teach(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        course_id = uuid4()
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_role_codes_for_users",
            AsyncMock(return_value={overview["user_id"]: ["teacher"]}),
        )
        monkeypatch.setattr(
            overview["courses_api"],
            "list_courses_for_teacher",
            AsyncMock(return_value=[_course(course_id, title="Compilers")]),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert [c.course_id for c in result.assigned_courses] == [course_id]
        assert result.assigned_courses[0].title == "Compilers"

    async def test_someone_who_is_both_gets_both_sections(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """A teacher enrolled on a course as a student is ordinary, and the
        page has to show both halves of what they do.
        """
        taught, learned = uuid4(), uuid4()
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_role_codes_for_users",
            AsyncMock(return_value={overview["user_id"]: ["student", "teacher"]}),
        )
        monkeypatch.setattr(
            overview["courses_api"],
            "list_courses_for_teacher",
            AsyncMock(return_value=[_course(taught)]),
        )
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(learned)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(learned))
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert [c.course_id for c in result.assigned_courses] == [taught]
        assert [c.course_id for c in result.courses] == [learned]

    async def test_membership_codes_are_folded_onto_the_identity(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """The student number is what a manager searches by and reads back
        to a registry, so it belongs on the detail page even though it is
        not part of the list projection."""
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_user_membership_codes",
            AsyncMock(
                return_value=SimpleNamespace(student_code="2210001", employee_code=None)
            ),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.user.student_code == "2210001"
        assert result.user.employee_code is None

    async def test_a_user_with_no_membership_codes_is_left_alone(
        self, overview: dict[str, Any]
    ) -> None:
        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )
        assert result.user.student_code is None


class TestTheStudentsCourseList:
    async def test_a_dropped_enrolment_is_shown_rather_than_skipped(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """The restored behaviour.

        Skipping these made a student who left a course vanish from the page
        entirely, so a manager asking "what happened to them in Databases?"
        got no answer at all. The status is on the row and the client renders
        it muted; the history is the point.
        """
        active, dropped = uuid4(), uuid4()
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(
                return_value=[
                    _enrollment(active, status="active"),
                    _enrollment(dropped, status="dropped"),
                ]
            ),
        )
        monkeypatch.setattr(
            overview["courses_api"],
            "get_course_by_id",
            AsyncMock(side_effect=lambda _db, course_id: _course(course_id)),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert {c.course_id for c in result.courses} == {active, dropped}
        assert {c.enrollment_status for c in result.courses} == {"active", "dropped"}

    async def test_an_enrolment_whose_course_is_gone_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """A row with no course has nothing to render -- no title, no link.

        This is the one case that is dropped, and it is different in kind
        from a dropped enrolment: there is no history to show.
        """
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(uuid4())]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=None)
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.courses == []

    async def test_progress_numbers_come_from_the_progress_feature(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """Recomputing them here would give this page a second opinion about
        a number the learner already sees elsewhere."""
        course_id = uuid4()
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"],
            "get_course_progress_for_user",
            AsyncMock(
                return_value=_progress(
                    completion_percent=62.5, completed_lessons=5, total_lessons=8
                )
            ),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        row = result.courses[0]
        assert row.completion_percent == 62.5
        assert (row.completed_lessons, row.total_lessons) == (5, 8)

    async def test_missing_progress_keys_read_as_zero(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """The progress summary crosses a feature boundary as a plain dict,
        so a key it stops sending must not take the whole page down."""
        course_id = uuid4()
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"], "get_course_progress_for_user", AsyncMock(return_value={})
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        row = result.courses[0]
        assert (row.completion_percent, row.completed_lessons, row.total_lessons) == (0, 0, 0)


class TestWhenTheStudentWasLastActive:
    """The number a manager uses to decide whether to follow up."""

    async def test_activity_beats_the_login_timestamp(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """Someone who signed in once in September and has been working
        since would otherwise look dormant on the page built to spot exactly
        that.
        """
        course_id = uuid4()
        worked_at = _NOW + timedelta(days=20)
        monkeypatch.setattr(
            admin_service,
            "_serialize_search_row",
            AsyncMock(return_value=_user_read(overview["user_id"], last_login_at=_NOW)),
        )
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"],
            "get_course_progress_for_user",
            AsyncMock(return_value=_progress(last_activity_at=worked_at)),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.last_active_at == worked_at

    async def test_an_older_activity_does_not_overwrite_a_newer_login(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """It is a maximum, not a last-write-wins."""
        course_id = uuid4()
        recent_login = _NOW + timedelta(days=30)
        monkeypatch.setattr(
            admin_service,
            "_serialize_search_row",
            AsyncMock(
                return_value=_user_read(overview["user_id"], last_login_at=recent_login)
            ),
        )
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"],
            "get_course_progress_for_user",
            AsyncMock(return_value=_progress(last_activity_at=_NOW)),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.last_active_at == recent_login

    async def test_a_student_who_never_logged_in_still_reports_their_work(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """``None`` is not a date to compare against, so the first activity
        seen becomes the answer rather than being discarded."""
        course_id = uuid4()
        worked_at = _NOW + timedelta(days=3)
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"],
            "get_course_progress_for_user",
            AsyncMock(return_value=_progress(last_activity_at=worked_at)),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.last_active_at == worked_at

    async def test_an_iso_string_from_the_boundary_is_parsed(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """The summary arrives as a plain dict, which may have been through
        JSON on the way -- comparing a string to a datetime would raise."""
        course_id = uuid4()
        monkeypatch.setattr(
            overview["enrollments_api"],
            "list_user_course_enrollments",
            AsyncMock(return_value=[_enrollment(course_id)]),
        )
        monkeypatch.setattr(
            overview["courses_api"], "get_course_by_id", AsyncMock(return_value=_course(course_id))
        )
        monkeypatch.setattr(
            overview["progress_api"],
            "get_course_progress_for_user",
            AsyncMock(return_value=_progress(last_activity_at="2026-10-05T09:00:00+00:00")),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.last_active_at == datetime(2026, 10, 5, 9, 0, tzinfo=UTC)


class TestTheCareerPathSection:
    def _path_row(self, path_id: UUID) -> dict[str, Any]:
        return {
            "career_path_id": path_id,
            "name": "Data Engineering",
            "slug": "data-engineering",
            "status": "active",
            "started_at": _NOW,
            "completed_at": None,
        }

    async def test_completed_courses_are_counted_from_the_satisfied_flag(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        path_id = uuid4()
        monkeypatch.setattr(
            overview["career_paths_api"],
            "list_user_career_enrollments",
            AsyncMock(return_value=[self._path_row(path_id)]),
        )
        monkeypatch.setattr(
            overview["career_paths_api"],
            "get_path_course_progress_for_user",
            AsyncMock(
                return_value=[
                    {"satisfied": True, "completion_percent": 100},
                    {"satisfied": True, "completion_percent": 100},
                    {"satisfied": False, "completion_percent": 40},
                ]
            ),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        row = result.career_paths[0]
        assert row.completed_courses == 2
        assert row.course_count == 3
        assert row.completion_percent == 80.0, "the mean across the path's courses"

    async def test_a_path_with_no_courses_reports_zero_rather_than_dividing(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """A freshly published path can legitimately have no courses yet,
        and a page that 500s on it is worse than one showing 0%."""
        path_id = uuid4()
        monkeypatch.setattr(
            overview["career_paths_api"],
            "list_user_career_enrollments",
            AsyncMock(return_value=[self._path_row(path_id)]),
        )
        monkeypatch.setattr(
            overview["career_paths_api"],
            "get_path_course_progress_for_user",
            AsyncMock(return_value=[]),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        row = result.career_paths[0]
        assert row.completion_percent == 0.0
        assert (row.completed_courses, row.course_count) == (0, 0)

    async def test_the_mean_is_rounded_for_display(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """Three courses rarely divide cleanly, and the page shows the
        number verbatim."""
        monkeypatch.setattr(
            overview["career_paths_api"],
            "list_user_career_enrollments",
            AsyncMock(return_value=[self._path_row(uuid4())]),
        )
        monkeypatch.setattr(
            overview["career_paths_api"],
            "get_path_course_progress_for_user",
            AsyncMock(
                return_value=[
                    {"satisfied": False, "completion_percent": 10},
                    {"satisfied": False, "completion_percent": 20},
                    {"satisfied": False, "completion_percent": 40},
                ]
            ),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert result.career_paths[0].completion_percent == 23.33

    async def test_programs_are_read_from_their_own_feature(
        self, monkeypatch: pytest.MonkeyPatch, overview: dict[str, Any]
    ) -> None:
        """A program pins a path VERSION, so its progress is measured
        against what the student was enrolled onto rather than the path's
        current head -- which is why it cannot be derived from the career
        path section sitting beside it.
        """
        row = {
            "enrollment_id": uuid4(),
            "learning_program_id": uuid4(),
            "program_name": "BSc Software Engineering",
            "program_version_no": 2,
            "status": "active",
            "enrolled_at": _NOW,
        }
        monkeypatch.setattr(
            overview["learning_programs_api"],
            "list_student_program_enrollments",
            AsyncMock(return_value=[row]),
        )

        result = await admin_service.get_user_overview(
            overview["db"], user_id=overview["user_id"]
        )

        assert len(result.programs) == 1
        assert result.programs[0].program_version_no == 2


class TestTheAdminInvite:
    @pytest.fixture
    def invite(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        monkeypatch.setattr(
            admin_service.user_queries, "get_user_by_email", AsyncMock(return_value=None)
        )
        grant = AsyncMock()
        monkeypatch.setattr(admin_service.access_control_api, "grant_org_role_access", grant)
        added: list[Any] = []
        return {
            "db": SimpleNamespace(add=added.append, flush=AsyncMock()),
            "added": added,
            "grant": grant,
            "actor_id": uuid4(),
        }

    def _payload(self, **over: Any) -> SimpleNamespace:
        base = {
            "primary_email": "new.teacher@test.local",
            "given_name": "Van A",
            "family_name": "Nguyen",
            "display_name": None,
            "organization_id": uuid4(),
            "role_code": "teacher",
            "student_code": None,
            "employee_code": None,
        }
        base.update(over)
        return SimpleNamespace(**base)

    async def test_the_admin_role_cannot_be_granted_through_an_org_invite(
        self, invite: dict[str, Any]
    ) -> None:
        """The guard worth more than the rest of this file.

        ``admin`` carries every permission and is global-scope, so granting
        it here would not merely over-privilege the invited account -- it
        would attach a global role through an org-scoped path, which is a
        privilege escalation available to anyone who can invite.
        """
        with pytest.raises(ForbiddenError, match="admin"):
            await admin_service.create_user_account(
                invite["db"],
                payload=self._payload(role_code="admin"),
                actor_id=invite["actor_id"],
            )

        invite["grant"].assert_not_awaited()
        assert invite["added"] == [], "no account is created either"

    async def test_the_refusal_names_the_roles_that_are_allowed(
        self, invite: dict[str, Any]
    ) -> None:
        """An admin hitting this needs to know what to pick instead."""
        with pytest.raises(ForbiddenError, match="student/teacher/hod/manager"):
            await admin_service.create_user_account(
                invite["db"],
                payload=self._payload(role_code="admin"),
                actor_id=invite["actor_id"],
            )

    @pytest.mark.parametrize("role", ["student", "teacher", "hod", "manager"])
    async def test_the_tenant_roles_are_invitable(
        self, invite: dict[str, Any], role: str
    ) -> None:
        await admin_service.create_user_account(
            invite["db"], payload=self._payload(role_code=role), actor_id=invite["actor_id"]
        )

        assert invite["grant"].await_args.kwargs["role_code"] == role

    async def test_an_existing_email_is_a_conflict(
        self, monkeypatch: pytest.MonkeyPatch, invite: dict[str, Any]
    ) -> None:
        """Two accounts on one address would both match at sign-in, and
        whichever the OAuth gate picked would be arbitrary."""
        monkeypatch.setattr(
            admin_service.user_queries,
            "get_user_by_email",
            AsyncMock(return_value=SimpleNamespace(id=uuid4())),
        )

        with pytest.raises(ConflictError, match="already exists"):
            await admin_service.create_user_account(
                invite["db"], payload=self._payload(), actor_id=invite["actor_id"]
            )

    async def test_an_invite_without_an_organization_is_refused(
        self, invite: dict[str, Any]
    ) -> None:
        """The router always supplies one, so this is a backstop for direct
        service callers -- and the thing it prevents is a membership row
        with a NULL org, which no scope check can evaluate.
        """
        with pytest.raises(ConflictError, match="organization_id is required"):
            await admin_service.create_user_account(
                invite["db"],
                payload=self._payload(organization_id=None),
                actor_id=invite["actor_id"],
            )

    async def test_the_account_is_created_active(self, invite: dict[str, Any]) -> None:
        """So the invited address can sign in through Google straight away:
        the pre-registration gate accepts an existing ``users`` row, and a
        pending account would make the invite a dead end.
        """
        await admin_service.create_user_account(
            invite["db"], payload=self._payload(), actor_id=invite["actor_id"]
        )

        users = [row for row in invite["added"] if hasattr(row, "primary_email")]
        assert len(users) == 1
        assert users[0].status == "active"

    async def test_the_display_name_falls_back_to_the_local_part(
        self, invite: dict[str, Any]
    ) -> None:
        """Not the whole address: the name appears next to the person on
        every roster, and showing their full email there publishes it to
        everyone who can see the page.
        """
        await admin_service.create_user_account(
            invite["db"],
            payload=self._payload(primary_email="van.a@test.local", display_name=None),
            actor_id=invite["actor_id"],
        )

        profiles = [row for row in invite["added"] if hasattr(row, "display_name")]
        assert profiles[0].display_name == "van.a"

    async def test_a_supplied_display_name_wins(self, invite: dict[str, Any]) -> None:
        await admin_service.create_user_account(
            invite["db"],
            payload=self._payload(display_name="Nguyen Van A"),
            actor_id=invite["actor_id"],
        )

        profiles = [row for row in invite["added"] if hasattr(row, "display_name")]
        assert profiles[0].display_name == "Nguyen Van A"

    async def test_the_inviting_admin_is_recorded_as_the_grantor(
        self, invite: dict[str, Any]
    ) -> None:
        """The role assignment is an audit record: who let this person in,
        with what, and when."""
        payload = self._payload(student_code="2210001")

        await admin_service.create_user_account(
            invite["db"], payload=payload, actor_id=invite["actor_id"]
        )

        kwargs = invite["grant"].await_args.kwargs
        assert kwargs["granted_by"] == invite["actor_id"]
        assert kwargs["organization_id"] == payload.organization_id
        assert kwargs["student_code"] == "2210001"


class TestTheSearchFilters:
    @pytest.fixture
    def search(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        query = AsyncMock(
            return_value=SimpleNamespace(
                items=[], total=0, page=0, page_size=25, total_pages=0
            )
        )
        monkeypatch.setattr(admin_service.user_queries, "search_users", query)
        monkeypatch.setattr(
            admin_service.user_queries, "list_profiles", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            admin_service.user_queries, "list_storage_objects", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_role_codes_for_users",
            AsyncMock(return_value={}),
        )
        monkeypatch.setattr(
            admin_service.access_control_api,
            "get_primary_orgs_for_users",
            AsyncMock(return_value={}),
        )
        return {"db": SimpleNamespace(), "query": query}

    async def test_several_filters_narrow_rather_than_widen(
        self, monkeypatch: pytest.MonkeyPatch, search: dict[str, Any]
    ) -> None:
        """"Teachers in Org A" means both, not either. A union would show a
        manager every teacher in the university the moment they also picked
        an organization.
        """
        shared, teacher_only, org_only = uuid4(), uuid4(), uuid4()
        monkeypatch.setattr(
            admin_service.access_control_api,
            "list_user_ids_with_role",
            AsyncMock(return_value=[shared, teacher_only]),
        )
        monkeypatch.setattr(
            admin_service.access_control_api,
            "list_user_ids_in_org",
            AsyncMock(return_value=[shared, org_only]),
        )

        await admin_service.search_users(
            search["db"], role="teacher", organization=uuid4()
        )

        assert search["query"].await_args.kwargs["restrict_ids"] == [shared]

    async def test_an_empty_intersection_short_circuits_the_query(
        self, monkeypatch: pytest.MonkeyPatch, search: dict[str, Any]
    ) -> None:
        """No teacher belongs to that org, so there is nothing to ask the
        database -- and an empty ``IN ()`` list is the kind of thing that
        quietly matches everything instead of nothing.
        """
        monkeypatch.setattr(
            admin_service.access_control_api,
            "list_user_ids_with_role",
            AsyncMock(return_value=[uuid4()]),
        )
        monkeypatch.setattr(
            admin_service.access_control_api,
            "list_user_ids_in_org",
            AsyncMock(return_value=[uuid4()]),
        )

        page = await admin_service.search_users(
            search["db"], role="teacher", organization=uuid4(), page=2, page_size=10
        )

        assert page.items == []
        assert page.total == 0
        assert (page.page, page.page_size) == (2, 10), "the caller's paging is echoed back"
        search["query"].assert_not_awaited()

    async def test_no_filters_means_no_allowlist(self, search: dict[str, Any]) -> None:
        """``None`` is not the same as an empty list: one means "do not
        restrict", the other would mean "match nothing"."""
        await admin_service.search_users(search["db"])

        assert search["query"].await_args.kwargs["restrict_ids"] is None


class TestTheAvatarOnAListRow:
    async def test_a_storage_blip_costs_one_avatar_not_the_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One unreadable object among fifty rows must not fail the list.

        The client falls back to initials, which is a better outcome than a
        manager unable to open the user list at all.
        """
        monkeypatch.setattr(
            admin_service,
            "create_stream_url",
            AsyncMock(side_effect=ConnectionError("s3 down")),
        )
        object_id = uuid4()
        profile = SimpleNamespace(avatar_object_id=object_id)

        assert (
            await admin_service._mint_avatar_url({object_id: SimpleNamespace()}, profile) is None
        )

    async def test_an_avatar_missing_from_the_batch_map_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The map is loaded once for the page; a pointer with no row in it
        must not trigger a lookup of its own and reintroduce the N+1."""
        presign = AsyncMock()
        monkeypatch.setattr(admin_service, "create_stream_url", presign)

        assert (
            await admin_service._mint_avatar_url({}, SimpleNamespace(avatar_object_id=uuid4()))
            is None
        )
        presign.assert_not_awaited()

    async def test_a_row_without_a_profile_needs_no_storage_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presign = AsyncMock()
        monkeypatch.setattr(admin_service, "create_stream_url", presign)

        assert await admin_service._mint_avatar_url({}, None) is None
        presign.assert_not_awaited()


class TestThePaginationCursor:
    def test_a_cursor_round_trips(self) -> None:
        """It is handed back to the client and returned verbatim, so a
        cursor that does not decode to what it encoded strands the caller
        mid-list."""
        user_id = uuid4()
        assert admin_service._decode_cursor(admin_service._encode_cursor(user_id)) == user_id

    def test_the_cursor_carries_no_padding(self) -> None:
        """It travels in a URL, where ``=`` has to be escaped; stripping it
        on encode is why the decoder puts it back."""
        assert "=" not in admin_service._encode_cursor(uuid4())

    def test_many_ids_round_trip(self) -> None:
        """Base64 padding depends on input length, so a single example can
        pass while a different id fails."""
        ids = [uuid4() for _ in range(50)]
        assert [
            admin_service._decode_cursor(admin_service._encode_cursor(i)) for i in ids
        ] == ids
