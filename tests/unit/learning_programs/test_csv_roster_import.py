"""Importing a roster file into a learning program.

This function exists because the hand-picked enrolment path is
all-or-nothing, and that is the wrong shape for a file. A manager uploads
a spreadsheet someone else maintains: it has a typo on line 40, the same
person on lines 12 and 58, and nine people who were already imported last
week. Aborting the batch would import nobody and tell the manager nothing
about which line was at fault.

So every row is decided on its own, and the four outcomes are deliberately
distinct. ``enrolled`` and ``created_users`` overlap by design -- a row can
enrol an account that already existed, or create one and enrol it.
``already_enrolled`` is *not* a failure, because re-uploading last week's
file is a normal thing to do and reporting it as an error trains the
manager to ignore the error list. ``failures`` carries a row number,
because "one row failed" is useless against a file of two thousand.

The one thing the per-row design must not buy is a way around the rules:
the concurrent-enrolment cap is checked on every row exactly as the
hand-picked path checks it, so importing a file cannot enrol someone the
UI would have refused.

Everything below the service is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.features.learning_programs import services


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A published program with a published version and a default path.

    Returns the knobs each test bends: the queries module's stubs, the
    identity resolver, and the per-org concurrency limit.
    """
    program = SimpleNamespace(
        id=uuid4(),
        organization_id=uuid4(),
        status="published",
        name="BSc Software Engineering",
    )
    version = SimpleNamespace(id=uuid4())

    monkeypatch.setattr(services, "_require_operator", AsyncMock())
    monkeypatch.setattr(services, "_get_enrollable_default_path", AsyncMock(return_value=None))
    monkeypatch.setattr(services, "_activate_default_path", AsyncMock())
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=3))

    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(
        services.queries, "get_current_version", AsyncMock(return_value=version)
    )
    monkeypatch.setattr(
        services.queries, "get_program_enrollment", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        services.queries, "count_concurrent_enrollments", AsyncMock(return_value=0)
    )

    find_or_create = AsyncMock(side_effect=lambda *a, **k: (uuid4(), False))
    monkeypatch.setattr(services.identity_api, "find_or_create_student", find_or_create)

    added: list[Any] = []
    db = SimpleNamespace(add=added.append)
    return {
        "db": db,
        "added": added,
        "program": program,
        "version": version,
        "find_or_create": find_or_create,
        "actor": SimpleNamespace(user_id=uuid4()),
    }


async def _run(world: dict[str, Any], rows: list[dict[str, str]]):
    return await services.import_students_from_csv(
        world["db"],
        program_id=world["program"].id,
        rows=rows,
        actor=world["actor"],
    )


class TestTheGatesBeforeAnyRowIsRead:
    """A file is rejected as a whole only for reasons that apply to it as a
    whole -- there is no per-row answer to "this program is not published".
    """

    async def test_an_unknown_program_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=None))
        with pytest.raises(NotFoundError, match="learning_program_not_found"):
            await _run(world, [{"email": "a@test.local"}])

    @pytest.mark.parametrize("status", ["draft", "archived"])
    async def test_only_a_published_program_accepts_a_roster(
        self, world: dict[str, Any], status: str
    ) -> None:
        world["program"].status = status
        with pytest.raises(ConflictError, match="only_published_programs_accept_enrollments"):
            await _run(world, [{"email": "a@test.local"}])

    async def test_a_program_with_no_published_version_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Enrolment pins a version, so there has to be one to pin."""
        monkeypatch.setattr(
            services.queries, "get_current_version", AsyncMock(return_value=None)
        )
        with pytest.raises(ConflictError, match="program_has_no_published_version"):
            await _run(world, [{"email": "a@test.local"}])

    async def test_the_operator_check_runs_before_the_file_is_touched(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Importing is a write on someone else's roster, so the permission
        failure must surface as a permission failure -- not as two thousand
        per-row errors."""
        monkeypatch.setattr(
            services, "_require_operator", AsyncMock(side_effect=ConflictError("not_an_operator"))
        )
        with pytest.raises(ConflictError, match="not_an_operator"):
            await _run(world, [{"email": "a@test.local"}])
        world["find_or_create"].assert_not_awaited()


class TestOneRowDoesNotDecideTheBatch:
    async def test_a_malformed_row_fails_alone(self, world: dict[str, Any]) -> None:
        """The reason this is not built on the hand-picked path.

        A file with one bad line is the normal case, not the exception, and
        the two good rows either side of it still have to land.
        """
        result = await _run(
            world,
            [
                {"email": "good1@test.local"},
                {"not_a_column": "x"},
                {"email": "good2@test.local"},
            ],
        )

        assert len(result.enrolled) == 2
        assert len(result.failures) == 1
        assert result.failures[0].row_number == 2

    async def test_a_failure_is_keyed_to_its_line_in_the_file(
        self, world: dict[str, Any]
    ) -> None:
        """Against a two-thousand-line file, "a row failed" is not
        actionable; the manager has to be able to open the file and look."""
        result = await _run(
            world,
            [{"email": "ok@test.local"}, {"email": "ok2@test.local"}, {"email": ""}],
        )

        assert [f.row_number for f in result.failures] == [3], "1-based, matching the file"
        assert result.failures[0].reason.startswith("invalid_row:")

    async def test_a_malformed_row_still_reports_whatever_identifier_it_had(
        self, world: dict[str, Any]
    ) -> None:
        """The row number locates the line; the email is what the manager
        searches for when the line numbers have shifted."""
        result = await _run(world, [{"email": "x", "unknown_column": "y"}])

        assert result.failures[0].identifier == "x"

    async def test_a_row_with_no_email_at_all_reports_no_identifier(
        self, world: dict[str, Any]
    ) -> None:
        result = await _run(world, [{"given_name": "Nameless"}])

        assert result.failures[0].identifier is None
        assert result.enrolled == []

    async def test_a_service_error_on_one_row_does_not_stop_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """A row can fail deep inside account creation -- a clashing student
        code, a role assignment that will not resolve. That is still one
        row's problem.
        """
        calls = {"n": 0}

        async def _find(*_a: Any, **_k: Any) -> tuple[UUID, bool]:
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConflictError("student_code_taken")
            return uuid4(), False

        monkeypatch.setattr(services.identity_api, "find_or_create_student", _find)

        result = await _run(
            world,
            [
                {"email": "a@test.local"},
                {"email": "b@test.local"},
                {"email": "c@test.local"},
            ],
        )

        assert len(result.enrolled) == 2
        assert [f.identifier for f in result.failures] == ["b@test.local"]
        assert result.failures[0].reason == "student_code_taken"


class TestTheSamePersonTwiceInOneFile:
    async def test_a_repeated_email_is_imported_once_and_not_reported(
        self, world: dict[str, Any]
    ) -> None:
        """A duplicate line is a property of the file, not of the person.

        Reporting the second one as "already enrolled" would be true only
        because the first line just enrolled them, which tells the manager
        nothing about their roster.
        """
        result = await _run(
            world, [{"email": "dup@test.local"}, {"email": "dup@test.local"}]
        )

        assert len(result.enrolled) == 1
        assert result.already_enrolled == []
        assert result.failures == []

    @pytest.mark.parametrize(
        "second", ["DUP@test.local", "  dup@test.local  ", "Dup@Test.Local"]
    )
    async def test_the_duplicate_check_ignores_case_and_padding(
        self, world: dict[str, Any], second: str
    ) -> None:
        """Spreadsheets carry both. An email that differs only in case is
        the same mailbox, and treating it otherwise would create a second
        account for the same person.
        """
        result = await _run(world, [{"email": "dup@test.local"}, {"email": second}])

        assert len(result.enrolled) == 1

    async def test_the_normalised_email_is_what_gets_looked_up(
        self, world: dict[str, Any]
    ) -> None:
        await _run(world, [{"email": "  MiXeD@Test.Local  "}])

        assert world["find_or_create"].await_args.kwargs["email"] == "mixed@test.local"


class TestWhoCountsAsAlreadyEnrolled:
    @pytest.mark.parametrize("status", ["awaiting_path", "active", "completed"])
    async def test_a_student_already_on_the_program_is_reported_not_failed(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any], status: str
    ) -> None:
        """Re-uploading last week's file is routine. Calling it a failure
        would fill the error list with rows that are perfectly fine and
        teach the manager to stop reading it.
        """
        student_id = uuid4()
        monkeypatch.setattr(
            services.identity_api,
            "find_or_create_student",
            AsyncMock(return_value=(student_id, False)),
        )
        monkeypatch.setattr(
            services.queries,
            "get_program_enrollment",
            AsyncMock(return_value=SimpleNamespace(status=status)),
        )

        result = await _run(world, [{"email": "already@test.local"}])

        assert result.already_enrolled == [student_id]
        assert result.enrolled == []
        assert result.failures == []

    async def test_a_withdrawn_student_is_reinstated_rather_than_skipped(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Someone who left and is on the new roster is coming back.

        Their old row is reused and cleared rather than a second one being
        written, because the enrolment is unique per (program, student) --
        a fresh insert would collide, and skipping them would silently
        leave them out of the program they were just re-added to.
        """
        student_id = uuid4()
        withdrawn = SimpleNamespace(
            status="withdrawn",
            program_version_id=uuid4(),
            enrolled_at=None,
            withdrawn_at="2026-01-01",
            withdrawal_reason="left the course",
            updated_by=None,
        )
        monkeypatch.setattr(
            services.identity_api,
            "find_or_create_student",
            AsyncMock(return_value=(student_id, False)),
        )
        monkeypatch.setattr(
            services.queries, "get_program_enrollment", AsyncMock(return_value=withdrawn)
        )

        result = await _run(world, [{"email": "back@test.local"}])

        assert result.enrolled == [student_id]
        assert withdrawn.status == "awaiting_path"
        assert withdrawn.withdrawn_at is None
        assert withdrawn.withdrawal_reason is None, "the old reason must not outlive the return"
        assert withdrawn.program_version_id == world["version"].id, (
            "they come back onto the version being imported, not the one they left"
        )
        assert world["added"] == [], "the existing row is reused rather than duplicated"


class TestCreatingAccountsAsNeeded:
    async def test_a_new_account_is_counted_separately_from_the_enrolment(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """The manager checks this number before sending welcome mail."""
        new_id = uuid4()
        monkeypatch.setattr(
            services.identity_api,
            "find_or_create_student",
            AsyncMock(return_value=(new_id, True)),
        )

        result = await _run(world, [{"email": "new@test.local"}])

        assert result.created_users == [new_id]
        assert result.enrolled == [new_id], "a created account is enrolled in the same pass"

    async def test_an_existing_account_is_enrolled_without_being_counted_as_created(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        existing_id = uuid4()
        monkeypatch.setattr(
            services.identity_api,
            "find_or_create_student",
            AsyncMock(return_value=(existing_id, False)),
        )

        result = await _run(world, [{"email": "known@test.local"}])

        assert result.enrolled == [existing_id]
        assert result.created_users == []

    async def test_new_accounts_land_in_the_programs_organization(
        self, world: dict[str, Any]
    ) -> None:
        """An import cannot be a way to mint accounts in another org."""
        await _run(world, [{"email": "new@test.local"}])

        kwargs = world["find_or_create"].await_args.kwargs
        assert kwargs["organization_id"] == world["program"].organization_id
        assert kwargs["actor_id"] == world["actor"].user_id

    async def test_the_optional_name_columns_are_passed_through(
        self, world: dict[str, Any]
    ) -> None:
        """They are only used when the account is new; an existing account
        keeps its own name, which the identity API enforces."""
        await _run(
            world,
            [
                {
                    "email": "new@test.local",
                    "given_name": "Van A",
                    "family_name": "Nguyen",
                    "display_name": "Nguyen Van A",
                }
            ],
        )

        kwargs = world["find_or_create"].await_args.kwargs
        assert kwargs["given_name"] == "Van A"
        assert kwargs["family_name"] == "Nguyen"
        assert kwargs["display_name"] == "Nguyen Van A"

    async def test_a_row_with_only_an_email_is_valid(self, world: dict[str, Any]) -> None:
        """Only ``email`` is required; a bare one-column file must import."""
        result = await _run(world, [{"email": "bare@test.local"}])

        assert len(result.enrolled) == 1
        assert world["find_or_create"].await_args.kwargs["given_name"] is None


class TestTheConcurrencyCapIsNotBypassable:
    """The cap exists so a student is not carrying more programs at once
    than the organization thinks anyone can.

    A per-row import that skipped it would be a way to put someone over the
    line that the hand-picked path would have refused -- and nothing about
    uploading a file rather than clicking names should change the answer.
    """

    async def test_a_student_at_the_cap_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services.queries, "count_concurrent_enrollments", AsyncMock(return_value=3)
        )

        result = await _run(world, [{"email": "busy@test.local"}])

        assert result.enrolled == []
        assert len(result.failures) == 1

    async def test_the_refusal_names_both_numbers(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """"Too many programs" leaves the manager guessing. The count and
        the limit together tell them whether to drop this student from the
        file or to ask for the setting to be raised.
        """
        monkeypatch.setattr(
            services.queries, "count_concurrent_enrollments", AsyncMock(return_value=5)
        )
        monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=2))

        result = await _run(world, [{"email": "busy@test.local"}])

        reason = result.failures[0].reason
        assert "5" in reason
        assert "2" in reason
        assert result.failures[0].identifier == "busy@test.local"

    async def test_a_student_below_the_cap_is_enrolled(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(
            services.queries, "count_concurrent_enrollments", AsyncMock(return_value=2)
        )

        result = await _run(world, [{"email": "fine@test.local"}])
        assert len(result.enrolled) == 1

    async def test_the_limit_is_resolved_for_the_programs_organization(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Different faculties tune this differently, so the ceiling has to
        be the one belonging to the program being imported into."""
        resolve = AsyncMock(return_value=3)
        monkeypatch.setattr(services, "resolve_setting", resolve)

        await _run(world, [{"email": "a@test.local"}])

        assert resolve.await_args.args[1] == "learning_program.max_concurrent_enrollments"
        assert resolve.await_args.kwargs["organization_id"] == world["program"].organization_id

    async def test_the_cap_is_read_once_for_the_whole_file(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Two thousand rows must not be two thousand settings lookups."""
        resolve = AsyncMock(return_value=3)
        monkeypatch.setattr(services, "resolve_setting", resolve)

        await _run(world, [{"email": f"s{n}@test.local"} for n in range(25)])

        assert resolve.await_count == 1


class TestWhatTheEnrolmentRowLooksLike:
    async def test_a_new_enrolment_pins_the_published_version(
        self, world: dict[str, Any]
    ) -> None:
        """The student is enrolled onto what was published when they were
        imported, not onto whatever the program becomes later."""
        await _run(world, [{"email": "new@test.local"}])

        assert len(world["added"]) == 1
        row = world["added"][0]
        assert row.program_version_id == world["version"].id
        assert row.learning_program_id == world["program"].id
        assert row.status == "awaiting_path"

    async def test_the_default_path_is_activated_for_every_enrolled_row(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """A single-path program should not leave twenty imported students
        sitting at "choose your path" with one thing to choose."""
        activate = AsyncMock()
        monkeypatch.setattr(services, "_activate_default_path", activate)

        await _run(world, [{"email": "a@test.local"}, {"email": "b@test.local"}])

        assert activate.await_count == 2

    async def test_an_empty_file_is_a_clean_empty_result(
        self, world: dict[str, Any]
    ) -> None:
        """Every list is present and empty, so the caller renders a summary
        rather than branching on ``None``."""
        result = await _run(world, [])

        assert result.enrolled == []
        assert result.created_users == []
        assert result.already_enrolled == []
        assert result.failures == []
