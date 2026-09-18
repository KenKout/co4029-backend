"""A program that declines to cap career paths, deferring to the org limit.

``max_career_paths_per_enrollment`` is now NULLABLE, and null is the value
worth testing: the program sets no cap of its own and the student is bounded
by ``learning_program.max_concurrent_paths_per_student`` -- counted across
every program they are in -- plus the implicit bound that they can only pick
paths this program actually offers.

Null used to be impossible: the column was NOT NULL with a server default of
1, so the only way to say "as many as the student is allowed" was to type the
maximum, which stops meaning that the moment the maximum moves.

Two ways to get null wrong, and neither raises:

* **Reading it as zero.** ``len(selected) >= None`` does raise, but the
  equivalent comparison on the client (``count < null``) is simply false, and
  an uncapped program silently looks full.
* **Losing it on the way in.** ``payload.x is not None`` cannot tell "clear
  the cap" from "I did not mention the cap", so removing a limit would leave
  the old number in place while the API answered 200.

The database and the query layer are mocked; these are the service's own
branches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import ConflictError
from abridgeai.features.learning_programs import services
from abridgeai.features.learning_programs.schemas import ProgramUpdate


def _recording_db() -> SimpleNamespace:
    added: list[Any] = []
    return SimpleNamespace(add=added.append, added=added)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A student holding one path, adding a second in the same program."""
    student_id = uuid4()
    target_path_id = uuid4()
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=student_id,
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="active",
        updated_by=None,
    )
    existing = SimpleNamespace(
        id=uuid4(),
        program_enrollment_id=enrollment.id,
        career_path_id=uuid4(),
        status="active",
    )
    program = SimpleNamespace(
        id=enrollment.learning_program_id, name="Data Science", organization_id=uuid4()
    )
    version = SimpleNamespace(
        id=enrollment.program_version_id, max_career_paths_per_enrollment=None
    )

    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_enrollment_out", AsyncMock(return_value="ok"))
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Path 2"))
    monkeypatch.setattr(services.career_paths_api, "ensure_program_path_access", AsyncMock())
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(services.queries, "get_version", AsyncMock(return_value=version))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[existing]))
    monkeypatch.setattr(
        services.queries,
        "list_version_paths",
        AsyncMock(
            return_value=[
                {
                    "career_path_id": target_path_id,
                    "career_path_version_id": uuid4(),
                    "status": "published",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        services.queries, "find_active_path_attempt_elsewhere", AsyncMock(return_value=None)
    )
    # Plenty of student-wide room unless a test says otherwise.
    monkeypatch.setattr(
        services.queries, "count_active_paths_for_student", AsyncMock(return_value=1)
    )
    monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=10))
    return {
        "db": _recording_db(),
        "version": version,
        "student_id": student_id,
        "target_path_id": target_path_id,
        "enrollment": enrollment,
        "monkeypatch": monkeypatch,
    }


async def _select(world: dict[str, Any]) -> Any:
    return await services.select_path(
        world["db"],
        enrollment_id=world["enrollment"].id,
        career_path_id=world["target_path_id"],
        student_id=world["student_id"],
    )


def _budget(world: dict[str, Any], *, used: int, limit: int) -> None:
    world["monkeypatch"].setattr(
        services.queries, "count_active_paths_for_student", AsyncMock(return_value=used)
    )
    world["monkeypatch"].setattr(services, "resolve_setting", AsyncMock(return_value=limit))


class TestSelectingAPathInAnUncappedProgram:
    async def test_a_second_path_is_allowed_where_a_cap_of_one_would_refuse(
        self, world: dict[str, Any]
    ) -> None:
        """The whole point of null.

        The student already holds one path in this program. Under the old
        NOT NULL default of 1 this was ``career_path_selection_limit_reached``.
        """
        assert await _select(world) == "ok"

        assert len(world["db"].added) == 1

    async def test_no_cap_does_not_mean_no_limit(self, world: dict[str, Any]) -> None:
        """The organization's cross-program ceiling still applies.

        This is the limit that replaced the per-program one, so an uncapped
        program handing out unlimited paths would defeat it.
        """
        _budget(world, used=10, limit=10)

        with pytest.raises(services.ProgramConflictError) as raised:
            await _select(world)

        assert raised.value.code == "student_path_limit_reached"
        assert world["db"].added == []

    async def test_a_path_the_program_does_not_offer_is_still_refused(
        self, world: dict[str, Any]
    ) -> None:
        """The other implicit bound: null is not a licence to pick anything.

        A student can only hold paths pinned to their program version, so
        "uncapped" tops out at the size of the program.
        """
        world["monkeypatch"].setattr(
            services.queries, "list_version_paths", AsyncMock(return_value=[])
        )

        with pytest.raises(ConflictError, match="path_is_not_in_the_pinned_program_version"):
            await _select(world)

    async def test_the_same_path_twice_is_still_refused(self, world: dict[str, Any]) -> None:
        """Removing the count check must not remove the duplicate check with
        it -- they sat next to each other."""
        world["monkeypatch"].setattr(
            services.queries,
            "list_attempts",
            AsyncMock(
                return_value=[
                    SimpleNamespace(career_path_id=world["target_path_id"], status="active")
                ]
            ),
        )

        with pytest.raises(ConflictError, match="path_already_selected"):
            await _select(world)


class TestAProgramThatDoesSetACap:
    async def test_the_cap_is_still_enforced(self, world: dict[str, Any]) -> None:
        """Null is a skip, not a removal: a manager who typed 1 still gets 1."""
        world["version"].max_career_paths_per_enrollment = 1

        with pytest.raises(services.ProgramConflictError) as raised:
            await _select(world)

        assert raised.value.code == "career_path_selection_limit_reached"
        assert world["db"].added == []

    async def test_room_under_the_cap_is_allowed(self, world: dict[str, Any]) -> None:
        world["version"].max_career_paths_per_enrollment = 3

        assert await _select(world) == "ok"

    async def test_the_message_is_singular_for_a_cap_of_one(
        self, world: dict[str, Any]
    ) -> None:
        world["version"].max_career_paths_per_enrollment = 1

        with pytest.raises(services.ProgramConflictError) as raised:
            await _select(world)

        assert "at most 1 selected career path." in raised.value.message


class TestTellingAClearedCapFromAnUnmentionedOne:
    """``ProgramUpdate.max_career_paths_per_enrollment`` is ``int | None``.

    Every other optional field on that model uses ``None`` to mean "leave it
    alone". This one cannot, because ``None`` is now a value a manager can
    choose -- so the service reads ``model_fields_set`` instead. A test per
    direction, because getting it wrong produces a 200 that changed nothing.
    """

    def test_an_omitted_limit_is_not_in_the_fields_set(self) -> None:
        payload = ProgramUpdate(name="Renamed")

        assert "max_career_paths_per_enrollment" not in payload.model_fields_set
        assert payload.max_career_paths_per_enrollment is None

    def test_an_explicit_null_is_in_the_fields_set(self) -> None:
        """Identical attribute value, opposite meaning. This distinction is
        the only thing that lets a manager remove a cap."""
        payload = ProgramUpdate(max_career_paths_per_enrollment=None)

        assert "max_career_paths_per_enrollment" in payload.model_fields_set
        assert payload.max_career_paths_per_enrollment is None

    def test_a_number_is_in_the_fields_set(self) -> None:
        payload = ProgramUpdate(max_career_paths_per_enrollment=4)

        assert "max_career_paths_per_enrollment" in payload.model_fields_set
        assert payload.max_career_paths_per_enrollment == 4

    @pytest.mark.parametrize("bad", [0, 11, -1])
    def test_a_value_outside_the_bound_is_rejected(self, bad: int) -> None:
        """Nullable widened what the field accepts; it did not widen the
        range. The CHECK constraint is still 1..10 for rows that set one."""
        with pytest.raises(ValueError, match="max_career_paths_per_enrollment"):
            ProgramUpdate(max_career_paths_per_enrollment=bad)

    def test_creating_without_a_limit_leaves_it_unset(self) -> None:
        """A new program imposes no cap of its own unless asked. This used to
        default to 1, so every program created through the API was
        single-path whether or not anyone intended it."""
        from abridgeai.features.learning_programs.schemas import ProgramCreate

        payload = ProgramCreate(faculty_id=uuid4(), slug="data-science", name="Data Science")

        assert payload.max_career_paths_per_enrollment is None


class TestPublishingAnUncappedProgram:
    """Publish refuses a cap larger than the number of paths attached -- a
    program cannot promise four paths while offering two. With no cap there
    is nothing to compare, and the check must not fire on ``None``."""

    @pytest.fixture
    def published(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        program = SimpleNamespace(
            id=uuid4(), organization_id=uuid4(), status="draft", updated_by=None
        )
        version = SimpleNamespace(
            id=uuid4(),
            status="draft",
            max_career_paths_per_enrollment=None,
            published_at=None,
            updated_by=None,
        )
        monkeypatch.setattr(services, "_require_operator", AsyncMock())
        monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
        monkeypatch.setattr(services, "_program_out", AsyncMock(return_value="published"))
        monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
        monkeypatch.setattr(
            services.queries, "get_current_version", AsyncMock(return_value=version)
        )
        monkeypatch.setattr(
            services.queries,
            "list_version_paths",
            AsyncMock(return_value=[{"is_default": True}]),
        )
        monkeypatch.setattr(
            services.queries, "list_unpublishable_version_path_ids", AsyncMock(return_value=[])
        )
        return {
            "db": SimpleNamespace(),
            "program": program,
            "version": version,
            "actor": SimpleNamespace(user_id=uuid4()),
        }

    async def _publish(self, published: dict[str, Any]) -> Any:
        return await services.publish_program(
            published["db"],
            program_id=published["program"].id,
            actor=published["actor"],
        )

    async def test_a_one_path_program_with_no_cap_publishes(
        self, published: dict[str, Any]
    ) -> None:
        assert await self._publish(published) == "published"
        assert published["version"].status == "published"

    async def test_a_cap_larger_than_the_program_is_still_refused(
        self, published: dict[str, Any]
    ) -> None:
        """One path attached, a cap of two: the comparison still runs for a
        program that sets one."""
        published["version"].max_career_paths_per_enrollment = 2

        with pytest.raises(services.ProgramConflictError) as raised:
            await self._publish(published)

        assert raised.value.code == "career_path_limit_exceeds_program_paths"

    async def test_a_cap_matching_the_program_publishes(
        self, published: dict[str, Any]
    ) -> None:
        published["version"].max_career_paths_per_enrollment = 1

        assert await self._publish(published) == "published"
