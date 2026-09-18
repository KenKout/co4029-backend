"""An organization ceiling on how many career paths one student runs at once.

``max_career_paths_per_enrollment`` is scoped to a single enrollment, so
before this limit existed a student's real ceiling was the SUM of their
programs' limits. Two programs capped at 1 and 2 let a student hold three
concurrent paths, and at no point is any program over its own limit -- the
first holds 1/1, the second 2/2. Nothing in the system was counting the
total, so nothing could refuse the third.

``learning_program.max_concurrent_paths_per_student`` is that count. It
defaults to 10, which changes nothing until an organization lowers it.

The interesting design point is WHERE it applies. Only two operations raise
a student's path count: choosing a path, and a program auto-starting its
default. A path *change* swaps one out for one in, so it is deliberately
not gated -- gating it would trap a student who is grandfathered above a
lowered ceiling, able to sit on their current path but never switch away
from it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import ConflictError
from abridgeai.features.learning_programs import services


def _recording_db() -> SimpleNamespace:
    added: list[Any] = []
    return SimpleNamespace(add=added.append, added=added)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Program B: a 2-path program, one slot already spent on its default.

    This is the second half of the reported scenario. Program A (capped at
    1) is running the student's first path; it is off-screen here and shows
    up only as the student's active-path count.
    """
    student_id = uuid4()
    target_path_id = uuid4()
    org_id = uuid4()
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
        id=enrollment.learning_program_id, name="Program B", organization_id=org_id
    )
    version = SimpleNamespace(id=enrollment.program_version_id, max_career_paths_per_enrollment=2)

    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_enrollment_out", AsyncMock(return_value="ok"))
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Path 3"))
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
    # The path being added is not running anywhere else; this is a budget
    # question, not a duplicate one.
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value=None),
    )
    return {
        "db": _recording_db(),
        "enrollment": enrollment,
        "student_id": student_id,
        "target_path_id": target_path_id,
        "monkeypatch": monkeypatch,
    }


def _budget(world: dict[str, Any], *, used: int, limit: int) -> None:
    world["monkeypatch"].setattr(
        services.queries, "count_active_paths_for_student", AsyncMock(return_value=used)
    )
    world["monkeypatch"].setattr(services, "resolve_setting", AsyncMock(return_value=limit))


async def _select(world: dict[str, Any]) -> Any:
    return await services.select_path(
        world["db"],
        enrollment_id=world["enrollment"].id,
        career_path_id=world["target_path_id"],
        student_id=world["student_id"],
    )


# --------------------------------------------------------------------------
# the reported scenario
# --------------------------------------------------------------------------


async def test_the_reported_case_program_a_1_plus_program_b_2_is_capped_at_2(
    world: dict[str, Any],
) -> None:
    """Program A holds path 1; program B holds path 2 and has a slot free.

    B's own limit (2) is not exhausted, so every per-program check passes and
    the "Add this path" button offers path 3. With the organization ceiling
    set to 2 the student is already at it, and the third path is refused.
    """
    _budget(world, used=2, limit=2)

    with pytest.raises(ConflictError) as excinfo:
        await _select(world)

    err = excinfo.value
    assert err.code == "student_path_limit_reached"
    assert err.fields == {"limit": 2, "current": 2}
    assert "2 career paths in progress" in err.message
    assert world["db"].added == [], "no third attempt may be written"


async def test_same_case_is_allowed_while_the_ceiling_has_room(
    world: dict[str, Any],
) -> None:
    """Default ceiling is 10, so nothing changes for an org that never lowers it."""
    _budget(world, used=2, limit=10)

    await _select(world)

    assert len(world["db"].added) == 1
    assert world["db"].added[0].career_path_id == world["target_path_id"]


async def test_the_boundary_is_the_last_free_slot_not_one_short(
    world: dict[str, Any],
) -> None:
    """used == limit - 1 must still be allowed; only used >= limit refuses."""
    _budget(world, used=1, limit=2)

    await _select(world)

    assert len(world["db"].added) == 1


async def test_a_grandfathered_student_over_the_lowered_ceiling_is_refused(
    world: dict[str, Any],
) -> None:
    """Lowering the setting does not cancel paths, but it does stop new ones."""
    _budget(world, used=5, limit=2)

    with pytest.raises(ConflictError) as excinfo:
        await _select(world)

    assert excinfo.value.code == "student_path_limit_reached"
    assert excinfo.value.fields == {"limit": 2, "current": 5}


async def test_the_message_reads_correctly_for_a_ceiling_of_one(
    world: dict[str, Any],
) -> None:
    """Singular/plural has to hold at the limit an org is most likely to pick."""
    _budget(world, used=1, limit=1)

    with pytest.raises(ConflictError) as excinfo:
        await _select(world)

    assert "1 career path in progress" in excinfo.value.message
    assert "career paths in progress" not in excinfo.value.message


# --------------------------------------------------------------------------
# the auto-started program default
# --------------------------------------------------------------------------


@pytest.fixture
def default_world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="awaiting_path",
        updated_by=None,
    )
    program = SimpleNamespace(
        id=enrollment.learning_program_id, name="Program C", organization_id=uuid4()
    )
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services.career_paths_api, "ensure_program_path_access", AsyncMock())
    # Default: the student has NOT already finished this path. The guard that
    # reads this is exercised in its own section below.
    monkeypatch.setattr(
        services.career_paths_api,
        "is_version_complete_for_user",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value=None),
    )
    return {
        "db": _recording_db(),
        "enrollment": enrollment,
        "default_path": {
            "career_path_id": uuid4(),
            "career_path_version_id": uuid4(),
            "status": "published",
            "is_default": True,
        },
        "actor_id": uuid4(),
        "monkeypatch": monkeypatch,
    }


async def test_default_path_is_left_unstarted_at_the_ceiling(
    default_world: dict[str, Any],
) -> None:
    """Enrolling still succeeds; the student just has to free a slot first.

    Failing the enrollment instead would break roster imports over one
    student, so the row stands and the enrollment waits in `awaiting_path`.
    """
    world = default_world
    _budget(world, used=2, limit=2)

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is False
    assert world["db"].added == []
    assert world["enrollment"].status == "awaiting_path"


async def test_default_path_starts_when_the_ceiling_has_room(
    default_world: dict[str, Any],
) -> None:
    world = default_world
    _budget(world, used=1, limit=2)

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is True
    assert len(world["db"].added) == 1
    assert world["enrollment"].status == "active"



# --------------------------------------------------------------------------
# a default path the student has already completed
# --------------------------------------------------------------------------
#
# Auto-starting one is how a program completes itself the moment a student
# joins it. Nothing fails at enrollment -- completion is event-driven, so the
# enrollment looks active until the next progress read calls
# ``complete_program_attempts``. That finds the inherited-complete attempt,
# completes it, sees no remaining active attempt, and flips the enrollment to
# ``completed``. ``select_path`` then refuses with
# ``paths_can_only_be_added_to_an_open_program``, so the student can never
# pick any of that program's OTHER paths.


def _already_completed(world: dict[str, Any], complete: bool) -> AsyncMock:
    check = AsyncMock(return_value=complete)
    world["monkeypatch"].setattr(
        services.career_paths_api, "is_version_complete_for_user", check
    )
    return check


async def test_a_default_path_the_student_already_finished_is_left_unstarted(
    default_world: dict[str, Any],
) -> None:
    """The enrollment stands and waits, exactly as at the path ceiling."""
    world = default_world
    _budget(world, used=0, limit=10)
    _already_completed(world, True)

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is False
    assert world["db"].added == []
    assert world["enrollment"].status == "awaiting_path"


async def test_a_default_path_still_starts_when_it_is_not_finished(
    default_world: dict[str, Any],
) -> None:
    """The guard must not cost every student their default path."""
    world = default_world
    _budget(world, used=0, limit=10)
    _already_completed(world, False)

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is True
    assert world["enrollment"].status == "active"


async def test_completion_is_judged_on_the_version_this_program_pinned(
    default_world: dict[str, Any],
) -> None:
    """Not on the career path as a whole.

    Two programs can pin different versions of one path, and finishing v1
    does not finish v2 -- ``complete_program_attempts`` says so explicitly.
    This guard has to ask the same question of the same version, or it
    refuses a default the sweep would never have completed.
    """
    world = default_world
    _budget(world, used=0, limit=10)
    check = _already_completed(world, False)

    await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert (
        check.await_args.kwargs["version_id"]
        == world["default_path"]["career_path_version_id"]
    )
    assert check.await_args.kwargs["student_id"] == world["enrollment"].student_id


async def test_a_skipped_default_grants_no_course_access(
    default_world: dict[str, Any],
) -> None:
    """No attempt, no entitlements. Granting access to a path the student was
    not enrolled onto would leave courses open that nothing accounts for."""
    world = default_world
    _budget(world, used=0, limit=10)
    _already_completed(world, True)
    grant = AsyncMock()
    world["monkeypatch"].setattr(
        services.career_paths_api, "ensure_program_path_access", grant
    )

    await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    grant.assert_not_awaited()

# --------------------------------------------------------------------------
# what the budget deliberately does NOT gate
# --------------------------------------------------------------------------


async def test_a_path_change_is_not_gated_because_it_is_net_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One out, one in. A student at the ceiling must still be able to switch.

    If the budget gated this, lowering the setting would leave grandfathered
    students permanently stuck on whatever path they happened to hold.
    """
    student_id, target_path_id = uuid4(), uuid4()
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
    version = SimpleNamespace(
        id=enrollment.program_version_id,
        max_path_switches=3,
        learning_program_id=enrollment.learning_program_id,
    )
    program = SimpleNamespace(
        id=enrollment.learning_program_id, name="Program B", organization_id=uuid4()
    )

    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_notify_owning_deans", AsyncMock())
    # The response model reads timestamps the DB would have filled in; this
    # test is about whether the row is written at all.
    monkeypatch.setattr(
        services, "PathChangeRequestRead", SimpleNamespace(model_validate=lambda row: row)
    )
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Path 3"))
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(services.queries, "get_version", AsyncMock(return_value=version))
    monkeypatch.setattr(services.queries, "list_active_attempts", AsyncMock(return_value=[attempt]))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[attempt]))
    monkeypatch.setattr(services.queries, "get_pending_request", AsyncMock(return_value=None))
    monkeypatch.setattr(services.queries, "count_approved_switches", AsyncMock(return_value=0))
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
    # Hard over the ceiling, and it must not matter.
    over_budget = AsyncMock(return_value=9)
    monkeypatch.setattr(services.queries, "count_active_paths_for_student", over_budget)
    monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=1))

    db = _recording_db()
    await services.request_path_change(
        db,
        enrollment_id=enrollment.id,
        target_path_id=target_path_id,
        reason="The data track fits my internship.",
        student_id=student_id,
    )

    assert len(db.added) == 1, "the change request must be filed"
    over_budget.assert_not_awaited()
