"""One career path may not run in two of a student's programs at once.

Every duplicate guard this feature had before was scoped to a single
program enrollment: the ``uq_program_path_attempts_active_path`` index is
keyed on ``(program_enrollment_id, career_path_id)``, the
``path_already_selected`` checks read only that enrollment's attempts, and
``max_career_paths_per_enrollment`` counts them. None of it can see a second
program.

That was unreachable while ``learning_program.max_concurrent_enrollments``
sat at its default of 1. Raising it opens the gap: two programs that both
offer path P let a student hold two live attempts on P, each one legal in
its own program. Because ``student_career_enrollments`` is keyed
``(student_id, career_path_id)``, the two attempts then share ONE progress
projection — the student does the work once and both programs count it.

So the guard has to be student-wide, and it has to sit on every route that
opens an attempt. There are four:

* the program default, started automatically at enrollment;
* a student picking a path themselves;
* a student *requesting* a switch to one;
* a dean *approving* that switch, which is the authoritative moment —
  the student may have joined another program while the request queued.

Everything below the service is mocked; the DB-backed mechanics live in
``tests/integration/test_learning_programs.py``.
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
    """A stand-in session that remembers every row the service adds."""
    added: list[Any] = []
    return SimpleNamespace(add=added.append, added=added)


# --------------------------------------------------------------------------
# 1. the program default, activated automatically at enrollment
# --------------------------------------------------------------------------


@pytest.fixture
def default_path_world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="awaiting_path",
        updated_by=None,
    )
    default_path = {
        "career_path_id": uuid4(),
        "career_path_version_id": uuid4(),
        "status": "published",
        "is_default": True,
    }
    program = SimpleNamespace(
        id=enrollment.learning_program_id, name="BSc Data Science", organization_id=uuid4()
    )
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services.career_paths_api, "ensure_program_path_access", AsyncMock())
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(
        services.queries, "count_active_paths_for_student", AsyncMock(return_value=0)
    )
    monkeypatch.setattr(
        services.career_paths_api,
        "is_version_complete_for_user",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=10))
    return {
        "db": _recording_db(),
        "enrollment": enrollment,
        "default_path": default_path,
        "actor_id": uuid4(),
    }


async def test_default_path_is_skipped_when_already_running_elsewhere(
    default_path_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joining a second program must not re-open a path already in flight.

    The enrollment is NOT failed. A manager importing a roster should not
    lose the row over this, so the student lands in ``awaiting_path`` and
    picks one of the program's other paths instead.
    """
    world = default_path_world
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value="BSc Data Science"),
    )

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is False
    assert world["db"].added == [], "no attempt row may be written"
    assert world["enrollment"].status == "awaiting_path"
    services.career_paths_api.ensure_program_path_access.assert_not_awaited()


async def test_default_path_starts_normally_when_not_running_elsewhere(
    default_path_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = default_path_world
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value=None),
    )

    started = await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    assert started is True
    assert len(world["db"].added) == 1
    attempt = world["db"].added[0]
    assert attempt.career_path_id == world["default_path"]["career_path_id"]
    assert attempt.selection_source == "program_default"
    assert world["enrollment"].status == "active"


async def test_default_path_guard_excludes_the_enrollment_being_started(
    default_path_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lookup must ignore this enrollment, or a re-enrol could self-block."""
    world = default_path_world
    probe = AsyncMock(return_value=None)
    monkeypatch.setattr(services.queries, "find_active_path_attempt_elsewhere", probe)

    await services._activate_default_path(
        world["db"],
        enrollment=world["enrollment"],
        actor_id=world["actor_id"],
        default_path=world["default_path"],
    )

    kwargs = probe.await_args.kwargs
    assert kwargs["student_id"] == world["enrollment"].student_id
    assert kwargs["career_path_id"] == world["default_path"]["career_path_id"]
    assert kwargs["excluding_enrollment_id"] == world["enrollment"].id


# --------------------------------------------------------------------------
# 2. a student picking a path themselves
# --------------------------------------------------------------------------


@pytest.fixture
def select_world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    student_id = uuid4()
    career_path_id = uuid4()
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=student_id,
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="awaiting_path",
        updated_by=None,
    )
    version = SimpleNamespace(
        id=enrollment.program_version_id,
        max_career_paths_per_enrollment=1,
    )
    program = SimpleNamespace(
        id=enrollment.learning_program_id,
        name="BSc Software Engineering",
        organization_id=uuid4(),
    )

    monkeypatch.setattr(
        services.queries, "count_active_paths_for_student", AsyncMock(return_value=1)
    )
    monkeypatch.setattr(services, "resolve_setting", AsyncMock(return_value=10))
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Data Engineering"))
    monkeypatch.setattr(services, "_enrollment_out", AsyncMock(return_value="ok"))
    monkeypatch.setattr(services.career_paths_api, "ensure_program_path_access", AsyncMock())
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(services.queries, "get_version", AsyncMock(return_value=version))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        services.queries,
        "list_version_paths",
        AsyncMock(
            return_value=[
                {
                    "career_path_id": career_path_id,
                    "career_path_version_id": uuid4(),
                    "status": "published",
                }
            ]
        ),
    )
    return {
        "db": _recording_db(),
        "enrollment": enrollment,
        "career_path_id": career_path_id,
        "student_id": student_id,
    }


async def test_select_path_rejects_a_path_active_in_another_program(
    select_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = select_world
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value="BSc Data Science"),
    )

    with pytest.raises(ConflictError) as excinfo:
        await services.select_path(
            world["db"],
            enrollment_id=world["enrollment"].id,
            career_path_id=world["career_path_id"],
            student_id=world["student_id"],
        )

    err = excinfo.value
    assert err.code == "path_active_in_another_program"
    # The sentence has to name both halves, or the student cannot act on it.
    assert "Data Engineering" in err.message
    assert "BSc Data Science" in err.message
    assert err.fields["conflicting_program_name"] == "BSc Data Science"
    assert err.fields["career_path_id"] == str(world["career_path_id"])
    assert world["db"].added == [], "no attempt row may be written"


async def test_select_path_allows_a_path_not_running_elsewhere(
    select_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = select_world
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value=None),
    )

    await services.select_path(
        world["db"],
        enrollment_id=world["enrollment"].id,
        career_path_id=world["career_path_id"],
        student_id=world["student_id"],
    )

    assert len(world["db"].added) == 1
    assert world["db"].added[0].selection_source == "student"
    assert world["enrollment"].status == "active"


async def test_select_path_guard_is_student_wide_not_enrollment_wide(
    select_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: an enrollment-scoped lookup would never see the other program."""
    world = select_world
    probe = AsyncMock(return_value=None)
    monkeypatch.setattr(services.queries, "find_active_path_attempt_elsewhere", probe)

    await services.select_path(
        world["db"],
        enrollment_id=world["enrollment"].id,
        career_path_id=world["career_path_id"],
        student_id=world["student_id"],
    )

    kwargs = probe.await_args.kwargs
    assert kwargs["student_id"] == world["student_id"]
    assert kwargs["excluding_enrollment_id"] == world["enrollment"].id


# --------------------------------------------------------------------------
# 3. a student requesting a switch  /  4. a dean approving it
# --------------------------------------------------------------------------


async def test_request_path_change_rejects_before_any_dean_is_notified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast: filing the request at all would queue work that cannot land."""
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
    notify_deans = AsyncMock()
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_notify_owning_deans", notify_deans)
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Data Engineering"))
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "list_active_attempts", AsyncMock(return_value=[attempt]))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[attempt]))
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value="BSc Data Science"),
    )

    db = _recording_db()
    with pytest.raises(ConflictError) as excinfo:
        await services.request_path_change(
            db,
            enrollment_id=enrollment.id,
            target_path_id=target_path_id,
            reason="The data track fits my internship.",
            student_id=student_id,
        )

    assert excinfo.value.code == "path_active_in_another_program"
    assert db.added == [], "no PathChangeRequest row may be written"
    notify_deans.assert_not_awaited()


async def test_approval_rechecks_because_the_student_may_have_joined_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request was legal when filed; the dean decides later.

    Between filing and approval the student can enrol in another program
    that starts the same target path by default. Approval is the moment
    that actually opens the attempt, so it re-checks rather than trusting
    the request.
    """
    student_id, dean_id = uuid4(), uuid4()
    enrollment = SimpleNamespace(
        id=uuid4(),
        student_id=student_id,
        learning_program_id=uuid4(),
        program_version_id=uuid4(),
        status="active",
        updated_by=None,
    )
    attempt = SimpleNamespace(
        id=uuid4(),
        program_enrollment_id=enrollment.id,
        career_path_id=uuid4(),
        status="active",
        exit_snapshot=None,
        ended_at=None,
        updated_by=None,
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
    )
    program = SimpleNamespace(id=enrollment.learning_program_id, name="BSc Software Engineering")

    monkeypatch.setattr(services, "_require_owner_dean", AsyncMock())
    monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(services, "_target_path_name", AsyncMock(return_value="Data Engineering"))
    monkeypatch.setattr(services, "_subject_path_name", AsyncMock(return_value="Data Engineering"))
    monkeypatch.setattr(services.notify, "notify_path_change_approved", AsyncMock())
    monkeypatch.setattr(services.queries, "get_change_request", AsyncMock(return_value=request))
    monkeypatch.setattr(services.queries, "get_enrollment", AsyncMock(return_value=enrollment))
    monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
    monkeypatch.setattr(services.queries, "get_attempt", AsyncMock(return_value=attempt))
    monkeypatch.setattr(services.queries, "list_attempts", AsyncMock(return_value=[attempt]))
    monkeypatch.setattr(
        services.queries,
        "find_active_path_attempt_elsewhere",
        AsyncMock(return_value="BSc Data Science"),
    )

    db = _recording_db()
    with pytest.raises(ConflictError) as excinfo:
        await services.decide_change_request(
            db,
            request_id=request.id,
            approve=True,
            decision_reason=None,
            actor=SimpleNamespace(user_id=dean_id),
        )

    assert excinfo.value.code == "path_active_in_another_program"
    assert db.added == [], "no replacement attempt may be written"
    assert request.status == "pending", "the request must survive for the dean to reject"
    assert attempt.status == "active", "the source path must not be switched out"
