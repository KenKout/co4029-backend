from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.features.enrollments.schemas import (
    BulkEnrollRequest,
    EnrollmentPatch,
    InvitationCodeCreate,
)
from abridgeai.features.enrollments.services import manager


@pytest.mark.asyncio
async def test_resolve_user_ids_reports_missing_duplicate_and_non_student(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct, from_email = uuid4(), uuid4()
    payload = BulkEnrollRequest(
        user_ids=[direct],
        emails=["found@example.com", "missing@example.com"],
    )
    monkeypatch.setattr(
        manager.authoring_queries,
        "lookup_users_by_email",
        AsyncMock(
            return_value=[
                {"primary_email": "found@example.com", "id": str(from_email)}
            ]
        ),
    )
    monkeypatch.setattr(
        manager.access_control_api,
        "get_role_codes_for_users",
        AsyncMock(return_value={direct: ("teacher",), from_email: ("student",)}),
    )

    resolved, failures = await manager._resolve_user_ids(object(), payload)

    assert resolved == [from_email]
    assert {(item.identifier, item.reason) for item in failures} == {
        ("missing@example.com", "user_not_found"),
        (str(direct), "not_student"),
    }


@pytest.mark.asyncio
async def test_ensure_enrollment_is_idempotent_reactivates_or_creates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = SimpleNamespace(status="active")
    dropped = SimpleNamespace(status="dropped", dropped_at=object(), updated_by=None)
    monkeypatch.setattr(
        manager.authoring_queries,
        "find_enrollment",
        AsyncMock(side_effect=[active, dropped, None]),
    )
    flush = AsyncMock()
    create = AsyncMock(return_value=SimpleNamespace(status="active"))
    monkeypatch.setattr(manager, "flush_or_conflict", flush)
    monkeypatch.setattr(manager, "_create_enrollment", create)
    actor_id, course_id, student_id = uuid4(), uuid4(), uuid4()

    assert (
        await manager.ensure_enrollment(
            object(), course_id=course_id, student_id=student_id, actor_id=actor_id
        )
        is active
    )
    assert (
        await manager.ensure_enrollment(
            object(), course_id=course_id, student_id=student_id, actor_id=actor_id
        )
        is dropped
    )
    assert dropped.status == "active"
    assert dropped.dropped_at is None
    flush.assert_awaited_once()
    created = await manager.ensure_enrollment(
        object(), course_id=course_id, student_id=student_id, actor_id=actor_id
    )
    assert created.status == "active"
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_notify_enrolled_only_for_published_course(monkeypatch: pytest.MonkeyPatch) -> None:
    course_id, student_id = uuid4(), uuid4()
    get_course = AsyncMock(
        side_effect=[None, SimpleNamespace(status="draft"), SimpleNamespace(status="published", title="SQL", slug="sql")]
    )
    notify = AsyncMock()
    monkeypatch.setattr(manager.courses_api, "get_course_by_id", get_course)
    monkeypatch.setattr(manager.courses_api, "notify_student_enrolled", notify)

    await manager._notify_enrolled_if_published(
        object(), course_id=course_id, student_ids=[], arq_pool=None
    )
    await manager._notify_enrolled_if_published(
        object(), course_id=course_id, student_ids=[student_id], arq_pool=None
    )
    await manager._notify_enrolled_if_published(
        object(), course_id=course_id, student_ids=[student_id], arq_pool=None
    )
    await manager._notify_enrolled_if_published(
        object(), course_id=course_id, student_ids=[student_id], arq_pool="pool"
    )

    notify.assert_awaited_once()
    assert notify.await_args.kwargs["student_user_id"] == student_id
    assert notify.await_args.kwargs["arq_pool"] == "pool"


@pytest.mark.asyncio
async def test_bulk_enroll_reports_existing_and_reactivates_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_id, dropped_id, new_id = uuid4(), uuid4(), uuid4()
    dropped = SimpleNamespace(status="dropped", dropped_at=object(), updated_by=None)
    monkeypatch.setattr(
        manager,
        "_resolve_user_ids",
        AsyncMock(return_value=([existing_id, dropped_id, new_id], [])),
    )
    monkeypatch.setattr(
        manager.authoring_queries,
        "find_enrollment",
        AsyncMock(side_effect=[SimpleNamespace(status="active"), dropped, None]),
    )
    create = AsyncMock()
    notify = AsyncMock()
    monkeypatch.setattr(manager, "_create_enrollment", create)
    monkeypatch.setattr(manager, "_notify_enrolled_if_published", notify)
    monkeypatch.setattr(manager, "flush_or_conflict", AsyncMock())
    actor = SimpleNamespace(user_id=uuid4())

    result = await manager.bulk_enroll_students(
        object(),
        uuid4(),
        BulkEnrollRequest(user_ids=[existing_id]),
        actor,
    )

    assert result.enrolled == [dropped_id, new_id]
    assert result.failures[0].reason == "already_enrolled"
    assert dropped.status == "active"
    create.assert_awaited_once()
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_patch_enrollment_not_found_and_status_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enrollment_id = uuid4()
    missing_db = SimpleNamespace(get=AsyncMock(return_value=None))
    with pytest.raises(NotFoundError):
        await manager.patch_enrollment(
            missing_db,
            enrollment_id,
            EnrollmentPatch(status="dropped"),
            SimpleNamespace(user_id=uuid4()),
        )

    enrollment = SimpleNamespace(
        dropped_at=None,
        status="active",
        updated_by=None,
    )
    db = SimpleNamespace(get=AsyncMock(return_value=enrollment), refresh=AsyncMock())
    monkeypatch.setattr(manager, "flush_or_conflict", AsyncMock())
    monkeypatch.setattr(manager, "_to_authoring", lambda value: value)
    actor = SimpleNamespace(user_id=uuid4())

    result = await manager.patch_enrollment(
        db, enrollment_id, EnrollmentPatch(status="dropped"), actor
    )
    assert result.status == "dropped"
    assert result.dropped_at is not None

    await manager.patch_enrollment(
        db, enrollment_id, EnrollmentPatch(status="active"), actor
    )
    assert enrollment.dropped_at is None


@pytest.mark.asyncio
async def test_unenroll_missing_and_invitation_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        manager.authoring_queries,
        "find_enrollment",
        AsyncMock(return_value=None),
    )
    with pytest.raises(NotFoundError, match="No enrollment"):
        await manager.unenroll_student(
            object(), uuid4(), uuid4(), SimpleNamespace(user_id=uuid4())
        )

    monkeypatch.setattr(
        manager.authoring_queries,
        "get_course_organization_id",
        AsyncMock(side_effect=[None, uuid4()]),
    )
    actor = SimpleNamespace(user_id=uuid4())
    payload = InvitationCodeCreate(code="JOIN-US")
    with pytest.raises(NotFoundError, match="Course"):
        await manager.create_invitation_code(object(), uuid4(), payload, actor)

    monkeypatch.setattr(
        manager.authoring_queries,
        "find_invitation_code_by_string",
        AsyncMock(return_value=object()),
    )
    with pytest.raises(ConflictError, match="invitation_code_taken"):
        await manager.create_invitation_code(object(), uuid4(), payload, actor)


@pytest.mark.asyncio
async def test_csv_import_missing_course_and_invalid_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    course_id = uuid4()
    monkeypatch.setattr(
        manager.authoring_queries,
        "get_course_organization_id",
        AsyncMock(side_effect=[None, uuid4()]),
    )
    actor = SimpleNamespace(user_id=uuid4())
    with pytest.raises(NotFoundError):
        await manager.bulk_import_students_from_csv(
            object(), course_id, [], actor
        )

    monkeypatch.setattr(manager, "_notify_enrolled_if_published", AsyncMock())
    result = await manager.bulk_import_students_from_csv(
        object(), course_id, [{"email": "invalid"}], actor
    )
    assert result.enrolled == []
    assert result.failures[0].reason.startswith("invalid_row")

