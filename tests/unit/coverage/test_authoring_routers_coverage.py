from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.features.career_paths.routers import authoring as career_router
from abridgeai.features.courses.routers import authoring as course_router


def _user() -> SimpleNamespace:
    return SimpleNamespace(user_id=uuid4())


def _db() -> SimpleNamespace:
    return SimpleNamespace(commit=AsyncMock())


@pytest.mark.parametrize(
    ("factory", "status_code", "error"),
    [
        (career_router._not_found, 404, "not_found"),
        (career_router._conflict, 409, "conflict"),
        (career_router._bad_request, 400, "bad_request"),
        (course_router._not_found, 404, "not_found"),
        (course_router._conflict, 409, "conflict"),
        (course_router._bad_request, 400, "bad_request"),
        (course_router._forbidden, 403, "permission_denied"),
    ],
)
def test_router_error_factories(factory: object, status_code: int, error: str) -> None:
    exc = factory("message")
    assert exc.status_code == status_code
    assert exc.detail == {"error": error, "message": "message"}


@pytest.mark.asyncio
async def test_path_org_guard_uses_path_owner_and_requested_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_id, org_id = uuid4(), uuid4()
    monkeypatch.setattr(
        career_router.authoring_service,
        "get_career_path",
        AsyncMock(return_value=SimpleNamespace(organization_id=org_id)),
    )
    access = AsyncMock()
    monkeypatch.setattr(career_router, "require_org_access", access)
    await career_router._ensure_caller_in_path_org(
        object(), _user(), path_id, ("course.read",)
    )
    assert access.await_args.kwargs["resource_id"] == path_id
    assert access.await_args.kwargs["permissions"] == ("course.read",)


@pytest.mark.asyncio
async def test_list_paths_requires_primary_org(monkeypatch: pytest.MonkeyPatch) -> None:
    from abridgeai.features.career_paths.queries import published

    monkeypatch.setattr(
        published,
        "get_user_primary_organization_id",
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException) as caught:
        await career_router.list_career_paths(_user(), object())
    assert caught.value.status_code == 400


@pytest.mark.asyncio
async def test_list_paths_checks_explicit_org_and_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id = uuid4()
    access = AsyncMock()
    listing = AsyncMock(return_value=["path"])
    monkeypatch.setattr(career_router, "require_org_access", access)
    monkeypatch.setattr(
        career_router.authoring_service,
        "list_career_paths_for_org",
        listing,
    )
    db = object()
    assert await career_router.list_career_paths(
        _user(), db, organization_id=org_id, include_archived=True
    ) == ["path"]
    access.assert_awaited_once()
    listing.assert_awaited_once_with(db, org_id, include_archived=True)


@pytest.mark.asyncio
async def test_create_path_maps_conflict_and_commits_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AsyncMock(side_effect=[ConflictError("slug exists"), "created"])
    monkeypatch.setattr(career_router.authoring_service, "create_career_path", service)
    db = _db()
    with pytest.raises(HTTPException) as caught:
        await career_router.create_career_path(object(), _user(), db)
    assert caught.value.status_code == 409

    assert await career_router.create_career_path(object(), _user(), db) == "created"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (NotFoundError("missing"), 404),
        (ConflictError("duplicate"), 409),
        (AppError("invalid"), 400),
    ],
)
async def test_create_stage_maps_domain_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status_code: int
) -> None:
    monkeypatch.setattr(career_router, "_ensure_caller_in_path_org", AsyncMock())
    monkeypatch.setattr(
        career_router.authoring_service,
        "create_stage",
        AsyncMock(side_effect=error),
    )
    with pytest.raises(HTTPException) as caught:
        await career_router.create_stage(uuid4(), object(), _user(), _db())
    assert caught.value.status_code == status_code


@pytest.mark.asyncio
async def test_direct_path_enrollment_routes_are_disabled() -> None:
    with pytest.raises(HTTPException) as enrolled:
        await career_router.enroll_student_in_path(
            uuid4(), object(), _user(), object()
        )
    with pytest.raises(HTTPException) as unenrolled:
        await career_router.unenroll_student_from_path(
            uuid4(), uuid4(), _user(), object()
        )
    assert enrolled.value.status_code == 409
    assert "Learning Program" in enrolled.value.detail["message"]
    assert unenrolled.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("function_name", "service_name", "error", "status_code"),
    [
        ("publish_path", "publish_path", NotFoundError("missing"), 404),
        ("publish_path", "publish_path", AppError("invalid"), 400),
        ("archive_path", "archive_path", AppError("active"), 409),
    ],
)
async def test_path_lifecycle_maps_service_errors(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    service_name: str,
    error: Exception,
    status_code: int,
) -> None:
    monkeypatch.setattr(career_router, "_ensure_caller_in_path_org", AsyncMock())
    monkeypatch.setattr(
        career_router.authoring_service,
        service_name,
        AsyncMock(side_effect=error),
    )
    with pytest.raises(HTTPException) as caught:
        await getattr(career_router, function_name)(uuid4(), _user(), _db())
    assert caught.value.status_code == status_code


@pytest.mark.asyncio
async def test_course_create_maps_errors_and_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AsyncMock(
        side_effect=[ConflictError("slug"), AppError("org"), "created"]
    )
    monkeypatch.setattr(course_router.authoring_service, "create_course", service)
    db = _db()
    with pytest.raises(HTTPException) as conflict:
        await course_router.create_course(object(), _user(), db)
    assert conflict.value.status_code == 409
    with pytest.raises(HTTPException) as bad_request:
        await course_router.create_course(object(), _user(), db)
    assert bad_request.value.status_code == 400
    assert await course_router.create_course(object(), _user(), db) == "created"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_course_rejects_manager_fields_without_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = SimpleNamespace(model_fields_set={"title"})
    monkeypatch.setattr(course_router, "load_course_permissions", AsyncMock(return_value=set()))
    with pytest.raises(HTTPException) as caught:
        await course_router.update_course(uuid4(), payload, _user(), _db())
    assert caught.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (NotFoundError("missing"), 404),
        (ConflictError("slug"), 409),
        (AppError("faculty"), 400),
    ],
)
async def test_update_course_maps_domain_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status_code: int
) -> None:
    payload = SimpleNamespace(model_fields_set={"description"})
    monkeypatch.setattr(
        course_router.authoring_service,
        "update_course",
        AsyncMock(side_effect=error),
    )
    with pytest.raises(HTTPException) as caught:
        await course_router.update_course(uuid4(), payload, _user(), _db())
    assert caught.value.status_code == status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (NotFoundError("missing"), 404),
        (ConflictError("no gradeable item"), 409),
        (AppError("invalid"), 400),
    ],
)
async def test_publish_course_maps_domain_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status_code: int
) -> None:
    monkeypatch.setattr(
        course_router.authoring_service,
        "publish_course",
        AsyncMock(side_effect=error),
    )
    with pytest.raises(HTTPException) as caught:
        await course_router.publish_course(uuid4(), _user(), _db())
    assert caught.value.status_code == status_code


@pytest.mark.asyncio
async def test_review_queue_not_found_is_http_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        course_router.authoring_service,
        "list_review_queue_items",
        AsyncMock(side_effect=NotFoundError("unknown kind")),
    )
    with pytest.raises(HTTPException) as caught:
        await course_router.list_review_queue_items(
            "materials", _user(), object()
        )
    assert caught.value.status_code == 404
