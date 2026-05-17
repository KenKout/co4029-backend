from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.features.career_paths.models import CareerPath, CareerPathCourse
from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.schemas import (
    CareerPathAuthoring,
    CareerPathCourseAuthoring,
    CareerPathCreate,
    CareerPathUpdate,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser

_OFFSET = 100_000


def _to_authoring(path: CareerPath) -> CareerPathAuthoring:
    return CareerPathAuthoring(
        id=path.id,
        organization_id=path.organization_id,
        org_unit_id=path.org_unit_id,
        slug=path.slug,
        name=path.name,
        description=path.description,
        status=path.status,
        created_at=path.created_at,
        updated_at=path.updated_at,
        created_by=path.created_by,
        updated_by=path.updated_by,
        deleted_at=path.deleted_at,
        deleted_by=path.deleted_by,
    )


async def _require_path(db: AsyncSession, career_path_id: UUID) -> CareerPath:
    path = await authoring_queries.get_career_path_for_authoring(db, career_path_id)
    if path is None or path.deleted_at is not None:
        raise NotFoundError(f"CareerPath {career_path_id} not found")
    return path


async def list_career_paths_for_org(
    db: AsyncSession, organization_id: UUID, *, include_archived: bool = False
) -> list[CareerPathAuthoring]:
    rows = await authoring_queries.list_career_paths_for_org(
        db, organization_id, include_archived=include_archived
    )
    return [_to_authoring(row) for row in rows]


async def get_career_path(db: AsyncSession, career_path_id: UUID) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    return _to_authoring(path)


async def list_career_path_courses(
    db: AsyncSession, career_path_id: UUID
) -> list[CareerPathCourseAuthoring]:
    await _require_path(db, career_path_id)
    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


async def create_career_path(
    db: AsyncSession, payload: CareerPathCreate, actor: CurrentUser
) -> CareerPathAuthoring:
    path = CareerPath(
        organization_id=payload.organization_id,
        org_unit_id=payload.org_unit_id,
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
        status="draft",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(path)
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def update_career_path(
    db: AsyncSession,
    career_path_id: UUID,
    payload: CareerPathUpdate,
    actor: CurrentUser,
) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(path, key, value)
    path.updated_by = actor.user_id
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def add_course_to_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    *,
    position: int | None,
    is_required: bool,
    actor: CurrentUser,
) -> CareerPathCourseAuthoring:
    path = await _require_path(db, career_path_id)
    if not await authoring_queries.course_belongs_to_org(db, course_id, path.organization_id):
        raise AppError(f"Course {course_id} does not belong to organization {path.organization_id}")
    existing = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if existing is not None:
        raise AppError(f"Course {course_id} already attached to career path {career_path_id}")
    target_position = position
    if target_position is None:
        target_position = await authoring_queries.next_path_course_position(db, career_path_id)
    else:
        await _make_room_for_position(db, career_path_id, target_position)

    link = CareerPathCourse(
        career_path_id=career_path_id,
        course_id=course_id,
        position=target_position,
        is_required=is_required,
    )
    db.add(link)
    await db.flush()
    del actor
    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    target = next(row for row in rows if row["course_id"] == course_id)
    return CareerPathCourseAuthoring.model_validate(target)


async def _make_room_for_position(
    db: AsyncSession, career_path_id: UUID, target_position: int
) -> None:
    links = await authoring_queries.list_path_course_links(db, career_path_id)
    affected = [link for link in links if link.position >= target_position]
    if not affected:
        return
    for idx, link in enumerate(affected):
        link.position = _OFFSET + idx
    await db.flush()
    for idx, link in enumerate(affected, start=1):
        link.position = target_position + idx
    await db.flush()


async def remove_course_from_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_id: UUID,
    actor: CurrentUser,
) -> None:
    del actor
    await _require_path(db, career_path_id)
    link = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} not attached to career path {career_path_id}")
    await db.delete(link)
    await db.flush()


async def reorder_courses_in_path(
    db: AsyncSession,
    career_path_id: UUID,
    course_ids: list[UUID],
    actor: CurrentUser,
) -> list[CareerPathCourseAuthoring]:
    del actor
    await _require_path(db, career_path_id)
    existing_links = await authoring_queries.list_path_course_links(db, career_path_id)
    existing_by_course = {link.course_id: link for link in existing_links}
    if set(course_ids) != set(existing_by_course):
        raise AppError(f"reorder course_ids must match existing path courses for {career_path_id}")

    for idx, course_id in enumerate(course_ids):
        existing_by_course[course_id].position = _OFFSET + idx
    await db.flush()

    for idx, course_id in enumerate(course_ids, start=1):
        existing_by_course[course_id].position = idx
    await db.flush()

    rows = await authoring_queries.list_authoring_career_path_courses(db, career_path_id)
    return [CareerPathCourseAuthoring.model_validate(row) for row in rows]


async def publish_path(
    db: AsyncSession, career_path_id: UUID, actor: CurrentUser
) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    if path.status == "archived":
        raise AppError(f"Cannot publish archived career path {career_path_id}")
    path.status = "published"
    path.updated_by = actor.user_id
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def archive_path(
    db: AsyncSession, career_path_id: UUID, actor: CurrentUser
) -> CareerPathAuthoring:
    path = await _require_path(db, career_path_id)
    if path.status == "archived":
        raise AppError(f"CareerPath {career_path_id} is already archived")
    path.status = "archived"
    path.updated_by = actor.user_id
    await db.flush()
    await db.refresh(path)
    return _to_authoring(path)


async def soft_delete_path(db: AsyncSession, career_path_id: UUID, actor: CurrentUser) -> None:
    path = await _require_path(db, career_path_id)
    await soft_delete_cascade(db, path, actor_id=actor.user_id)


__all__ = [
    "add_course_to_path",
    "archive_path",
    "create_career_path",
    "get_career_path",
    "list_career_path_courses",
    "list_career_paths_for_org",
    "publish_path",
    "remove_course_from_path",
    "reorder_courses_in_path",
    "soft_delete_path",
    "update_career_path",
]
