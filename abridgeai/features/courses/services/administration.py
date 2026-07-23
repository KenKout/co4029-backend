"""IT Admin-side service for the courses aggregate (plan §4214).

Composes :mod:`features.courses.queries.administration` and serializes
results into the existing authoring / response Pydantic schemas.
Per the import-linter contract this module never imports sqlalchemy at
module level; ``AsyncSession`` is type-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.pagination import Page
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.queries import (
    CursorPage,
)
from abridgeai.features.courses.queries import (
    administration as admin_queries,
)
from abridgeai.features.courses.schemas import CourseAuthoring

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_all_courses_admin(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    limit: int = 20,
    cursor: str | None = None,
) -> CursorPage[CourseAuthoring]:
    """Cursor-paginated admin list of every course (plan §4215).

    ``include_deleted=True`` lifts the T0.7 soft-delete loader filter via
    ``execution_options(include_deleted=True)`` on the underlying query.
    """
    page = await admin_queries.list_all_courses_admin(
        db, include_deleted=include_deleted, limit=limit, cursor=cursor
    )
    return CursorPage(
        items=[CourseAuthoring.model_validate(course) for course in page.items],
        next_cursor=page.next_cursor,
    )


async def search_all_courses_admin(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    status: str | None = None,
    search: str | None = None,
    sort: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 25,
) -> Page[CourseAuthoring]:
    """Offset page of courses (server-side search + whitelisted sort) as
    ``CourseAuthoring``. Thin delegate to the query layer, which owns the
    SQLAlchemy statement and soft-delete handling."""
    result = await admin_queries.search_all_courses_admin(
        db,
        include_deleted=include_deleted,
        status=status,
        search=search,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    return Page(
        items=[CourseAuthoring.model_validate(c) for c in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


async def restore_soft_deleted_course(
    db: AsyncSession, course_id: UUID, actor: CurrentUser
) -> CourseAuthoring:
    """Clear ``deleted_at`` / ``deleted_by`` on a soft-deleted course.

    Children (modules, lessons, ...) keep their current state — see the
    administration query module's docstring for the rationale.
    """
    restored = await admin_queries.restore_soft_deleted_course(db, course_id)
    if not restored:
        raise NotFoundError(f"No soft-deleted course {course_id} to restore")

    await admin_queries.stamp_updated_by(db, course_id, actor.user_id)

    course = await admin_queries.get_course_including_deleted(db, course_id)
    if course is None:  # pragma: no cover  -- restore_soft_deleted_course returned True
        raise NotFoundError(f"Course {course_id} not found after restore")
    return CourseAuthoring.model_validate(course)


async def soft_delete_course(
    db: AsyncSession, course_id: UUID, actor: CurrentUser
) -> CourseAuthoring:
    """Soft-delete a course (+ its module/lesson/item subtree) via cascade.

    Reversible: the tombstone set here is exactly what
    :func:`restore_soft_deleted_course` lifts. Consistent with the
    project-wide no-HARD-delete invariant — nothing is physically removed,
    the row is just stamped ``deleted_at`` / ``deleted_by`` and filtered
    out of every non-admin ``Course`` SELECT by the T0.7 loader.

    Returns the tombstoned course (fetched with the soft-delete filter
    lifted so the response reflects the new ``deleted_at``). Raises
    ``NotFoundError`` when the course does not exist or is already
    soft-deleted (idempotent guard — nothing to delete).
    """
    course = await admin_queries.get_course_including_deleted(db, course_id)
    if course is None or course.deleted_at is not None:
        raise NotFoundError(f"No active course {course_id} to delete")

    await soft_delete_cascade(db, course, actor_id=actor.user_id)

    refreshed = await admin_queries.get_course_including_deleted(db, course_id)
    if refreshed is None:  # pragma: no cover -- just soft-deleted, still present
        raise NotFoundError(f"Course {course_id} not found after delete")
    return CourseAuthoring.model_validate(refreshed)


async def get_course_processing_audit(db: AsyncSession, course_id: UUID) -> dict[str, Any]:
    """Aggregate AI processing audit for ``course_id`` (plan §4217).

    Joined view across ``ai_model_calls`` + ``processing_jobs`` +
    ``generation_runs`` filtered by ``course_id`` — returns total cost,
    token totals, call count, and distinct pipeline / job / run counts.
    """
    return await admin_queries.get_course_processing_audit(db, course_id)


async def list_course_processing_jobs(
    db: AsyncSession, course_id: UUID, *, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent processing_jobs rows tied to course_id."""
    return await admin_queries.list_course_processing_jobs(db, course_id, limit=limit)


async def get_course_stats(db: AsyncSession, *, top_draft_owners_limit: int = 10) -> dict[str, Any]:
    """Org-wide course aggregates for the admin dashboard."""
    return await admin_queries.get_course_stats(db, top_draft_owners_limit=top_draft_owners_limit)


__all__ = [
    "get_course_processing_audit",
    "get_course_stats",
    "list_all_courses_admin",
    "list_course_processing_jobs",
    "restore_soft_deleted_course",
    "soft_delete_course",
]
