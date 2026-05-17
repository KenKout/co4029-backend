"""IT Admin-side service for the courses aggregate (plan §4214).

Composes :mod:`features.courses.queries.administration` and serializes
results into the existing authoring / response Pydantic schemas.
Per the import-linter contract this module never imports sqlalchemy at
module level; ``AsyncSession`` is type-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
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


async def restore_soft_deleted_course(
    db: AsyncSession, course_id: UUID, actor: CurrentUser
) -> CourseAuthoring:
    """Clear ``deleted_at`` / ``deleted_by`` on a soft-deleted course.

    Children (modules, lessons, ...) keep their current state — see the
    administration query module's docstring for the rationale.
    """
    del actor
    restored = await admin_queries.restore_soft_deleted_course(db, course_id)
    if not restored:
        raise NotFoundError(f"No soft-deleted course {course_id} to restore")

    course = await admin_queries.get_course_including_deleted(db, course_id)
    if course is None:  # pragma: no cover  -- restore_soft_deleted_course returned True
        raise NotFoundError(f"Course {course_id} not found after restore")
    return CourseAuthoring.model_validate(course)


async def get_course_processing_audit(db: AsyncSession, course_id: UUID) -> dict[str, Any]:
    """Aggregate AI processing audit for ``course_id`` (plan §4217).

    Joined view across ``ai_model_calls`` + ``processing_jobs`` +
    ``generation_runs`` filtered by ``course_id`` — returns total cost,
    token totals, call count, and distinct pipeline / job / run counts.
    """
    return await admin_queries.get_course_processing_audit(db, course_id)


__all__ = [
    "get_course_processing_audit",
    "list_all_courses_admin",
    "restore_soft_deleted_course",
]
