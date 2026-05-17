from __future__ import annotations

from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.courses.models import (
    Course,
    Lesson,
    LessonResource,
    Module,
)
from abridgeai.features.courses.queries._pagination import (
    CursorPage,
    decode_cursor,
    encode_cursor,
)
from abridgeai.features.courses.visibility import (
    published_course_clause,
    published_lesson_clause,
    published_module_clause,
    student_visible_resource_clause,
)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _clamp(limit: int) -> int:
    return min(max(limit, 1), _MAX_LIMIT)


_PUBLISHED_CONTENT_TREE_SQL = text(
    resources.files("abridgeai.features.courses.queries.sql")
    .joinpath("published_content_tree.sql")
    .read_text(encoding="utf-8")
)


async def list_published_courses(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Cursor-paginated published courses scoped to an organization.

    Reconciliation §A10/§D2: cursor pagination, not offset.
    Visibility composes :func:`published_course_clause` from T3.3.
    """
    capped = _clamp(limit)
    after = decode_cursor(cursor) if cursor else None
    stmt = (
        select(Course)
        .where(
            published_course_clause(),
            Course.organization_id == organization_id,
        )
        .order_by(Course.id)
        .limit(capped)
    )
    if after is not None:
        stmt = stmt.where(Course.id > after)
    rows = list((await db.execute(stmt)).scalars().all())
    next_cursor = encode_cursor(rows[-1].id) if len(rows) == capped else None
    return CursorPage(items=rows, next_cursor=next_cursor)


async def list_enrolled_courses(
    db: AsyncSession,
    user_id: UUID,
    *,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Active enrollments → published courses for a student.

    The ``course_enrollments`` table is present in the baseline schema
    (T0.9 migration 0001) but its ORM model lives in
    ``features/enrollments`` which ports in Phase 7. Until then we
    resolve enrolled course ids via raw SQL and re-load Course rows
    through the ORM so the soft-delete loader-criteria still fires.
    """
    capped = _clamp(limit)
    after = decode_cursor(cursor) if cursor else None
    after_clause = "AND c.id > :after_id" if after is not None else ""
    sql = text(
        f"""
        SELECT c.id
        FROM course_enrollments e
        JOIN courses c ON c.id = e.course_id
        WHERE e.student_id = :student_id
          AND e.status = 'active'
          AND c.status = 'published'
          AND c.deleted_at IS NULL
          {after_clause}
        ORDER BY c.id
        LIMIT :limit
        """  # noqa: S608  -- after_clause is a checked-in literal fragment
    )
    params: dict[str, Any] = {"student_id": user_id, "limit": capped}
    if after is not None:
        params["after_id"] = after
    rows = (await db.execute(sql, params)).all()
    course_ids = [row.id for row in rows]
    if not course_ids:
        return CursorPage(items=[], next_cursor=None)
    courses_stmt = select(Course).where(Course.id.in_(course_ids)).order_by(Course.id)
    courses = list((await db.execute(courses_stmt)).scalars().all())
    next_cursor = encode_cursor(courses[-1].id) if len(courses) == capped else None
    return CursorPage(items=courses, next_cursor=next_cursor)


async def get_published_course_by_slug(
    db: AsyncSession, slug: str, organization_id: UUID
) -> Course | None:
    """Org-scoped slug lookup (Reconciliation §A11).

    Slug uniqueness is per-organization (partial unique index
    ``uq_courses_org_slug`` from migration 0002). Two orgs may both own
    a course slug ``"intro-to-python"``; this query MUST be passed an
    ``organization_id`` so the wrong org's course never leaks.
    """
    stmt = select(Course).where(
        Course.slug == slug,
        Course.organization_id == organization_id,
        published_course_clause(),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_published_course_by_id(db: AsyncSession, course_id: UUID) -> Course | None:
    stmt = select(Course).where(Course.id == course_id, published_course_clause())
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_published_course_content(db: AsyncSession, course_id: UUID) -> dict[str, Any] | None:
    """Recursive-CTE fetch of the published course tree.

    Returns ``{"course": dict, "modules": list, "items": list}`` or
    ``None`` when the course is missing / unpublished / soft-deleted.
    DRAFT_VISIBILITY rule (plan §4153): items pointing to draft / soft-
    deleted lessons are excluded entirely (not nullified).
    """
    result = await db.execute(_PUBLISHED_CONTENT_TREE_SQL, {"course_id": course_id})
    row = result.one_or_none()
    if row is None or row.course is None:
        return None
    return {"course": row.course, "modules": row.modules, "items": row.items}


async def list_published_modules(db: AsyncSession, course_id: UUID) -> list[Module]:
    stmt = (
        select(Module)
        .join(Course, Course.id == Module.course_id)
        .where(
            Course.id == course_id,
            published_course_clause(),
            published_module_clause(),
        )
        .order_by(Module.position)
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_published_lessons(db: AsyncSession, module_id: UUID) -> list[Lesson]:
    stmt = (
        select(Lesson)
        .join(Module, Module.id == Lesson.module_id)
        .where(
            Module.id == module_id,
            published_module_clause(),
            published_lesson_clause(),
        )
    )
    return list((await db.execute(stmt)).scalars().all())


async def list_visible_lesson_resources(db: AsyncSession, lesson_id: UUID) -> list[LessonResource]:
    """Resources that are both attached to a published lesson AND have
    ``visible_to_students = TRUE`` (T3.3 :func:`student_visible_resource_clause`).
    """
    stmt = (
        select(LessonResource)
        .join(Lesson, Lesson.id == LessonResource.lesson_id)
        .where(
            LessonResource.lesson_id == lesson_id,
            published_lesson_clause(),
            student_visible_resource_clause(),
        )
        .order_by(LessonResource.position)
    )
    return list((await db.execute(stmt)).scalars().all())


__all__ = [
    "get_published_course_by_id",
    "get_published_course_by_slug",
    "get_published_course_content",
    "list_enrolled_courses",
    "list_published_courses",
    "list_published_lessons",
    "list_published_modules",
    "list_visible_lesson_resources",
]
