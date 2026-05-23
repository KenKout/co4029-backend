from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.pagination import (
    CursorPage,
    decode_composite_cursor,
    encode_composite_cursor,
)
from abridgeai.features.career_paths.models import StudentCareerEnrollment
from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.queries import student as student_queries
from abridgeai.features.career_paths.schemas import (
    CareerPathProgressRead,
    CareerPathPublic,
    CourseProgressSummary,
    MyCareerEnrollmentRead,
    StudentCareerEnrollmentAuthoring,
    StudentPathProgressAuthoring,
)
from abridgeai.features.career_paths.schemas.public import CareerPathCoursePublic

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser
    from abridgeai.features.career_paths.models import CareerPath


def _to_path_public(path: CareerPath, courses: list[dict[str, object]]) -> CareerPathPublic:
    return CareerPathPublic.model_validate(
        {
            "id": path.id,
            "slug": path.slug,
            "name": path.name,
            "description": path.description,
            "status": "published",
            "courses": [
                CareerPathCoursePublic.model_validate(
                    {
                        "course_id": row["course_id"],
                        "slug": row["course_slug"],
                        "title": row["course_title"],
                        "position": row["position"],
                        "is_required": row["is_required"],
                    }
                )
                for row in courses
            ],
        }
    )


def _to_authoring_enrollment(
    enrollment: StudentCareerEnrollment,
) -> StudentCareerEnrollmentAuthoring:
    return StudentCareerEnrollmentAuthoring.model_validate(
        {
            "id": enrollment.id,
            "career_path_id": enrollment.career_path_id,
            "student_id": enrollment.student_id,
            "status": enrollment.status,
            "started_at": enrollment.started_at,
            "completed_at": enrollment.completed_at,
            "created_at": enrollment.created_at,
            "updated_at": enrollment.updated_at,
            "created_by": enrollment.created_by,
            "updated_by": enrollment.updated_by,
        }
    )


async def enroll_student_in_path(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    student_id: UUID,
    actor: CurrentUser,
) -> StudentCareerEnrollmentAuthoring:
    path = await authoring_queries.get_career_path_for_authoring(db, career_path_id)
    if path is None or path.deleted_at is not None:
        raise NotFoundError(f"CareerPath {career_path_id} not found")

    existing = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    if existing is not None and existing.status != "dropped":
        raise AppError(f"Student {student_id} already enrolled in path {career_path_id}")
    if existing is not None and existing.status == "dropped":
        existing.status = "active"
        existing.completed_at = None
        existing.started_at = datetime.now(tz=UTC)
        existing.updated_by = actor.user_id
        await flush_or_conflict(db)
        return _to_authoring_enrollment(existing)

    enrollment = StudentCareerEnrollment(
        career_path_id=career_path_id,
        student_id=student_id,
        status="active",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(enrollment)
    await flush_or_conflict(db)
    return _to_authoring_enrollment(enrollment)


async def unenroll_student(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    student_id: UUID,
    actor: CurrentUser,
) -> StudentCareerEnrollmentAuthoring:
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    if enrollment is None:
        raise NotFoundError(f"No enrollment for path={career_path_id} student={student_id}")
    enrollment.status = "dropped"
    enrollment.updated_by = actor.user_id
    await flush_or_conflict(db)
    return _to_authoring_enrollment(enrollment)


async def list_my_career_enrollments(
    db: AsyncSession, student_id: UUID
) -> list[MyCareerEnrollmentRead]:
    rows = await student_queries.list_my_career_enrollments(db, student_id)
    return [MyCareerEnrollmentRead.model_validate(row) for row in rows]


async def get_my_path_progress(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID
) -> CareerPathProgressRead:
    rows = await student_queries.get_path_course_progress(
        db, career_path_id=career_path_id, student_id=student_id
    )
    courses = [
        CourseProgressSummary(
            course_id=row["course_id"],
            slug=row["slug"],
            title=row["title"],
            status=row["status"],
            completion_percent=float(row["completion_percent"]),
        )
        for row in rows
    ]
    course_count = len(courses)
    completed = sum(1 for c in courses if c.completion_percent >= 100)
    in_progress = sum(1 for c in courses if 0 < c.completion_percent < 100)
    overall = sum(c.completion_percent for c in courses) / course_count if course_count else 0.0
    return CareerPathProgressRead(
        career_path_id=career_path_id,
        overall_percent=overall,
        course_count=course_count,
        completed_courses=completed,
        in_progress_courses=in_progress,
        courses=courses,
    )


async def get_published_path_with_courses(
    db: AsyncSession, *, slug: str, organization_id: UUID
) -> CareerPathPublic | None:
    from abridgeai.features.career_paths.queries import (
        get_published_career_path_by_slug,
        list_published_career_path_courses,
    )

    path = await get_published_career_path_by_slug(db, slug=slug, organization_id=organization_id)
    if path is None:
        return None
    courses = await list_published_career_path_courses(db, path.id)
    return _to_path_public(path, courses)


async def get_published_path_for_user(
    db: AsyncSession, *, slug: str, user_id: UUID
) -> CareerPathPublic | None:
    from abridgeai.features.career_paths.queries import get_user_primary_organization_id

    organization_id = await get_user_primary_organization_id(db, user_id)
    if organization_id is None:
        return None
    return await get_published_path_with_courses(db, slug=slug, organization_id=organization_id)


async def list_published_paths(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int,
    cursor: str | None,
) -> CursorPage[CareerPathPublic]:
    """Cursor-paginated published career paths ordered by ``(created_at DESC, id DESC)``."""
    from abridgeai.features.career_paths.queries import (
        list_published_career_path_courses,
        list_published_career_paths,
    )

    after_created_at: datetime | None = None
    after_id: UUID | None = None
    if cursor:
        sort_value, last_id = decode_composite_cursor(cursor)
        if not isinstance(sort_value, datetime):
            raise ValueError("Invalid cursor")
        after_created_at = sort_value
        after_id = last_id

    paths = await list_published_career_paths(
        db,
        organization_id=organization_id,
        limit=limit,
        after_created_at=after_created_at,
        after_id=after_id,
    )
    results: list[CareerPathPublic] = []
    for path in paths:
        courses = await list_published_career_path_courses(db, path.id)
        results.append(_to_path_public(path, courses))
    next_cursor = (
        encode_composite_cursor(paths[-1].created_at, paths[-1].id)
        if len(paths) == limit
        else None
    )
    return CursorPage(items=results, next_cursor=next_cursor)


async def list_published_paths_for_user(
    db: AsyncSession,
    *,
    user_id: UUID,
    limit: int,
    cursor: str | None,
) -> CursorPage[CareerPathPublic]:
    from abridgeai.features.career_paths.queries import get_user_primary_organization_id

    organization_id = await get_user_primary_organization_id(db, user_id)
    if organization_id is None:
        return CursorPage(items=[], next_cursor=None)
    return await list_published_paths(
        db, organization_id=organization_id, limit=limit, cursor=cursor
    )


async def get_roster_progress(
    db: AsyncSession, career_path_id: UUID
) -> list[StudentPathProgressAuthoring]:
    rows = await student_queries.get_roster_path_progress(db, career_path_id)
    return [
        StudentPathProgressAuthoring.model_validate(
            {
                "student_id": row["student_id"],
                "student_email": row["primary_email"],
                "overall_percent": float(row["overall_percent"]),
                "completed_courses": int(row["completed_courses"]),
                "course_count": int(row["course_count"]),
            }
        )
        for row in rows
    ]


__all__ = [
    "enroll_student_in_path",
    "get_my_path_progress",
    "get_published_path_for_user",
    "get_published_path_with_courses",
    "get_roster_progress",
    "list_my_career_enrollments",
    "list_published_paths",
    "list_published_paths_for_user",
    "unenroll_student",
]
