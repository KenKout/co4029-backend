from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
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
    StageProgressRead,
    StartCourseResult,
    StudentCareerEnrollmentAuthoring,
    StudentPathProgressAuthoring,
)
from abridgeai.features.career_paths.schemas.public import CareerPathCoursePublic
from abridgeai.features.career_paths.services import stages as stage_service
from abridgeai.features.enrollments.api import public as enrollments_api

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser
    from abridgeai.features.career_paths.models import CareerPath

_STAGE_AWARE_FORMULA = 2


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
                        "stage_id": row.get("stage_id"),
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
    """Assign a student to a career path (manager action).

    Pattern B (lazy enrollment): this grants access to the PATH only. It
    deliberately does NOT create course enrollments — the eager
    ``_autoenroll_required_courses`` fan-out that used to run here was
    removed, because it enrolled a student in every required course of every
    stage at once, including stages still locked to them. Course enrollments
    are now created one at a time by :func:`start_course_in_path` when the
    student actually starts a course in an unlocked stage.
    """
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
        await db.refresh(existing)
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
    await db.refresh(enrollment)
    return _to_authoring_enrollment(enrollment)


async def sync_enrollment_completion(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    student_id: UUID,
    overall_percent: float,
) -> bool:
    """Flip an ``active`` enrollment to ``completed`` once the path is 100%
    done — the "prepared" milestone. Idempotent; returns ``True`` iff it
    flipped on this call (so the caller knows whether to commit). Caller
    owns the transaction.
    """
    if overall_percent < 100:
        return False
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    if enrollment is None or enrollment.status != "active":
        return False
    enrollment.status = "completed"
    enrollment.completed_at = datetime.now(tz=UTC)
    enrollment.updated_by = student_id
    await flush_or_conflict(db)
    return True


async def list_my_career_enrollments(
    db: AsyncSession, student_id: UUID
) -> list[MyCareerEnrollmentRead]:
    """Enrollments enriched with derived pathway completion + the "prepared"
    flag. Also lazily flips completed enrollments (see
    :func:`sync_enrollment_completion`) — the router commits.
    """
    rows = await student_queries.list_my_career_enrollments(db, student_id)
    result: list[MyCareerEnrollmentRead] = []
    for row in rows:
        career_path_id = row["career_path_id"]
        progress = await get_my_path_progress(
            db, career_path_id=career_path_id, student_id=student_id
        )
        overall = progress.overall_percent
        flipped = await sync_enrollment_completion(
            db,
            career_path_id=career_path_id,
            student_id=student_id,
            overall_percent=overall,
        )
        result.append(
            MyCareerEnrollmentRead.model_validate(
                {
                    **row,
                    "status": "completed" if flipped else row["status"],
                    "completed_at": datetime.now(tz=UTC) if flipped else row["completed_at"],
                    "overall_percent": overall,
                    "is_prepared": overall >= 100,
                }
            )
        )
    return result


async def get_my_path_progress(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID
) -> CareerPathProgressRead:
    """Stage-aware pathway progress for one student.

    Also writes the stage latch for any stage that has just become complete
    (append-only; see :class:`~..models.StudentStageProgress`). The caller
    owns the transaction — the router commits.

    ``overall_percent`` is produced by whichever formula the global
    ``careerpath.progress_formula_version`` setting selects, and the version
    used is returned so the readiness snapshot can stamp exactly what it
    measured.
    """
    rows = await student_queries.get_path_course_progress(
        db, career_path_id=career_path_id, student_id=student_id
    )
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    evals = await stage_service.evaluate_stages(
        db,
        career_path_id=career_path_id,
        student_id=student_id,
        enrollment_id=enrollment.id if enrollment is not None else None,
    )
    if enrollment is not None:
        await stage_service.latch_completed_stages(db, enrollment_id=enrollment.id, evals=evals)

    courses = [_to_course_summary(row) for row in rows]
    course_count = len(courses)
    completed = sum(1 for c in courses if c.satisfied)
    in_progress = sum(1 for c in courses if not c.satisfied and c.completion_percent > 0)

    formula_version = await stage_service.resolve_formula_version(db)
    overall = (
        stage_service.path_progress_percent(evals)
        if formula_version >= _STAGE_AWARE_FORMULA
        else stage_service.legacy_progress_percent(rows)
    )

    path = await authoring_queries.get_career_path_for_authoring(db, career_path_id)
    cap = path.max_concurrent if path is not None else None
    active_in_path = await enrollments_api.count_active_enrollments_in_courses(
        db, student_id=student_id, course_ids=[c.course_id for c in courses]
    )

    return CareerPathProgressRead(
        career_path_id=career_path_id,
        overall_percent=overall,
        course_count=course_count,
        completed_courses=completed,
        in_progress_courses=in_progress,
        courses=courses,
        stages=[_to_stage_read(ev) for ev in evals],
        formula_version=formula_version,
        max_concurrent=cap,
        active_in_path=active_in_path,
        # Advisory only — the cap NEVER blocks, not even under `hard`
        # enforcement (which governs stage lock exclusively).
        over_concurrency_cap=cap is not None and active_in_path >= cap,
    )


def _to_course_summary(row: dict[str, object]) -> CourseProgressSummary:
    return CourseProgressSummary(
        course_id=row["course_id"],  # type: ignore[arg-type]
        slug=row["slug"],  # type: ignore[arg-type]
        title=row["title"],  # type: ignore[arg-type]
        status=row["status"],  # type: ignore[arg-type]
        completion_percent=float(row["completion_percent"]),  # type: ignore[arg-type]
        stage_id=row.get("stage_id"),  # type: ignore[arg-type]
        is_required=bool(row["is_required"]),
        satisfied=bool(row["satisfied"]),
        is_enrolled=bool(row["is_enrolled"]),
    )


def _to_stage_read(ev: stage_service.StageEval) -> StageProgressRead:
    return StageProgressRead(
        stage_id=ev.stage.id,
        position=ev.stage.position,
        title=ev.stage.title,
        description=ev.stage.description,
        min_optional_to_complete=ev.stage.min_optional_to_complete,
        unlock_policy=ev.stage.unlock_policy,
        enforcement=ev.stage.enforcement,
        unlocked=ev.unlocked,
        complete=ev.complete,
        latched=ev.latched,
        required_count=len(ev.required),
        satisfied_required=ev.satisfied_required,
        optional_count=len(ev.optional),
        satisfied_optional=ev.satisfied_optional,
        stage_total=ev.stage_total,
        stage_done=ev.stage_done,
        courses=[_to_course_summary(row) for row in ev.courses],
    )


async def start_course_in_path(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    course_id: UUID,
    student_id: UUID,
) -> StartCourseResult:
    """Student-initiated lazy enrollment into ONE course of a path (Pattern B).

    This is the carve-out to the "students cannot self-enroll" rule, and the
    framing is what makes it safe: the student never names an arbitrary
    course. The server derives eligibility entirely from a manager-made
    assignment, and every one of these must hold or the call 403s:

    1. the caller is **actively enrolled** in the path (a manager put them
       there — this is the manager-made assignment the permission rests on);
    2. the course is **in that path**;
    3. the course's stage is **unlocked** for this caller.

    So the reachable set is exactly "courses a manager already assigned me,
    in stages I have already earned". A student cannot enroll themselves in
    anything a manager did not put on their path.

    Idempotent: an existing enrollment is returned with ``created=False``
    (and a dropped one is reactivated) via the same
    ``ensure_course_enrollment`` primitive the manager bulk flow uses.

    The attention cap is reported, never enforced — exceeding
    ``max_concurrent`` returns a warning flag with a successful Start.
    """
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    if enrollment is None or enrollment.status != "active":
        raise ForbiddenError(
            "start_requires_active_path_enrollment: you are not actively "
            f"enrolled in career path {career_path_id}"
        )

    link = await authoring_queries.get_path_course_link(db, career_path_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} is not part of career path {career_path_id}")

    evals = await stage_service.evaluate_stages(
        db,
        career_path_id=career_path_id,
        student_id=student_id,
        enrollment_id=enrollment.id,
    )
    target = next((ev for ev in evals if ev.stage.id == link.stage_id), None)
    if target is None:
        raise NotFoundError(f"Stage {link.stage_id} not found in career path {career_path_id}")
    if not target.unlocked:
        raise ForbiddenError(
            "stage_locked: this course is in a stage that is not unlocked for you yet"
        )

    before = await enrollments_api.get_course_enrollment(
        db, student_id=student_id, course_id=course_id
    )
    await enrollments_api.ensure_course_enrollment(
        db,
        student_id=student_id,
        course_id=course_id,
        actor_id=student_id,
    )
    created = before is None or before.status == "dropped"

    course_ids = [
        row["course_id"]
        for row in await student_queries.get_path_course_progress(
            db, career_path_id=career_path_id, student_id=student_id
        )
    ]
    path = await authoring_queries.get_career_path_for_authoring(db, career_path_id)
    cap = path.max_concurrent if path is not None else None
    active_in_path = await enrollments_api.count_active_enrollments_in_courses(
        db, student_id=student_id, course_ids=course_ids
    )
    return StartCourseResult(
        course_id=course_id,
        stage_id=link.stage_id,
        created=created,
        over_concurrency_cap=cap is not None and active_in_path > cap,
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
        encode_composite_cursor(paths[-1].created_at, paths[-1].id) if len(paths) == limit else None
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
    "start_course_in_path",
    "sync_enrollment_completion",
    "unenroll_student",
]
