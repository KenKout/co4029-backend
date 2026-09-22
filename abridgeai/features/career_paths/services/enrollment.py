from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.pagination import (
    CursorPage,
    decode_composite_cursor,
    encode_composite_cursor,
)
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.career_paths.models import StudentCareerEnrollment
from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.queries import student as student_queries
from abridgeai.features.career_paths.schemas import (
    CareerPathDetailPublic,
    CareerPathProgressRead,
    CareerPathPublic,
    CareerPathStagePublic,
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
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser
    from abridgeai.features.career_paths.models import CareerPath


class _RosterAvatarTarget:
    """Duck-typed storage target for :func:`create_stream_url` (bucket + key).

    Mirrors courses/services/authoring._AuthoringStorageTarget; a local
    definition keeps the career-paths feature from importing across features.
    """

    def __init__(self, *, bucket: str, object_key: str) -> None:
        self.bucket = bucket
        self.object_key = object_key



async def _course_thumbnail_urls(
    db: AsyncSession, courses: list[dict[str, object]]
) -> dict[UUID, str]:
    """Presigned thumbnails for the courses of ONE path.

    Used by the single-path reads, which actually render course rows. The
    paths LISTING deliberately skips this: it draws path cards, so minting a
    URL for every course of every path on the page would be presigns nobody
    displays.

    Cross-feature traffic goes through ``courses.api.public`` per the
    import-linter contract. A storage failure yields a missing key, never an
    exception — the client falls back to the slug gradient.
    """
    from abridgeai.features.courses.api import public as courses_api  # noqa: PLC0415

    ids = [cast("UUID", c["course_id"]) for c in courses]
    if not ids:
        return {}
    return await courses_api.get_course_thumbnail_urls(db, ids)


def _to_path_public(
    path: CareerPath,
    courses: list[dict[str, object]],
    *,
    thumbnail_url: str | None = None,
    course_thumbnail_urls: dict[UUID, str] | None = None,
) -> CareerPathPublic:
    """``thumbnail_url`` is the PATH's own image; ``course_thumbnail_urls``
    maps course id -> the course's, so a path lists courses with the same
    artwork the catalogue shows. A course missing from the map keeps
    ``thumbnail_url = None`` and the client falls back to its gradient."""
    course_thumbnails = course_thumbnail_urls or {}
    return CareerPathPublic.model_validate(
        {
            "id": path.id,
            "slug": path.slug,
            "name": path.name,
            "description": path.description,
            "thumbnail_url": thumbnail_url,
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
                        "thumbnail_url": course_thumbnails.get(
                            cast("UUID", row["course_id"])
                        ),
                    }
                )
                for row in courses
            ],
        }
    )


async def _thumbnail_urls(
    db: AsyncSession, path_ids: list[UUID]
) -> dict[UUID, str]:
    targets = await authoring_queries.list_career_path_thumbnail_storage_targets(
        db, path_ids
    )
    urls: dict[UUID, str] = {}
    for path_id, target in targets.items():
        try:
            url, _ = await create_stream_url(
                _RosterAvatarTarget(bucket=target[0], object_key=target[1])
            )
            urls[path_id] = url
        except Exception:  # noqa: BLE001, S112 -- image failure must not break reads
            continue
    return urls


async def _thumbnail_url(db: AsyncSession, path_id: UUID) -> str | None:
    return (await _thumbnail_urls(db, [path_id])).get(path_id)


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


async def _resolve_pin_version(db: AsyncSession, career_path_id: UUID) -> UUID:
    """The version a NEW enrollment pins to (Gap 3 D3a).

    Latest published version; a draft path with no published version pins
    to its authoring version (it will be published before any student can
    meaningfully walk it).
    """
    published = await authoring_queries.get_published_version(db, career_path_id)
    if published is not None:
        return published.id
    authoring = await authoring_queries.get_current_authoring_version(db, career_path_id)
    if authoring is None:
        raise AppError(f"Career path {career_path_id} has no version to enroll against")
    return authoring.id


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
    # Career-path enrolments are student-only, mirroring the course
    # bulk-enroll guard (enrollments/services/manager.py::_resolve_student_ids):
    # a teacher, manager, HOD or admin must not be attached to a learner
    # pathway. The picker already filters to students; this is the backend
    # backstop so a crafted request gets a clear 409 instead of a weird row.
    role_codes = await access_control_api.get_role_codes_for_users(db, [student_id])
    if "student" not in role_codes.get(student_id, ()):
        raise AppError(
            f"User {student_id} is not a student and cannot be enrolled in a career path"
        )
    if existing is not None and existing.status == "dropped":
        existing.status = "active"
        existing.completed_at = None
        existing.started_at = datetime.now(tz=UTC)
        # Re-activation is a fresh start: pin to the CURRENT latest
        # published version, not the one the dropped enrollment used.
        existing.version_id = await _resolve_pin_version(db, career_path_id)
        existing.updated_by = actor.user_id
        await flush_or_conflict(db)
        await db.refresh(existing)
        return _to_authoring_enrollment(existing)

    enrollment = StudentCareerEnrollment(
        career_path_id=career_path_id,
        version_id=await _resolve_pin_version(db, career_path_id),
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


def is_path_complete(progress: CareerPathProgressRead) -> bool:
    """Has the student finished this path, under the stage-aware rules?"""
    return bool(progress.stages) and all(stage.complete for stage in progress.stages)


async def sync_enrollment_completion(
    db: AsyncSession,
    *,
    career_path_id: UUID,
    student_id: UUID,
    progress: CareerPathProgressRead,
) -> bool:
    """Flip an ``active`` enrollment to ``completed`` once the path is done —
    the "prepared" milestone. Idempotent; returns ``True`` iff it flipped on
    this call (so the caller knows whether to commit). Caller owns the
    transaction."""
    if not is_path_complete(progress):
        return False
    from abridgeai.features.learning_programs.api import public as programs_api

    await programs_api.complete_program_attempts(
        db, student_id=student_id, career_path_id=career_path_id
    )
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
    flag.

    A pure read. The flip to ``completed`` and the stage latch it depends on
    are driven from course completion (:func:`sync_paths_after_course_completion`)
    and swept nightly, so nothing here writes and the router does not commit.
    """
    rows = await student_queries.list_my_career_enrollments(db, student_id)
    result: list[MyCareerEnrollmentRead] = []
    for row in rows:
        career_path_id = row["career_path_id"]
        progress = await get_my_path_progress(
            db, career_path_id=career_path_id, student_id=student_id
        )
        overall = progress.overall_percent
        complete = is_path_complete(progress)
        # Live standing, for the case where it disagrees with the milestone.
        currently = bool(progress.stages) and all(
            stage.live_complete for stage in progress.stages
        )
        result.append(
            MyCareerEnrollmentRead.model_validate(
                {
                    **row,
                    "status": row["status"],
                    "completed_at": row["completed_at"],
                    "overall_percent": overall,
                    "is_prepared": complete,
                    "is_currently_complete": currently,
                }
            )
        )
    return result


async def sync_paths_after_course_completion(
    db: AsyncSession, *, student_id: UUID
) -> int:
    """Re-evaluate every pathway this student is still working, and write.

    Called from the course-completion writer, which is the event that can make
    a stage complete. Returns how many career enrollments flipped to
    ``completed``, so the caller knows whether anything changed.

    Every live enrollment is re-evaluated rather than only those containing the
    finished course. Resolving course to pathways would need a second mapping
    query, and a student holds at most a handful of paths (bounded by the
    organisation's concurrent-path limit), so the superset is cheaper than the
    lookup and cannot miss a path the mapping would have.

    Already-``completed`` enrollments are skipped: their stages are latched and
    there is no flip left to make.

    Caller owns the transaction.
    """
    rows = await student_queries.list_my_career_enrollments(db, student_id)
    flipped = 0
    for row in rows:
        if row["status"] != "active":
            continue
        career_path_id = row["career_path_id"]
        progress = await get_my_path_progress(
            db, career_path_id=career_path_id, student_id=student_id, latch=True
        )
        if await sync_enrollment_completion(
            db,
            career_path_id=career_path_id,
            student_id=student_id,
            progress=progress,
        ):
            flipped += 1
    return flipped


async def get_my_path_progress(
    db: AsyncSession, *, career_path_id: UUID, student_id: UUID, latch: bool = False
) -> CareerPathProgressRead:
    """Stage-aware pathway progress for one student.

    ``overall_percent`` is produced by the stage-aware formula.

    Reads nothing back into the database by default. ``latch=True`` additionally
    writes the stage latch for any stage that has just become complete
    (append-only; see :class:`~..models.StudentStageProgress`), and the caller
    then owns the commit.

    The default is off because this function backs two GET endpoints, and a GET
    that writes is a GET that a prefetch, a retry or a crawler can fire. The
    latch is driven instead from the write that causes it — see
    :func:`sync_paths_after_course_completion`.
    """
    enrollment = await student_queries.get_my_career_enrollment(
        db, student_id=student_id, career_path_id=career_path_id
    )
    # Gap 3: an enrolled student's progress reads their PINNED version — the
    # route they started. A preview (no enrollment) reads the latest
    # published version.
    if enrollment is not None:
        version_id = enrollment.version_id
    else:
        published = await authoring_queries.get_published_version(db, career_path_id)
        if published is None:
            raise NotFoundError(f"CareerPath {career_path_id} not found")
        version_id = published.id

    rows = await student_queries.get_path_course_progress(
        db, version_id=version_id, student_id=student_id
    )
    evals = await stage_service.evaluate_stages(
        db,
        version_id=version_id,
        student_id=student_id,
        enrollment_id=enrollment.id if enrollment is not None else None,
    )
    if latch and enrollment is not None:
        await stage_service.latch_completed_stages(db, enrollment_id=enrollment.id, evals=evals)

    courses = [_to_course_summary(row) for row in rows]
    course_count = len(courses)
    completed = sum(1 for c in courses if c.satisfied)
    # "In progress" is enrolled-but-not-satisfied, NOT completion_percent > 0.
    #
    # Completion is counted in whole units now (a lesson/quiz/interview is done
    # or it is not), so a student who has started a course but not yet finished
    # a single unit reads 0% — under the old fractional lesson average they read
    # something above 0. Keying off the percent therefore stopped counting
    # exactly the students who most obviously have work in flight. Under
    # Pattern B an enrollment row only exists because the student pressed
    # Start, which is a better signal of "in progress" than any percentage.
    in_progress = sum(1 for c in courses if c.is_enrolled and not c.satisfied)

    overall = stage_service.path_progress_percent(evals)

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
        max_concurrent=cap,
        active_in_path=active_in_path,
        # Advisory only — the cap NEVER blocks, not even under `hard`
        # enforcement (which governs stage lock exclusively).
        over_concurrency_cap=cap is not None and active_in_path >= cap,
    )


def _to_course_summary(row: dict[str, object]) -> CourseProgressSummary:
    return CourseProgressSummary(
        course_id=row["course_id"],
        slug=row["slug"],
        title=row["title"],
        status=row["status"],
        completion_percent=float(row["completion_percent"]),  # type: ignore[arg-type]
        unit_total=int(row.get("unit_total") or 0),
        unit_done=int(row.get("unit_done") or 0),
        stage_id=row.get("stage_id"),
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
        live_complete=ev.live_complete,
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
    3. the course's stage is **unlocked** for this caller — or, if locked, its
       ``enforcement`` is not ``hard`` (``soft``/``advisory`` allow the Start
       and return ``stage_locked_warning=True``).

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

    # Gap 3: the reachable route is the VERSION this enrollment is pinned
    # to — never the path's current authoring version.
    link = await authoring_queries.get_version_course_link(db, enrollment.version_id, course_id)
    if link is None:
        raise NotFoundError(f"Course {course_id} is not part of career path {career_path_id}")

    evals = await stage_service.evaluate_stages(
        db,
        version_id=enrollment.version_id,
        student_id=student_id,
        enrollment_id=enrollment.id,
    )
    target = next((ev for ev in evals if ev.stage.id == link.stage_id), None)
    if target is None:
        raise NotFoundError(f"Stage {link.stage_id} not found in career path {career_path_id}")
    # Only `enforcement='hard'` blocks. `soft` and `advisory` are display/warn
    # levels: the manager UI literally offers them as "Show a warning, still
    # allow" and "Only mark it in the interface", so blocking them here would
    # make the settings popover lie. `soft` is also the DDL default, which is
    # why this must go through the helper rather than test `unlocked` directly.
    if stage_service.stage_is_hard_locked(target):
        raise ForbiddenError(
            "stage_locked: this course is in a stage that is not unlocked for you yet"
        )
    # Locked but not hard — the Start succeeds and the caller is told they are
    # working ahead. Without this flag a soft-locked Start would look exactly
    # like a normal one and the student would never see the warning they were
    # promised.
    stage_locked_warning = not target.unlocked

    before = await enrollments_api.get_course_enrollment(
        db, student_id=student_id, course_id=course_id
    )
    await enrollments_api.ensure_course_enrollment(
        db,
        student_id=student_id,
        course_id=course_id,
        actor_id=student_id,
    )
    current_course_enrollment = await enrollments_api.get_course_enrollment(
        db, student_id=student_id, course_id=course_id
    )
    if current_course_enrollment is not None:
        from abridgeai.features.learning_programs.api import public as programs_api

        await programs_api.grant_active_path_entitlement(
            db,
            student_id=student_id,
            career_path_id=career_path_id,
            course_enrollment_id=current_course_enrollment.id,
            actor_id=student_id,
        )
    created = before is None or before.status == "dropped"

    course_ids = [
        row["course_id"]
        for row in await student_queries.get_path_course_progress(
            db, version_id=enrollment.version_id, student_id=student_id
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
        stage_locked_warning=stage_locked_warning,
        active_in_path=active_in_path,
        max_concurrent=cap,
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
    from abridgeai.features.career_paths.queries import authoring as authoring_queries

    published = await authoring_queries.get_published_version(db, path.id)
    if published is None:
        # Published path with only a draft version (transient pre-publish
        # state): browse the authoring version rather than 404.
        published = await authoring_queries.get_current_authoring_version(db, path.id)
    if published is None:
        return None
    courses = await list_published_career_path_courses(db, published.id)
    return _to_path_public(
        path,
        courses,
        thumbnail_url=await _thumbnail_url(db, path.id),
        course_thumbnail_urls=await _course_thumbnail_urls(db, courses),
    )


async def _visible_career_path_ids(db: AsyncSession, user_id: UUID) -> set[UUID] | None:
    """Which career paths this student may see, or ``None`` for "all of them".

    A student inside a learning program may only take the paths their program
    version pins — ``request_path_change`` enforces exactly that server-side
    (``target_path_is_not_in_the_pinned_program_version``). Before this scope
    existed the catalog offered every published path in the org, so an off-menu
    path rendered a detail page with no action and no explanation; it read as a
    broken button rather than a path that was never on offer.

    ``None`` (no restriction) when the student belongs to no learning program:
    that is the pre-program catalog behaviour and still the right answer for a
    directly-enrolled or merely browsing student. Paths they are already
    enrolled on are unioned in, so a live enrolment can never be hidden by a
    program version that later dropped the path.
    """
    program_ids = await student_queries.list_my_program_career_path_ids(db, user_id)
    if not program_ids:
        return None
    return program_ids | await student_queries.list_my_enrolled_career_path_ids(db, user_id)


async def get_published_path_detail_for_user(
    db: AsyncSession, *, slug: str, user_id: UUID
) -> CareerPathDetailPublic | None:
    """Published path WITH its stage breakdown, for a prospective student.

    The plain :func:`get_published_path_for_user` returns a flat course list,
    which is all the catalog needs. Choosing a path inside a learning program
    is a bigger decision than browsing, so that screen gets the roadmap: the
    stages, their gating policy, and which courses sit in each.

    Courses are grouped by their ``stage_id``. A path authored before stages
    existed has ``stage_id = NULL`` on every course; those fall through as an
    empty ``stages`` list and the client keeps rendering the flat list, so
    this is additive rather than a breaking change for older paths.
    """
    from abridgeai.features.career_paths.queries import authoring as authoring_queries
    from abridgeai.features.career_paths.queries import (
        get_published_career_path_by_slug,
        get_user_primary_organization_id,
        list_published_career_path_courses,
    )

    organization_id = await get_user_primary_organization_id(db, user_id)
    if organization_id is None:
        return None
    path = await get_published_career_path_by_slug(db, slug=slug, organization_id=organization_id)
    if path is None:
        return None
    # Off-menu for this student's program → treat as absent rather than serving
    # a page whose only action would 409. 404 also keeps the org catalog from
    # being an existence oracle for paths the student was never offered.
    visible = await _visible_career_path_ids(db, user_id)
    if visible is not None and path.id not in visible:
        return None

    published = await authoring_queries.get_published_version(db, path.id)
    if published is None:
        published = await authoring_queries.get_current_authoring_version(db, path.id)
    if published is None:
        return None

    courses = await list_published_career_path_courses(db, published.id)
    base = _to_path_public(
        path,
        courses,
        thumbnail_url=await _thumbnail_url(db, path.id),
        course_thumbnail_urls=await _course_thumbnail_urls(db, courses),
    )
    stages = await authoring_queries.list_stages_for_version(db, published.id)

    by_stage: dict[str, list[CareerPathCoursePublic]] = {}
    for course in base.courses:
        if course.stage_id is not None:
            by_stage.setdefault(str(course.stage_id), []).append(course)

    stage_dtos: list[CareerPathStagePublic] = []
    for stage in stages:
        in_stage = by_stage.get(str(stage.id), [])
        stage_dtos.append(
            CareerPathStagePublic(
                stage_id=stage.id,
                position=stage.position,
                title=stage.title,
                description=stage.description,
                unlock_policy=stage.unlock_policy,
                min_optional_to_complete=stage.min_optional_to_complete,
                required_count=sum(1 for c in in_stage if c.is_required),
                optional_count=sum(1 for c in in_stage if not c.is_required),
                courses=in_stage,
            )
        )

    return CareerPathDetailPublic(
        **base.model_dump(),
        stages=stage_dtos,
        course_count=len(base.courses),
        required_course_count=sum(1 for c in base.courses if c.is_required),
        stage_count=len(stage_dtos),
    )


async def get_published_path_for_user(
    db: AsyncSession, *, slug: str, user_id: UUID
) -> CareerPathPublic | None:
    from abridgeai.features.career_paths.queries import get_user_primary_organization_id

    organization_id = await get_user_primary_organization_id(db, user_id)
    if organization_id is None:
        return None
    result = await get_published_path_with_courses(
        db, slug=slug, organization_id=organization_id
    )
    if result is None:
        return None
    # Same program scope as the /detail read — the two must agree, or the slim
    # read would still hand out a path the roadmap read refuses.
    visible = await _visible_career_path_ids(db, user_id)
    if visible is not None and result.id not in visible:
        return None
    return result


async def list_published_paths(
    db: AsyncSession,
    *,
    organization_id: UUID,
    limit: int,
    cursor: str | None,
    restrict_to_ids: set[UUID] | None = None,
) -> CursorPage[CareerPathPublic]:
    """Cursor-paginated published career paths ordered by ``(created_at DESC, id DESC)``."""
    from abridgeai.features.career_paths.queries import (
        list_published_career_path_courses_by_versions,
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
        restrict_to_ids=restrict_to_ids,
    )
    path_ids = [path.id for path in paths]
    published_versions = await authoring_queries.list_published_versions(
        db, organization_id=organization_id, career_path_ids=path_ids
    )
    version_ids = [row["version_id"] for row in published_versions]
    courses_by_version = await list_published_career_path_courses_by_versions(db, version_ids)
    version_by_path = {row["career_path_id"]: row["version_id"] for row in published_versions}
    results: list[CareerPathPublic] = []
    thumbnail_urls = await _thumbnail_urls(db, [path.id for path in paths])
    for path in paths:
        version_id = version_by_path.get(path.id)
        if version_id is None:
            continue
        courses = courses_by_version.get(version_id, [])
        results.append(
            _to_path_public(
                path, courses, thumbnail_url=thumbnail_urls.get(path.id)
            )
        )
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
        db,
        organization_id=organization_id,
        limit=limit,
        cursor=cursor,
        # Scope the catalog to the paths this student's learning program offers;
        # None when they are in no program (browse the whole org, as before).
        restrict_to_ids=await _visible_career_path_ids(db, user_id),
    )


async def get_roster_progress(
    db: AsyncSession, career_path_id: UUID
) -> list[StudentPathProgressAuthoring]:
    # Gap 3: the roster measures the route students are actually on — the
    # path's current PUBLISHED version (each student's pin resolves their
    # own; the published version is the shared denominator managers track).
    published = await authoring_queries.get_published_version(db, career_path_id)
    if published is None:
        return []
    stages = await authoring_queries.list_stages_for_version(db, published.id)
    rows = await student_queries.get_roster_path_progress(
        db, career_path_id=career_path_id
    )
    out: list[StudentPathProgressAuthoring] = []
    for row in rows:
        evals = await stage_service.evaluate_stages(
            db,
            version_id=published.id,
            student_id=row["student_id"],
            enrollment_id=None,
            prefetched_stages=stages,
        )
        courses = [course for ev in evals for course in ev.courses]
        overall_percent = stage_service.path_progress_percent(evals)
        completed_courses = sum(1 for course in courses if course["satisfied"])
        course_count = len(courses)
        avatar_url: str | None = None
        bucket = row.pop("avatar_bucket", None)
        object_key = row.pop("avatar_object_key", None)
        if bucket and object_key:
            try:
                url, _ = await create_stream_url(
                    _RosterAvatarTarget(bucket=bucket, object_key=object_key)
                )
                avatar_url = url
            except Exception:  # noqa: BLE001 — a storage blip must not break the roster
                avatar_url = None
        out.append(
            StudentPathProgressAuthoring.model_validate(
                {
                    "student_id": row["student_id"],
                    "student_email": row["primary_email"],
                    "student_display_name": row.get("display_name"),
                    "student_avatar_url": avatar_url,
                    "overall_percent": overall_percent,
                    "completed_courses": completed_courses,
                    "course_count": course_count,
                }
            )
        )
    return out


__all__ = [
    "enroll_student_in_path",
    "get_my_path_progress",
    "get_published_path_for_user",
    "get_published_path_with_courses",
    "get_roster_progress",
    "is_path_complete",
    "list_my_career_enrollments",
    "list_published_paths",
    "list_published_paths_for_user",
    "start_course_in_path",
    "sync_enrollment_completion",
    "sync_paths_after_course_completion",
    "unenroll_student",
]
