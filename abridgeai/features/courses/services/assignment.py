"""HOD/Manager-side teacher assignment service for courses (plan §4209).

Bridges the courses feature to the access-control feature for the narrow
case of "assign teacher X to course Y". Per Reconciliation §A1 + the
import-linter contract, the cross-feature reach is read-and-write but
intentionally narrow: this module composes raw-SQL helpers from
:mod:`features.courses.queries.assignment` + the seeded ``role_code='teacher'``
catalog row from T1.12.

Soft-revoke semantics — :func:`remove_teacher_from_course` sets
``active_until = NOW()`` rather than DELETE-ing the assignment row; the
legacy ``backend/app/routes/courses/`` flow had no remove endpoint, so
the locked decision per plan §4211 is "match revoke pattern from
T1.10 admin", which is the soft-revoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.core.exceptions import AppError, ConflictError, ForbiddenError, NotFoundError
from abridgeai.core.runtime_settings import resolve_setting
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.queries import (
    assignment as assignment_queries,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.schemas import CourseAuthoring, InstructorAuthoring, InstructorRead
from abridgeai.features.courses.services import notify
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.courses.models import Course


@dataclass
class _StorageTarget:
    bucket: str
    object_key: str


async def assign_teacher_to_course(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
    *,
    course_role: str = "teacher_assistant",
    arq_pool: object | None = None,
) -> dict[str, Any]:
    """Create (or no-op return) a ``role=teacher, scope=course`` assignment.

    If an active assignment already exists for ``(course_id, user_id)`` the
    existing row is returned unchanged. Otherwise a new row is INSERT-ed
    with ``role_id`` resolved from the seeded T1.12 catalog,
    ``scope_kind='course'``, and ``granted_by=actor.user_id``.

    Staffing bounds (admin config, user decision 2026-08-18):

    * **max** — assigning when the course is already at ``courses.max_teachers
      per_course`` is rejected (hard). Existing over-cap courses are
      grandfathered: no forced removal, just no growth.
    * **exactly one Course Instructor** — the FIRST teacher on a course is
      always the Course Instructor; a later request to add a second CI is
      rejected; every other teacher defaults to Teacher Assistant. The
      ``uq_course_teachers_one_instructor`` partial index is the DB-level
      backstop for the at-most-one half.

    When the course is already **published**, the newly-assigned teacher is
    notified (in-app + email) with a deep-link to the course. A no-op re-assign
    does not re-notify. Draft courses do not notify here — the teacher is told
    when the course publishes (see :func:`publish_course`).
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")

    # The assignee MUST belong to the course's organization. Enforced here,
    # server-side, rather than by whatever list the UI happened to render: the
    # request carries a bare user_id, so a client could otherwise name any user
    # in the system and grant them course.update on another org's course. The
    # course-scoped permission dep upstream checks the ACTOR's reach, not the
    # assignee's membership — different question.
    if not await _user_is_in_org(db, user_id=user_id, org_id=course.organization_id):
        raise ForbiddenError(
            f"teacher_not_in_course_org: user {user_id} is not a member of the "
            f"organization that owns course {course_id}"
        )

    existing = await assignment_queries.find_active_teacher_assignment(
        db, course_id=course_id, user_id=user_id
    )
    if existing is not None:
        existing_row = await assignment_queries.get_active_teacher_assignment_row(
            db, course_id=course_id, user_id=user_id
        )
        return {
            "id": existing,
            "course_id": course_id,
            "user_id": user_id,
            "role_code": "teacher",
            "scope_kind": "course",
            "organization_id": course.organization_id,
            "granted_by": actor.user_id,
            "course_role": existing_row.course_role if existing_row else "teacher_assistant",
        }

    max_teachers = int(
        await resolve_setting(
            db, "courses.max_teachers_per_course", course.organization_id
        )
    )
    current = await assignment_queries.count_active_course_teachers(db, course_id)
    if current >= max_teachers:
        raise ConflictError(
            f"course_teacher_max_reached: course {course_id} already has "
            f"the maximum of {max_teachers} teachers"
        )

    # Resolve the title. Exactly one Course Instructor per course.
    instructor_id = await assignment_queries.find_course_instructor_id(db, course_id)
    if current == 0:
        chosen = "course_instructor"
    elif instructor_id is not None and course_role == "course_instructor":
        raise ConflictError(
            "course_teacher_one_instructor: a course may only have one "
            "Course Instructor — promote or demote titles instead of adding "
            "a second instructor"
        )
    elif instructor_id is None:
        # Course has teachers but (edge case after a bad backfill) no CI —
        # fill the gap rather than leave a titleless course.
        chosen = "course_instructor"
    else:
        chosen = "teacher_assistant"

    role_id = await assignment_queries.get_teacher_role_id(db)
    new_id = uuid4()
    await assignment_queries.insert_teacher_assignment(
        db,
        assignment_id=new_id,
        user_id=user_id,
        role_id=role_id,
        organization_id=course.organization_id,
        course_id=course_id,
        granted_by=actor.user_id,
        course_role=chosen,
    )

    # Notify on assignment for draft AND published alike. The manager flow is
    # create (draft) -> assign teacher -> teacher edits content -> publish, so
    # at assignment time the course is ALWAYS a draft: gating this on
    # `status == "published"` meant the notification never fired in the real
    # flow and the teacher was handed work nobody told them about.
    #
    # Archived is the one status with nothing left to act on.
    # Never let a notification failure roll back the assignment.
    if course.status != "archived":
        await notify.notify_teacher_assigned(
            db,
            teacher_user_id=user_id,
            course_id=course_id,
            course_title=course.title,
            arq_pool=arq_pool,
        )

    return {
        "id": new_id,
        "course_id": course_id,
        "user_id": user_id,
        "role_code": "teacher",
        "scope_kind": "course",
        "organization_id": course.organization_id,
        "granted_by": actor.user_id,
        "course_role": chosen,
    }


async def set_course_role(
    db: AsyncSession,
    *,
    course_id: UUID,
    user_id: UUID,
    course_role: str,
    actor: CurrentUser,
) -> dict[str, Any]:
    """Switch an assigned teacher's title (Course Instructor / TA).

    Rules (exactly one Course Instructor per course):

    * promoting a Teacher Assistant to Course Instructor when a DIFFERENT user
      already holds it -> rejected;
    * demoting the sole Course Instructor when they are the only teacher ->
      rejected (the course would be left with zero instructors);
    * demoting the Course Instructor when Teacher Assistants exist is the
      normal "hand off the lead" move and is allowed (a TA becomes the new CI
      through a separate call).
    """
    del actor
    assignment = await assignment_queries.get_active_teacher_assignment_row(
        db, course_id=course_id, user_id=user_id
    )
    if assignment is None:
        raise NotFoundError(
            f"No active teacher assignment for course={course_id} user={user_id}"
        )
    if course_role not in ("course_instructor", "teacher_assistant"):
        raise AppError(f"Unknown course_role: {course_role!r}")

    if assignment.course_role == course_role:
        return {
            "course_id": course_id,
            "user_id": user_id,
            "course_role": assignment.course_role,
        }

    current_instructor = await assignment_queries.find_course_instructor_id(db, course_id)
    if course_role == "course_instructor":
        if current_instructor is not None and current_instructor != user_id:
            raise ConflictError(
                "course_teacher_one_instructor: another teacher is already "
                "this course's Course Instructor"
            )
        assignment.course_role = "course_instructor"
    else:  # demote
        if current_instructor == user_id:
            total = await assignment_queries.count_active_course_teachers(db, course_id)
            if total <= 1:
                raise ConflictError(
                    "course_teacher_sole_instructor: a course must have exactly "
                    "one Course Instructor; assign or promote a Teacher "
                    "Assistant before demoting the instructor"
                )
        assignment.course_role = "teacher_assistant"
    await db.flush()
    return {
        "course_id": course_id,
        "user_id": user_id,
        "course_role": assignment.course_role,
    }


async def get_course_readiness(db: AsyncSession, course_id: UUID) -> dict[str, Any]:
    """The four things that decide whether a course can actually be delivered.

    Answers, before the manager presses publish, the questions that otherwise
    surface as a 409 or — worse — as silence:

    * **teacher** — nobody is going to author the content otherwise.
    * **content** — at least one published lesson / quiz / interview. This is
      a publish gate; showing it here is the difference between "fix it now"
      and a 409 weeks later.
    * **learning outcomes** — at least one. Also a publish gate, and the one
      most easily forgotten: unlike missing content, a course with no stated
      outcomes looks finished from the authoring screens.
    * **career path** — a course on no path is unreachable for students; the
      paths are how they enrol. Not a publish blocker, but it means done-looking
      work that nobody can see.
    * **published** — the course's own status.

    `can_publish` is exactly the publish gate's condition, so the checklist and
    the 409 can never disagree: both read the gradeable-unit count.
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")

    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    teachers = await assignment_queries.list_teachers_for_course(db, course_id)
    units = await enrollments_api.count_course_gradeable_units(db, course_id=course_id)
    outcomes = await authoring_queries.count_course_outcomes(db, course_id)
    paths = await assignment_queries.list_career_paths_containing_course(db, course_id)

    min_teachers = int(
        await resolve_setting(
            db, "courses.min_teachers_per_course", course.organization_id
        )
    )
    max_teachers = int(
        await resolve_setting(
            db, "courses.max_teachers_per_course", course.organization_id
        )
    )
    course_instructor_count = int(
        await assignment_queries.find_course_instructor_id(db, course_id) is not None
    )

    return {
        "course_id": course_id,
        "status": course.status,
        "teacher_count": len(teachers),
        "course_instructor_count": course_instructor_count,
        "min_teachers_per_course": min_teachers,
        "max_teachers_per_course": max_teachers,
        "gradeable_unit_count": units,
        "learning_outcome_count": outcomes,
        "career_paths": paths,
        # A REQUIRED course with no gradeable unit does not merely fail to
        # complete: it locks its stage and every stage behind it, for every
        # student on that path. Surfaced separately because the fix is urgent
        # in a way the plain "no content" row is not.
        "blocks_required_stage": units == 0
        and any(path["is_required"] for path in paths),
        # The staffing minimum is a first-publish gate; a course that is
        # already published is grandfathered and shows as ready on staffing.
        "staffing_ok": len(teachers) >= min_teachers
        or course.status != "draft",
        # Must stay the EXACT conjunction publish_course gates on. A checklist
        # that says "ready" and a publish that answers 409 is worse than no
        # checklist: the manager trusts the green tick and blames the button.
        # The teacher minimum is a first-publish gate (grandfathered once the
        # course is published), which `staffing_ok` encodes.
        "can_publish": units > 0
        and outcomes > 0
        and course.status != "archived"
        and (
            len(teachers) >= min_teachers or course.status != "draft"
        ),
    }


async def list_assignable_teachers_for_creator(
    db: AsyncSession, creator: CurrentUser
) -> list[dict[str, Any]]:
    """Teachers the creator could staff a course with, BEFORE the course exists.

    The create-course wizard picks teachers in the same form that creates the
    course, so there is no ``course_id`` to derive the organization from yet.
    It resolves to the same organization ``create_course`` will stamp on the
    new row — the creator's primary org, from the token — so the picker cannot
    offer someone the subsequent assignment would then reject. Deriving it from
    the token rather than a client parameter keeps the org restriction a
    server-side fact here too.
    """
    from abridgeai.features.career_paths.queries import (  # noqa: PLC0415
        get_user_primary_organization_id,
    )

    org_id = await get_user_primary_organization_id(db, creator.user_id)
    if org_id is None:
        # Same condition create_course raises on, surfaced here as an empty
        # picker would be a lie ("no teachers exist") rather than the truth
        # ("you have no organization, so you cannot create a course at all").
        raise AppError(
            f"User {creator.user_id} has no primary organization; cannot staff a course."
        )
    return await assignment_queries.list_assignable_teachers(db, organization_id=org_id)


async def list_assignable_teachers(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Teachers a manager may assign to ``course_id``: same org, teacher role.

    The organization is derived from the COURSE, never from a client
    parameter — "belongs to that org" has to be a server-side fact, otherwise
    it is only a UI convention that a crafted request walks straight past.
    """
    course = await authoring_queries.get_course_for_authoring(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")
    return await assignment_queries.list_assignable_teachers(
        db, organization_id=course.organization_id, course_id=course_id
    )


async def _user_is_in_org(db: AsyncSession, *, user_id: UUID, org_id: UUID) -> bool:
    """Whether ``user_id`` has an active membership in ``org_id``.

    Lazy import keeps the courses -> access_control edge out of module import
    time, matching the pattern in ``courses.services.catalog``.
    """
    from abridgeai.features.access_control.api import public as access_api  # noqa: PLC0415

    return await access_api.is_user_member_of_org(db, user_id=user_id, org_id=org_id)


async def remove_teacher_from_course(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
) -> None:
    """Soft-revoke the active teacher assignment for ``(course_id, user_id)``.

    Sets ``active_until = NOW()`` on the assignment row rather than
    deleting it, preserving the audit trail (legacy parity with the
    T1.10 admin revoke flow). 404 when no active assignment exists.

    Refuses to remove the ONLY Course Instructor while other teachers remain
    (a course must keep exactly one instructor) — the manager must promote a
    Teacher Assistant first. Removing the sole teacher (a CI with nobody else)
    is allowed: that simply empties the course.
    """
    del actor
    assignment = await assignment_queries.get_active_teacher_assignment_row(
        db, course_id=course_id, user_id=user_id
    )
    if assignment is None:
        raise NotFoundError(f"No active teacher assignment for course={course_id} user={user_id}")
    if assignment.course_role == "course_instructor":
        current = await assignment_queries.count_active_course_teachers(db, course_id)
        if current > 1:
            raise ConflictError(
                "course_teacher_remove_sole_instructor: promote a Teacher "
                "Assistant to Course Instructor (or assign one) before "
                "removing this course's only instructor"
            )
    await assignment_queries.revoke_teacher_assignment(db, assignment.id)


async def _mint_avatar_url(bucket: str | None, object_key: str | None) -> str | None:
    """Presign an avatar object, degrading to ``None`` rather than failing the list.

    The sign call is local-only (no network/DB round-trip), so calling it once
    per row does not reintroduce an N+1; a storage blip drops that one avatar
    back to initials instead of 500-ing the whole staffing table.
    """
    if not bucket or not object_key:
        return None
    try:
        url, _ = await create_stream_url(_StorageTarget(bucket=bucket, object_key=object_key))
    except Exception:  # noqa: BLE001 — a storage blip must not break the list
        return None
    return url


async def list_teachers_for_course(db: AsyncSession, course_id: UUID) -> list[InstructorRead]:
    """Return ``InstructorRead`` rows for the active teachers of ``course_id``.

    Ordered Course Instructor first, then Teacher Assistants, so the student
    page shows the instructor up front with TAs behind (user decision
    2026-08-18).
    """
    rows = await assignment_queries.list_teachers_for_course(db, course_id)
    ordered = sorted(
        rows,
        key=lambda r: (r["course_role"] != "course_instructor", r["active_from"] or r["user_id"]),
    )
    return [
        InstructorRead.model_validate(
            {
                "user_id": row["user_id"],
                "display_name": row["display_name"] or row["primary_email"],
                "avatar_url": await _mint_avatar_url(
                    row.get("avatar_bucket"), row.get("avatar_object_key")
                ),
                "headline": None,
            }
        )
        for row in ordered
    ]


async def _attach_health_projections(
    db: AsyncSession,
    courses: list[Course],
    dtos: list[CourseAuthoring],
) -> None:
    """Attach the manager worklist projections to course DTOs, batched.

    One call to :func:`~abridgeai.features.courses.queries.authoring.
    count_students_and_modules_for_courses` (student + module counts) and one
    to :func:`~abridgeai.features.courses.queries.authoring.
    list_instructors_for_courses` (owner profile block) — the same no-N+1
    pattern the "My courses" grid uses. Courses whose owner has no profile
    keep ``instructor=None`` so the SPA can render an "Unassigned" chip.

    Instructors with an avatar get a presigned ``avatar_url`` minted here
    (mirrors the public catalog path). The sign call is local-only (no
    network/DB round-trip), so per-instructor minting does not reintroduce
    N+1; a storage blip degrades to initials rather than failing the list.
    """
    counts = await authoring_queries.count_students_and_modules_for_courses(
        db, [c.id for c in courses]
    )
    instructors = await authoring_queries.list_instructors_for_courses(
        db, [c.id for c in courses]
    )
    for dto, orm in zip(dtos, courses, strict=True):
        students, modules = counts.get(orm.id, (0, 0))
        dto.student_count = students
        dto.module_count = modules
        instructor_data = instructors.get(orm.id)
        if instructor_data is not None:
            bucket = instructor_data.pop("avatar_bucket", None)
            object_key = instructor_data.pop("avatar_object_key", None)
            avatar_url: str | None = None
            if bucket and object_key:
                try:
                    url, _ = await create_stream_url(
                        _StorageTarget(bucket=bucket, object_key=object_key)
                    )
                    avatar_url = url
                except Exception:  # noqa: BLE001 — a storage blip must not break the list
                    avatar_url = None
            instructor_data["avatar_url"] = avatar_url
            dto.instructor = InstructorAuthoring.model_validate(instructor_data)


async def list_courses_in_dept(db: AsyncSession, org_unit_id: UUID) -> list[CourseAuthoring]:
    """All courses in ``org_unit_id`` **and every unit below it**.

    Serves both the HOD's default list and the manager's "filter by node"
    picker, which is why it expands the subtree: standing on a faculty and
    seeing none of its departments' courses would make the org tree useless
    as a filter, and it is also what the permission engine already grants
    (see :func:`access_control.api.public.get_org_unit_subtree_ids`).
    """
    # Lazy import, matching the sibling helpers in this module: a
    # module-level courses.services -> access_control edge would be a new
    # import-time cross-feature dependency.
    from abridgeai.features.access_control.api import public as access_api  # noqa: PLC0415

    unit_ids = await access_api.get_org_unit_subtree_ids(db, org_unit_id)
    courses = await authoring_queries.list_courses_in_org_units(db, unit_ids)
    dtos = [CourseAuthoring.model_validate(course) for course in courses]
    await _attach_health_projections(db, courses, dtos)
    return dtos


async def list_courses_for_organization(
    db: AsyncSession, organization_id: UUID | None = None
) -> list[CourseAuthoring]:
    """Manager/Admin overview — courses optionally filtered by organization."""
    courses = await assignment_queries.list_courses_by_organization(db, organization_id)
    dtos = [CourseAuthoring.model_validate(course) for course in courses]
    await _attach_health_projections(db, courses, dtos)
    return dtos


async def list_teachers_with_emails(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Active teachers for a course with email + display_name + presigned avatar.

    The avatar bucket/key the query projects are swapped for a short-TTL
    ``avatar_url`` here, so the staffing tab renders the same photo the manager
    worklist does instead of falling back to initials for everyone.
    """
    rows = await assignment_queries.list_teachers_for_course(db, course_id)
    ordered = sorted(
        rows,
        key=lambda r: (
            r["course_role"] != "course_instructor",
            r["active_from"] or r["user_id"],
        ),
    )
    for row in ordered:
        bucket = row.pop("avatar_bucket", None)
        object_key = row.pop("avatar_object_key", None)
        row["avatar_url"] = await _mint_avatar_url(bucket, object_key)
    return ordered


async def list_course_roster(db: AsyncSession, course_id: UUID) -> list[dict[str, Any]]:
    """Enrolled students for a course (HOD/Manager view), with presigned avatars.

    Pops the query's raw bucket/key so they never reach the response body — the
    SPA only ever sees a short-TTL ``avatar_url`` (or ``None`` → initials).
    """
    rows = await authoring_queries.list_course_roster(db, course_id)
    for row in rows:
        bucket = row.pop("avatar_bucket", None)
        object_key = row.pop("avatar_object_key", None)
        row["avatar_url"] = await _mint_avatar_url(bucket, object_key)
    return rows


__all__ = [
    "assign_teacher_to_course",
    "list_course_roster",
    "list_courses_for_organization",
    "list_courses_in_dept",
    "list_teachers_for_course",
    "list_teachers_with_emails",
    "remove_teacher_from_course",
    "set_course_role",
]
