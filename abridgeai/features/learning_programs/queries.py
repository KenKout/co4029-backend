from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, exists, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.models import (
    CareerPath,
    OrgUnit,
    Role,
    UserFacultyAssignment,
    UserRoleAssignment,
)
from abridgeai.features.career_paths.api import public as career_paths_api
from abridgeai.features.career_paths.models import CareerPathVersion
from abridgeai.features.learning_programs.models import (
    PATH_CHANGE_OPEN_STATUSES,
    LearningProgram,
    LearningProgramVersion,
    LearningProgramVersionPath,
    PathChangeRequest,
    ProgramEnrollment,
    ProgramPathAttempt,
)


async def get_program(
    db: AsyncSession, program_id: UUID, *, lock: bool = False
) -> LearningProgram | None:
    stmt = select(LearningProgram).where(
        LearningProgram.id == program_id, LearningProgram.deleted_at.is_(None)
    )
    if lock:
        stmt = stmt.with_for_update()
    return (await db.scalars(stmt)).one_or_none()


async def list_programs(db: AsyncSession, organization_id: UUID) -> list[LearningProgram]:
    stmt = (
        select(LearningProgram)
        .where(
            LearningProgram.organization_id == organization_id,
            LearningProgram.deleted_at.is_(None),
        )
        .order_by(LearningProgram.name)
    )
    return list((await db.scalars(stmt)).all())


async def list_program_list_cards(
    db: AsyncSession, program_ids: list[UUID]
) -> dict[UUID, dict[str, int | bool]]:
    """Batched card statistics for the management list, 3 GROUP BY queries.

    Returns ``{program_id: {student_count, path_change_request_count,
    has_draft_version}}``. ``path_change_request_count`` counts OPEN change
    requests — ``pending`` plus ``in_progress`` — because that is the dean's
    review inbox: a request they already acknowledged is still theirs to
    finish, and dropping it from the badge the moment it is picked up would
    hide half the queue.
    """
    if not program_ids:
        return {}

    enroll_stmt = (
        select(
            ProgramEnrollment.learning_program_id.label("program_id"),
            func.count().label("n"),
        )
        .where(ProgramEnrollment.learning_program_id.in_(program_ids))
        .group_by(ProgramEnrollment.learning_program_id)
    )
    enroll_rows = (await db.execute(enroll_stmt)).all()

    requests_stmt = (
        select(
            ProgramEnrollment.learning_program_id.label("program_id"),
            func.count().label("n"),
        )
        .join(PathChangeRequest, PathChangeRequest.program_enrollment_id == ProgramEnrollment.id)
        .where(
            ProgramEnrollment.learning_program_id.in_(program_ids),
            PathChangeRequest.status.in_(PATH_CHANGE_OPEN_STATUSES),
        )
        .group_by(ProgramEnrollment.learning_program_id)
    )
    request_rows = (await db.execute(requests_stmt)).all()

    draft_stmt = (
        select(LearningProgramVersion.learning_program_id.label("program_id"))
        .where(
            LearningProgramVersion.learning_program_id.in_(program_ids),
            LearningProgramVersion.status == "draft",
        )
        .distinct()
    )
    draft_ids = {row.program_id for row in (await db.execute(draft_stmt)).all()}

    stats: dict[UUID, dict[str, int | bool]] = {
        pid: {"student_count": 0, "path_change_request_count": 0, "has_draft_version": False}
        for pid in program_ids
    }
    for row in enroll_rows:
        stats[row.program_id]["student_count"] = row.n
    for row in request_rows:
        stats[row.program_id]["path_change_request_count"] = row.n
    for pid in draft_ids:
        stats[pid]["has_draft_version"] = True
    return stats


async def get_current_version(
    db: AsyncSession, program_id: UUID, *, published_only: bool = False
) -> LearningProgramVersion | None:
    stmt = select(LearningProgramVersion).where(
        LearningProgramVersion.learning_program_id == program_id,
        LearningProgramVersion.deleted_at.is_(None),
    )
    if published_only:
        stmt = stmt.where(LearningProgramVersion.status == "published")
    stmt = stmt.order_by(LearningProgramVersion.version_no.desc()).limit(1)
    return (await db.scalars(stmt)).one_or_none()


async def get_version(db: AsyncSession, version_id: UUID) -> LearningProgramVersion | None:
    return await db.get(LearningProgramVersion, version_id)


async def list_versions(db: AsyncSession, program_id: UUID) -> list[LearningProgramVersion]:
    stmt = (
        select(LearningProgramVersion)
        .where(
            LearningProgramVersion.learning_program_id == program_id,
            LearningProgramVersion.deleted_at.is_(None),
        )
        .order_by(LearningProgramVersion.version_no.desc())
    )
    return list((await db.scalars(stmt)).all())


async def list_program_authoring_options(
    db: AsyncSession, *, organization_id: UUID, actor_id: UUID
) -> tuple[list[OrgUnit], list[CareerPath], UUID | None]:
    faculties = list(
        (
            await db.scalars(
                select(OrgUnit)
                .where(
                    OrgUnit.organization_id == organization_id,
                    OrgUnit.unit_type == "faculty",
                    OrgUnit.deleted_at.is_(None),
                )
                .order_by(OrgUnit.name)
            )
        ).all()
    )
    paths = list(
        (
            await db.scalars(
                select(CareerPath)
                .where(
                    CareerPath.organization_id == organization_id,
                    CareerPath.status == "published",
                    CareerPath.deleted_at.is_(None),
                    select(CareerPathVersion.id)
                    .where(
                        CareerPathVersion.career_path_id == CareerPath.id,
                        CareerPathVersion.status == "published",
                        CareerPathVersion.deleted_at.is_(None),
                    )
                    .exists(),
                )
                .order_by(CareerPath.name)
            )
        ).all()
    )
    default_faculty_id = await db.scalar(
        select(UserRoleAssignment.org_unit_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .join(OrgUnit, OrgUnit.id == UserRoleAssignment.org_unit_id)
        .where(
            UserRoleAssignment.user_id == actor_id,
            UserRoleAssignment.scope_kind == "org_unit",
            UserRoleAssignment.organization_id == organization_id,
            UserRoleAssignment.deleted_at.is_(None),
            Role.code.in_(("manager", "hod")),
            OrgUnit.unit_type == "faculty",
            OrgUnit.deleted_at.is_(None),
        )
        .limit(1)
    )
    return faculties, paths, default_faculty_id


async def list_version_paths(db: AsyncSession, version_id: UUID) -> list[dict[str, object]]:
    stmt = (
        select(
            LearningProgramVersionPath.career_path_id,
            LearningProgramVersionPath.career_path_version_id,
            CareerPathVersion.version_no.label("career_path_version_no"),
            CareerPath.name,
            CareerPath.slug,
            CareerPath.description,
            CareerPath.status,
            LearningProgramVersionPath.position,
            LearningProgramVersionPath.is_default,
        )
        .join(CareerPath, CareerPath.id == LearningProgramVersionPath.career_path_id)
        .join(
            CareerPathVersion,
            CareerPathVersion.id == LearningProgramVersionPath.career_path_version_id,
        )
        .where(LearningProgramVersionPath.program_version_id == version_id)
        .order_by(LearningProgramVersionPath.position)
    )
    return [dict(row) for row in (await db.execute(stmt)).mappings()]


async def delete_version_paths(db: AsyncSession, version_id: UUID) -> None:
    await db.execute(
        delete(LearningProgramVersionPath).where(
            LearningProgramVersionPath.program_version_id == version_id
        )
    )


async def list_unpublishable_version_path_ids(
    db: AsyncSession, *, version_id: UUID, organization_id: UUID
) -> list[UUID]:
    """Return path mappings that cannot be frozen into a new Program version.

    Drafts deliberately retain exact Career Path versions while authors edit
    them. Availability is rechecked at the publish boundary so a path archived
    (or otherwise invalidated) after the draft was created cannot leak into a
    newly published Program version.
    """

    stmt = (
        select(LearningProgramVersionPath.career_path_id)
        .outerjoin(
            CareerPath,
            CareerPath.id == LearningProgramVersionPath.career_path_id,
        )
        .outerjoin(
            CareerPathVersion,
            CareerPathVersion.id == LearningProgramVersionPath.career_path_version_id,
        )
        .where(
            LearningProgramVersionPath.program_version_id == version_id,
            (
                CareerPath.id.is_(None)
                | (CareerPath.organization_id != organization_id)
                | CareerPath.deleted_at.is_not(None)
                | (CareerPath.status != "published")
                | CareerPathVersion.id.is_(None)
                | CareerPathVersion.deleted_at.is_not(None)
                | (CareerPathVersion.status != "published")
                | (CareerPathVersion.career_path_id != LearningProgramVersionPath.career_path_id)
            ),
        )
    )
    return list((await db.scalars(stmt)).all())


async def resolve_published_path_versions(
    db: AsyncSession, *, organization_id: UUID, career_path_ids: list[UUID]
) -> list[dict[str, object]]:
    """Latest published version per path, asked of the career_paths feature.

    This used to rank the versions itself with its own window function. That
    was a second spelling of a rule career_paths already owns — "published,
    not deleted, highest version_no" — and one that nothing would have caught
    drifting, because both spellings were individually correct.
    """
    return await career_paths_api.resolve_published_versions(
        db, organization_id=organization_id, career_path_ids=career_path_ids
    )


async def list_all_org_paths(db: AsyncSession, *, organization_id: UUID) -> list[CareerPath]:
    """Every live career path in the org regardless of publish status.

    Feeds the authoring-options picker: published paths come back
    selectable=True, drafts/archived ones selectable=False with a reason,
    so the UI can show WHY a path cannot be attached instead of hiding it
    or letting the manager hit the attach gate's 409 blind.
    """
    stmt = (
        select(CareerPath)
        .where(
            CareerPath.organization_id == organization_id,
            CareerPath.deleted_at.is_(None),
        )
        .order_by(CareerPath.name)
    )
    return list((await db.scalars(stmt)).all())


async def list_owning_deans(
    db: AsyncSession,
    *,
    organization_id: UUID,
    faculty_id: UUID,
) -> list[UUID]:
    """User ids holding the ``hod`` role for a program's org + faculty.

    The inverse of :func:`actor_has_program_role` for the dean check: every
    user with an active Faculty-Dean assignment at the organization scope OR
    at this faculty (org_unit scope backed by an active affiliation). Used to
    fan out the "student filed a path change request" notification.
    """
    now = datetime.now(UTC)
    stmt = (
        select(UserRoleAssignment.user_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= now,
            (UserRoleAssignment.active_until.is_(None) | (UserRoleAssignment.active_until > now)),
            Role.code == "hod",
            Role.deleted_at.is_(None),
            (
                (
                    (UserRoleAssignment.scope_kind == "organization")
                    & (UserRoleAssignment.organization_id == organization_id)
                )
                | (
                    (UserRoleAssignment.scope_kind == "org_unit")
                    & (UserRoleAssignment.org_unit_id == faculty_id)
                    & exists(
                        select(UserFacultyAssignment.id).where(
                            UserFacultyAssignment.user_id == UserRoleAssignment.user_id,
                            UserFacultyAssignment.organization_id == organization_id,
                            UserFacultyAssignment.faculty_id == faculty_id,
                            UserFacultyAssignment.status == "active",
                            UserFacultyAssignment.deleted_at.is_(None),
                            (UserFacultyAssignment.active_until.is_(None))
                            | (UserFacultyAssignment.active_until > now),
                        )
                    )
                )
            ),
        )
        .distinct()
    )
    return list((await db.scalars(stmt)).all())


async def actor_has_program_role(
    db: AsyncSession,
    *,
    user_id: UUID,
    organization_id: UUID,
    faculty_id: UUID,
    role_codes: tuple[str, ...],
) -> bool:
    now = datetime.now(UTC)
    stmt = (
        select(UserRoleAssignment.id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            UserRoleAssignment.user_id == user_id,
            UserRoleAssignment.deleted_at.is_(None),
            UserRoleAssignment.active_from <= now,
            (UserRoleAssignment.active_until.is_(None) | (UserRoleAssignment.active_until > now)),
            Role.code.in_(role_codes),
            Role.deleted_at.is_(None),
            (
                (
                    (UserRoleAssignment.scope_kind == "organization")
                    & (UserRoleAssignment.organization_id == organization_id)
                )
                | (
                    (UserRoleAssignment.scope_kind == "org_unit")
                    & (UserRoleAssignment.org_unit_id == faculty_id)
                    & exists(
                        select(UserFacultyAssignment.id).where(
                            UserFacultyAssignment.user_id == user_id,
                            UserFacultyAssignment.organization_id == organization_id,
                            UserFacultyAssignment.faculty_id == faculty_id,
                            UserFacultyAssignment.status == "active",
                            UserFacultyAssignment.deleted_at.is_(None),
                            (UserFacultyAssignment.active_until.is_(None))
                            | (UserFacultyAssignment.active_until > now),
                        )
                    )
                )
            ),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).first() is not None


async def faculty_is_valid(db: AsyncSession, faculty_id: UUID, organization_id: UUID) -> bool:
    stmt = select(OrgUnit.id).where(
        OrgUnit.id == faculty_id,
        OrgUnit.organization_id == organization_id,
        OrgUnit.unit_type == "faculty",
        OrgUnit.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).first() is not None


async def get_enrollment(
    db: AsyncSession, enrollment_id: UUID, *, lock: bool = False
) -> ProgramEnrollment | None:
    stmt = select(ProgramEnrollment).where(ProgramEnrollment.id == enrollment_id)
    if lock:
        stmt = stmt.with_for_update()
    return (await db.scalars(stmt)).one_or_none()


async def get_program_enrollment(
    db: AsyncSession, program_id: UUID, student_id: UUID
) -> ProgramEnrollment | None:
    stmt = select(ProgramEnrollment).where(
        ProgramEnrollment.learning_program_id == program_id,
        ProgramEnrollment.student_id == student_id,
    )
    return (await db.scalars(stmt)).one_or_none()


async def list_student_enrollments(db: AsyncSession, student_id: UUID) -> list[ProgramEnrollment]:
    stmt = (
        select(ProgramEnrollment)
        .where(ProgramEnrollment.student_id == student_id)
        .order_by(ProgramEnrollment.enrolled_at.desc())
    )
    return list((await db.scalars(stmt)).all())


async def list_program_enrollments(db: AsyncSession, program_id: UUID) -> list[ProgramEnrollment]:
    stmt = (
        select(ProgramEnrollment)
        .where(ProgramEnrollment.learning_program_id == program_id)
        .order_by(ProgramEnrollment.enrolled_at.desc())
    )
    return list((await db.scalars(stmt)).all())


async def count_concurrent_enrollments(
    db: AsyncSession, *, organization_id: UUID, student_id: UUID
) -> int:
    stmt = (
        select(func.count())
        .select_from(ProgramEnrollment)
        .join(LearningProgram, LearningProgram.id == ProgramEnrollment.learning_program_id)
        .where(
            LearningProgram.organization_id == organization_id,
            ProgramEnrollment.student_id == student_id,
            ProgramEnrollment.status.in_(("awaiting_path", "active")),
        )
    )
    return int((await db.scalar(stmt)) or 0)


async def list_attempts(db: AsyncSession, enrollment_id: UUID) -> list[ProgramPathAttempt]:
    stmt = (
        select(ProgramPathAttempt)
        .where(ProgramPathAttempt.program_enrollment_id == enrollment_id)
        .order_by(ProgramPathAttempt.selected_at)
    )
    return list((await db.scalars(stmt)).all())


async def list_active_attempts(
    db: AsyncSession, enrollment_id: UUID, *, lock: bool = False
) -> list[ProgramPathAttempt]:
    stmt = select(ProgramPathAttempt).where(
        ProgramPathAttempt.program_enrollment_id == enrollment_id,
        ProgramPathAttempt.status == "active",
    )
    if lock:
        stmt = stmt.with_for_update()
    return list((await db.scalars(stmt)).all())


async def get_attempt(
    db: AsyncSession, attempt_id: UUID, *, lock: bool = False
) -> ProgramPathAttempt | None:
    stmt = select(ProgramPathAttempt).where(ProgramPathAttempt.id == attempt_id)
    if lock:
        stmt = stmt.with_for_update()
    return (await db.scalars(stmt)).one_or_none()


async def count_approved_switches(db: AsyncSession, enrollment_id: UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(PathChangeRequest)
        .where(
            PathChangeRequest.program_enrollment_id == enrollment_id,
            PathChangeRequest.status == "approved",
        )
    )
    return int((await db.scalar(stmt)) or 0)


async def count_other_active_path_attempts(
    db: AsyncSession,
    *,
    student_id: UUID,
    career_path_id: UUID,
    excluding_attempt_id: UUID,
) -> int:
    stmt = (
        select(func.count())
        .select_from(ProgramPathAttempt)
        .join(ProgramEnrollment, ProgramEnrollment.id == ProgramPathAttempt.program_enrollment_id)
        .where(
            ProgramEnrollment.student_id == student_id,
            ProgramPathAttempt.career_path_id == career_path_id,
            ProgramPathAttempt.status == "active",
            ProgramPathAttempt.id != excluding_attempt_id,
        )
    )
    return int((await db.scalar(stmt)) or 0)


async def count_active_paths_for_student(
    db: AsyncSession, *, organization_id: UUID, student_id: UUID
) -> int:
    """Career paths this student has running right now, across every program.

    Org-scoped exactly like ``count_concurrent_enrollments``, because the
    limit it feeds is an organization setting. Counts attempts rather than
    distinct paths: ``_require_path_not_active_elsewhere`` already makes one
    path running twice impossible, so the two numbers are the same.
    """
    stmt = (
        select(func.count())
        .select_from(ProgramPathAttempt)
        .join(ProgramEnrollment, ProgramEnrollment.id == ProgramPathAttempt.program_enrollment_id)
        .join(LearningProgram, LearningProgram.id == ProgramEnrollment.learning_program_id)
        .where(
            LearningProgram.organization_id == organization_id,
            ProgramEnrollment.student_id == student_id,
            ProgramPathAttempt.status == "active",
        )
    )
    return int((await db.scalar(stmt)) or 0)


async def find_active_path_attempt_elsewhere(
    db: AsyncSession,
    *,
    student_id: UUID,
    career_path_id: UUID,
    excluding_enrollment_id: UUID,
) -> str | None:
    """Name of another live program already running this path for this student.

    Every other duplicate guard in this feature stops at the enrollment
    boundary: ``uq_program_path_attempts_active_path`` is keyed on
    ``(program_enrollment_id, career_path_id)``, and
    ``max_career_paths_per_enrollment`` counts one enrollment's own attempts.
    Neither can see a second program, so only a student-wide lookup catches
    one path running twice.

    Deliberately filtered on the ATTEMPT status alone, with no condition on
    the owning program or enrollment -- exactly like
    ``count_other_active_path_attempts``, which decides when path access is
    released. The guard and the refcount have to agree: if the refcount still
    believes the student holds this path, selecting it again is a duplicate.

    Returns the program name rather than a count because "already active in
    Data Engineering" is the part a student can act on.
    """
    stmt = (
        select(LearningProgram.name)
        .select_from(ProgramPathAttempt)
        .join(ProgramEnrollment, ProgramEnrollment.id == ProgramPathAttempt.program_enrollment_id)
        .join(LearningProgram, LearningProgram.id == ProgramEnrollment.learning_program_id)
        .where(
            ProgramEnrollment.student_id == student_id,
            ProgramPathAttempt.career_path_id == career_path_id,
            ProgramPathAttempt.status == "active",
            ProgramPathAttempt.program_enrollment_id != excluding_enrollment_id,
        )
        .limit(1)
    )
    name = await db.scalar(stmt)
    return str(name) if name is not None else None


async def get_change_request(
    db: AsyncSession, request_id: UUID, *, lock: bool = False
) -> PathChangeRequest | None:
    stmt = select(PathChangeRequest).where(PathChangeRequest.id == request_id)
    if lock:
        stmt = stmt.with_for_update()
    return (await db.scalars(stmt)).one_or_none()


async def get_pending_request(db: AsyncSession, enrollment_id: UUID) -> PathChangeRequest | None:
    """The enrolment's single OPEN request, if any.

    "Open" is ``pending`` OR ``in_progress`` — the dean acknowledging a request
    must not free the slot, or a student could file a second one mid-review.
    Enforced in the DB by the partial unique index
    ``uq_path_change_requests_one_open`` (migration 0097); this query is the
    read side of the same invariant, which is why it can still use
    ``one_or_none()``.
    """
    stmt = select(PathChangeRequest).where(
        PathChangeRequest.program_enrollment_id == enrollment_id,
        PathChangeRequest.status.in_(PATH_CHANGE_OPEN_STATUSES),
    )
    return (await db.scalars(stmt)).one_or_none()


async def list_enrollment_change_requests(
    db: AsyncSession, enrollment_id: UUID
) -> list[PathChangeRequest]:
    """Full request history for one enrolment, newest first.

    Feeds the student's own view: a rejected request must remain visible with
    its reason, not disappear the moment it stops being pending.
    """
    stmt = (
        select(PathChangeRequest)
        .where(PathChangeRequest.program_enrollment_id == enrollment_id)
        .order_by(PathChangeRequest.created_at.desc())
    )
    return list((await db.scalars(stmt)).all())


async def get_career_path_name(db: AsyncSession, career_path_id: UUID) -> str | None:
    """Display name of one career path, for notification copy.

    ``career_paths`` has ``name`` (not ``title``). Returns ``None`` when the row
    is gone so callers can fall back rather than crash a notification.
    """
    stmt = select(CareerPath.name).where(CareerPath.id == career_path_id)
    return await db.scalar(stmt)


async def list_program_change_requests(
    db: AsyncSession, program_id: UUID
) -> list[PathChangeRequest]:
    stmt = (
        select(PathChangeRequest)
        .join(ProgramEnrollment, ProgramEnrollment.id == PathChangeRequest.program_enrollment_id)
        .where(ProgramEnrollment.learning_program_id == program_id)
        .order_by(PathChangeRequest.created_at.desc())
    )
    return list((await db.scalars(stmt)).all())


async def build_exit_snapshot(
    db: AsyncSession, *, student_id: UUID, attempt: ProgramPathAttempt
) -> dict[str, object]:
    rows = (
        (
            await db.execute(
                text("""
                SELECT cci.course_id,
                       EXISTS (
                         SELECT 1 FROM course_completion_awards cca
                         WHERE cca.student_id = :student_id
                           AND cca.course_id = cci.course_id
                           AND cca.revoked_at IS NULL
                       ) AS completed
                FROM career_course_items cci
                WHERE cci.version_id = :version_id
                ORDER BY cci.position
            """),
                {"student_id": student_id, "version_id": attempt.career_path_version_id},
            )
        )
        .mappings()
        .all()
    )
    completed = [str(row["course_id"]) for row in rows if row["completed"]]
    total = len(rows)
    # The formula-version stamp was retired with migration 0126. Existing
    # exit_snapshot blobs are frozen; new snapshots deliberately omit it.
    overall_percent = await career_paths_api.get_version_progress_percent_for_user(
        db,
        version_id=attempt.career_path_version_id,
        student_id=student_id,
    )
    return {
        "career_path_id": str(attempt.career_path_id),
        "career_path_version_id": str(attempt.career_path_version_id),
        "completed_course_ids": completed,
        "completed_courses": len(completed),
        "total_courses": total,
        "overall_percent": overall_percent,
        "captured_at": datetime.now(UTC).isoformat(),
    }


async def _terminalize_live_assessments_for_enrollments(
    db: AsyncSession, enrollment_ids: list[UUID]
) -> None:
    """Close live quizzes/interviews when their course access is dropped.

    Path entitlements are the source of truth for course access. This helper is
    called only with enrollment rows that transitioned from active to dropped,
    so shared courses retained by another path are never terminalized.
    """
    if not enrollment_ids:
        return
    params = {"enrollment_ids": enrollment_ids}
    await db.execute(
        text(
            """
            UPDATE quiz_attempts qa
            SET status = 'abandoned',
                submitted_at = NOW(),
                time_taken_seconds = GREATEST(
                    0, EXTRACT(EPOCH FROM (NOW() - qa.started_at))::int
                )
            FROM quizzes q
            JOIN course_enrollments ce ON ce.course_id = q.course_id
            WHERE qa.quiz_id = q.id
              AND ce.id = ANY(:enrollment_ids)
              AND qa.student_id = ce.student_id
              AND qa.status = 'in_progress'
            """
        ),
        params,
    )
    await db.execute(
        text(
            """
            UPDATE interview_sessions s
            SET status = 'abandoned', ended_at = NOW()
            FROM interview_configs ic
            JOIN course_enrollments ce ON ce.course_id = ic.course_id
            WHERE s.interview_config_id = ic.id
              AND ce.id = ANY(:enrollment_ids)
              AND s.student_id = ce.student_id
              AND s.status = 'in_progress'
            """
        ),
        params,
    )


async def transfer_path_entitlements(
    db: AsyncSession,
    *,
    old_attempt_id: UUID,
    new_attempt_id: UUID,
    new_path_version_id: UUID,
    actor_id: UUID,
) -> None:
    """Keep shared in-progress courses; revoke old-only course access.

    Completion rows are never dropped. An active course is dropped only when
    the revoked path-attempt grant was its final live entitlement.
    """
    await db.execute(
        text(
            """
            INSERT INTO course_enrollment_entitlements (
                id, course_enrollment_id, source_type, source_id,
                granted_at, created_by
            )
            SELECT gen_random_uuid(), cee.course_enrollment_id, 'path_attempt',
                   :new_attempt_id, NOW(), :actor_id
            FROM course_enrollment_entitlements cee
            JOIN course_enrollments ce ON ce.id = cee.course_enrollment_id
            JOIN career_course_items cci
              ON cci.course_id = ce.course_id
             AND cci.version_id = :new_path_version_id
            WHERE cee.source_type = 'path_attempt'
              AND cee.source_id = :old_attempt_id
              AND cee.revoked_at IS NULL
            ON CONFLICT (course_enrollment_id, source_type, source_id) DO NOTHING
            """
        ),
        {
            "old_attempt_id": old_attempt_id,
            "new_attempt_id": new_attempt_id,
            "new_path_version_id": new_path_version_id,
            "actor_id": actor_id,
        },
    )
    affected = (
        (
            await db.execute(
                text(
                    """
                UPDATE course_enrollment_entitlements
                SET revoked_at = NOW()
                WHERE source_type = 'path_attempt'
                  AND source_id = :old_attempt_id
                  AND revoked_at IS NULL
                RETURNING course_enrollment_id
                """
                ),
                {"old_attempt_id": old_attempt_id},
            )
        )
        .scalars()
        .all()
    )
    if not affected:
        return
    dropped_enrollment_ids = (
        (
            await db.execute(
                text(
                    """
                UPDATE course_enrollments ce
                SET status = 'dropped', dropped_at = NOW(), updated_at = NOW()
                WHERE ce.id = ANY(:enrollment_ids)
                  AND ce.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM course_enrollment_entitlements live
                      WHERE live.course_enrollment_id = ce.id
                        AND live.revoked_at IS NULL
                  )
                RETURNING ce.id
                """
                ),
                {"enrollment_ids": list(set(affected))},
            )
        )
        .scalars()
        .all()
    )
    await _terminalize_live_assessments_for_enrollments(db, dropped_enrollment_ids)


async def revoke_path_entitlements(
    db: AsyncSession,
    *,
    attempt_id: UUID,
) -> None:
    """Revoke an ended attempt without granting a replacement path.

    Active course enrollments are dropped only when no other live entitlement
    still grants access. Completion awards and completed enrollments remain
    untouched, so earned results continue to transfer to future paths.
    """
    affected = (
        (
            await db.execute(
                text(
                    """
                UPDATE course_enrollment_entitlements
                SET revoked_at = NOW()
                WHERE source_type = 'path_attempt'
                  AND source_id = :attempt_id
                  AND revoked_at IS NULL
                RETURNING course_enrollment_id
                """
                ),
                {"attempt_id": attempt_id},
            )
        )
        .scalars()
        .all()
    )
    if not affected:
        return
    dropped_enrollment_ids = (
        (
            await db.execute(
                text(
                    """
                UPDATE course_enrollments ce
                SET status = 'dropped', dropped_at = NOW(), updated_at = NOW()
                WHERE ce.id = ANY(:enrollment_ids)
                  AND ce.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM course_enrollment_entitlements live
                      WHERE live.course_enrollment_id = ce.id
                        AND live.revoked_at IS NULL
                  )
                RETURNING ce.id
                """
                ),
                {"enrollment_ids": list(set(affected))},
            )
        )
        .scalars()
        .all()
    )
    await _terminalize_live_assessments_for_enrollments(db, dropped_enrollment_ids)


__all__ = [name for name in globals() if not name.startswith("_")]
