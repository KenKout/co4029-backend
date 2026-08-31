from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.access_control.models import CareerPath, OrgUnit, Role, UserRoleAssignment
from abridgeai.features.career_paths.models import CareerPathVersion
from abridgeai.features.learning_programs.models import (
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
    has_draft_version}}``. ``path_change_request_count`` counts PENDING
    change requests only — the dean's review inbox number, which is what the
    card's dean-only flag shows.
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
            PathChangeRequest.status == "pending",
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
                | (
                    CareerPathVersion.career_path_id
                    != LearningProgramVersionPath.career_path_id
                )
            ),
        )
    )
    return list((await db.scalars(stmt)).all())


async def resolve_published_path_versions(
    db: AsyncSession, *, organization_id: UUID, career_path_ids: list[UUID]
) -> list[tuple[CareerPath, CareerPathVersion]]:
    if not career_path_ids:
        return []
    ranked = (
        select(
            CareerPathVersion.id.label("version_id"),
            CareerPathVersion.career_path_id,
            func.row_number()
            .over(
                partition_by=CareerPathVersion.career_path_id,
                order_by=CareerPathVersion.version_no.desc(),
            )
            .label("rank"),
        )
        .where(
            CareerPathVersion.status == "published",
            CareerPathVersion.deleted_at.is_(None),
        )
        .subquery()
    )
    stmt = (
        select(CareerPath, CareerPathVersion)
        .join(ranked, ranked.c.career_path_id == CareerPath.id)
        .join(CareerPathVersion, CareerPathVersion.id == ranked.c.version_id)
        .where(
            CareerPath.id.in_(career_path_ids),
            CareerPath.organization_id == organization_id,
            CareerPath.deleted_at.is_(None),
            ranked.c.rank == 1,
        )
    )
    rows = list((await db.execute(stmt)).all())
    by_id = {path.id: (path, version) for path, version in rows}
    return [by_id[path_id] for path_id in career_path_ids if path_id in by_id]


async def list_all_org_paths(
    db: AsyncSession, *, organization_id: UUID
) -> list[CareerPath]:
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


async def get_active_attempt(
    db: AsyncSession, enrollment_id: UUID, *, lock: bool = False
) -> ProgramPathAttempt | None:
    stmt = select(ProgramPathAttempt).where(
        ProgramPathAttempt.program_enrollment_id == enrollment_id,
        ProgramPathAttempt.status == "active",
    )
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


async def get_change_request(
    db: AsyncSession, request_id: UUID, *, lock: bool = False
) -> PathChangeRequest | None:
    stmt = select(PathChangeRequest).where(PathChangeRequest.id == request_id)
    if lock:
        stmt = stmt.with_for_update()
    return (await db.scalars(stmt)).one_or_none()


async def get_pending_request(db: AsyncSession, enrollment_id: UUID) -> PathChangeRequest | None:
    stmt = select(PathChangeRequest).where(
        PathChangeRequest.program_enrollment_id == enrollment_id,
        PathChangeRequest.status == "pending",
    )
    return (await db.scalars(stmt)).one_or_none()


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
    return {
        "career_path_id": str(attempt.career_path_id),
        "career_path_version_id": str(attempt.career_path_version_id),
        "completed_course_ids": completed,
        "completed_courses": len(completed),
        "total_courses": total,
        "overall_percent": round((len(completed) / total * 100) if total else 0, 2),
        "captured_at": datetime.now(UTC).isoformat(),
        "formula_version": 1,
    }


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
            """
        ),
        {"enrollment_ids": list(set(affected))},
    )


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
            """
        ),
        {"enrollment_ids": list(set(affected))},
    )


__all__ = [name for name in globals() if not name.startswith("_")]
