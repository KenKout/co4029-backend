from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db.conflict_mapper import flush_or_conflict, register_conflict_mappings
from abridgeai.features.career_paths.api import public as career_paths_api
from abridgeai.features.learning_programs.models import (
    CourseCompletionAward,
    CourseEnrollmentEntitlement,
    ProgramEnrollment,
    ProgramPathAttempt,
)

register_conflict_mappings(
    {
        "uq_course_completion_awards_student_course": "course_completion_already_awarded",
        "uq_course_enrollment_entitlements_source": "course_entitlement_already_exists",
    }
)


async def list_program_governance_rows(
    db: AsyncSession, *, organization_id: UUID, actor: object
) -> list[dict[str, object]]:
    """Programs in an organization with their governance counters, for a dashboard.

    Backs the manager / faculty-dean decision queue in
    ``courses.services.management_dashboard``. Cross-feature callers cannot
    reach ``learning_programs.services``, so this is the blessed surface.

    Delegates to the SAME service the management list page uses, so a program
    hidden from a dean there is hidden here too: ``list_programs`` filters
    per-program by ``_actor_can_operate``, which is what restricts a faculty
    dean to the programs they own. Re-implementing the filter would be a second
    definition of "my programs" and the two would drift.

    Returns plain dicts rather than ``ProgramRead`` so the schema stays inside
    this feature -- the same contract
    :func:`list_student_program_enrollments` follows.

    ``path_change_request_count`` counts OPEN statuses only (``pending`` +
    ``in_progress``), matching the management list card. The per-program
    drill-down returns every status and will legitimately show more rows.
    """
    from abridgeai.features.learning_programs import services  # noqa: PLC0415

    # ``actor`` is the caller's real ``CurrentUser``, passed straight through
    # rather than reconstructed: ``list_programs`` reads only ``user_id`` today,
    # but fabricating a principal here would silently diverge the moment it
    # starts consulting permissions or the session. Typed ``object`` so this
    # module needs no import from ``core.security``.
    programs = await services.list_programs(
        db,
        organization_id=organization_id,
        actor=actor,  # type: ignore[arg-type]
    )
    return [
        {
            "id": program.id,
            "name": program.name,
            "slug": program.slug,
            "status": program.status,
            "organization_id": program.organization_id,
            "faculty_id": program.faculty_id,
            "student_count": program.student_count,
            "path_change_request_count": program.path_change_request_count,
            "has_draft_version": program.has_draft_version,
        }
        for program in programs
    ]


async def list_student_program_enrollments(
    db: AsyncSession, *, student_id: UUID
) -> list[dict[str, object]]:
    """Learning-program enrolments for ``student_id``, projected for display.

    Backs the manager/HOD user-detail "programs" section, the sibling of the
    career-path section right beside it.

    Returns plain dicts rather than ``ProgramEnrollmentRead`` so the schema
    stays inside this feature — the same contract
    ``career_paths.api.public.list_user_career_enrollments`` follows. Each row
    carries the enrolment, its progress against the pinned path version, and
    ``attempts``: the path the student picked, plus every path they switched
    away from. That attempt list IS the enrolment history — a switch is
    recorded as a new attempt rather than by mutating the old one.
    """
    from abridgeai.features.learning_programs import services  # noqa: PLC0415

    rows = await services.list_my_enrollments(db, student_id)
    out: list[dict[str, object]] = []
    for row in rows:
        # `PathAttemptRead` carries only `career_path_id`; the display name
        # lives on the version's path list, so resolve it here rather than
        # making the frontend fetch the program to label its own history.
        names = {p.career_path_id: p.name for p in row.paths}
        out.append(
            {
                "enrollment_id": row.id,
                "learning_program_id": row.learning_program_id,
                "program_name": row.program_name,
                "program_version_no": row.program_version_no,
                "status": row.status,
                "enrolled_at": row.enrolled_at,
                "completed_at": row.completed_at,
                "withdrawn_at": row.withdrawn_at,
                "completion_percent": row.current_progress_percent,
                "completed_courses": row.current_completed_courses,
                "course_count": row.current_total_courses,
                "max_path_switches": row.max_path_switches,
                "approved_switch_count": row.approved_switch_count,
                "attempts": [
                    {
                        "career_path_id": a.career_path_id,
                        "career_path_name": names.get(a.career_path_id),
                        "status": a.status,
                        "selected_at": a.selected_at,
                        "ended_at": a.ended_at,
                        "exit_snapshot": (
                            a.exit_snapshot.model_dump(mode="json")
                            if a.exit_snapshot is not None
                            else None
                        ),
                        "progress_percent": a.progress_percent,
                        "completed_courses": a.completed_courses,
                        "total_courses": a.total_courses,
                    }
                    for a in row.attempts
                ],
            }
        )
    return out


async def list_versions_with_active_attempts(
    db: AsyncSession, *, version_ids: list[UUID]
) -> set[UUID]:
    """Which of these career-path versions still have a student mid-path.

    A programme path attempt pins its own ``career_path_version_id``, so a
    version can be nobody's current one and still be the exact route somebody
    is walking. Anything deciding whether a version is safe to disturb has to
    ask here as well as asking career_paths.
    """
    if not version_ids:
        return set()
    stmt = select(ProgramPathAttempt.career_path_version_id).where(
        ProgramPathAttempt.career_path_version_id.in_(version_ids),
        ProgramPathAttempt.status == "active",
    )
    return set((await db.scalars(stmt)).all())


async def complete_program_attempts(
    db: AsyncSession, *, student_id: UUID, career_path_id: UUID
) -> int:
    """Complete matching paths, then programs whose every selected path is done."""
    stmt = (
        select(ProgramPathAttempt, ProgramEnrollment)
        .join(ProgramEnrollment, ProgramEnrollment.id == ProgramPathAttempt.program_enrollment_id)
        .where(
            ProgramEnrollment.student_id == student_id,
            ProgramEnrollment.status == "active",
            ProgramPathAttempt.career_path_id == career_path_id,
            ProgramPathAttempt.status == "active",
        )
        .with_for_update()
    )
    rows = list((await db.execute(stmt)).all())
    now = datetime.now(UTC)
    touched_enrollments: dict[UUID, ProgramEnrollment] = {}
    for attempt, enrollment in rows:
        # The same path can be pinned at different versions by different
        # programs. Never let a 100% result on one version complete them all.
        if not await career_paths_api.is_version_complete_for_user(
            db,
            version_id=attempt.career_path_version_id,
            student_id=student_id,
        ):
            continue
        attempt.status = "completed"
        attempt.ended_at = now
        attempt.updated_by = student_id
        touched_enrollments[enrollment.id] = enrollment
    if touched_enrollments:
        await flush_or_conflict(db)
    completed = 0
    for enrollment in touched_enrollments.values():
        remaining = await db.scalar(
            select(ProgramPathAttempt.id)
            .where(
                ProgramPathAttempt.program_enrollment_id == enrollment.id,
                ProgramPathAttempt.status == "active",
            )
            .limit(1)
        )
        if remaining is not None:
            continue
        enrollment.status = "completed"
        enrollment.completed_at = now
        enrollment.updated_by = student_id
        completed += 1
    if completed:
        await flush_or_conflict(db)
    return completed


async def ensure_completion_award(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
    source_enrollment_id: UUID,
) -> None:
    """Create the immutable academic completion used across path switches."""
    stmt = select(CourseCompletionAward).where(
        CourseCompletionAward.student_id == student_id,
        CourseCompletionAward.course_id == course_id,
    )
    award = (await db.scalars(stmt)).one_or_none()
    if award is None:
        db.add(
            CourseCompletionAward(
                student_id=student_id,
                course_id=course_id,
                source_enrollment_id=source_enrollment_id,
            )
        )
        await flush_or_conflict(db)


async def grant_active_path_entitlement(
    db: AsyncSession,
    *,
    student_id: UUID,
    career_path_id: UUID,
    course_enrollment_id: UUID,
    actor_id: UUID,
) -> None:
    """Attribute a lazy course start to the active program path attempt."""
    stmt = (
        select(ProgramPathAttempt)
        .join(ProgramEnrollment, ProgramEnrollment.id == ProgramPathAttempt.program_enrollment_id)
        .where(
            ProgramEnrollment.student_id == student_id,
            ProgramPathAttempt.career_path_id == career_path_id,
            ProgramPathAttempt.status == "active",
        )
    )
    attempts = list((await db.scalars(stmt)).all())
    for attempt in attempts:
        existing = await db.scalar(
            select(CourseEnrollmentEntitlement.id).where(
                CourseEnrollmentEntitlement.course_enrollment_id == course_enrollment_id,
                CourseEnrollmentEntitlement.source_type == "path_attempt",
                CourseEnrollmentEntitlement.source_id == attempt.id,
            )
        )
        if existing is None:
            db.add(
                CourseEnrollmentEntitlement(
                    course_enrollment_id=course_enrollment_id,
                    source_type="path_attempt",
                    source_id=attempt.id,
                    granted_at=datetime.now(UTC),
                    created_by=actor_id,
                )
            )
    await flush_or_conflict(db)


__all__ = [
    "complete_program_attempts",
    "list_versions_with_active_attempts",
    "list_program_governance_rows",
    "list_student_program_enrollments",
    "ensure_completion_award",
    "grant_active_path_entitlement",
]
