from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict, register_conflict_mappings
from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.core.runtime_settings import resolve_setting
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.career_paths.api import public as career_paths_api
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.learning_programs import queries
from abridgeai.features.learning_programs.models import (
    LearningProgram,
    LearningProgramVersion,
    LearningProgramVersionPath,
    PathChangeRequest,
    ProgramEnrollment,
    ProgramPathAttempt,
)
from abridgeai.features.learning_programs.schemas import (
    PathAttemptRead,
    PathChangeRequestRead,
    ProgramAuthoringOptions,
    ProgramCreate,
    ProgramCsvImportFailure,
    ProgramCsvImportResult,
    ProgramCsvImportRow,
    ProgramEnrollmentRead,
    ProgramOptionRead,
    ProgramPathRead,
    ProgramRead,
    ProgramUpdate,
    ProgramVersionRead,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser


register_conflict_mappings(
    {
        "uq_learning_programs_org_slug": "learning_program_slug_taken",
        "uq_program_enrollments_program_student": "student_already_enrolled_in_program",
        "uq_program_path_attempts_one_active": "program_already_has_an_active_path",
        "uq_path_change_requests_one_pending": "program_already_has_a_pending_path_change",
    }
)


def _now() -> datetime:
    return datetime.now(UTC)


async def _require_operator(db: AsyncSession, *, actor_id: UUID, program: LearningProgram) -> None:
    if not await queries.actor_has_program_role(
        db,
        user_id=actor_id,
        organization_id=program.organization_id,
        faculty_id=program.faculty_id,
        role_codes=("manager", "hod"),
    ):
        raise ForbiddenError("manager_or_faculty_dean_scope_required")


async def _require_owner_dean(
    db: AsyncSession, *, actor_id: UUID, program: LearningProgram
) -> None:
    if not await queries.actor_has_program_role(
        db,
        user_id=actor_id,
        organization_id=program.organization_id,
        faculty_id=program.faculty_id,
        role_codes=("hod",),
    ):
        raise ForbiddenError("active_faculty_dean_assignment_required")


async def _paths_for_version(db: AsyncSession, version_id: UUID) -> list[ProgramPathRead]:
    return [
        ProgramPathRead.model_validate(row)
        for row in await queries.list_version_paths(db, version_id)
    ]


async def _program_out(
    db: AsyncSession, program: LearningProgram, version: LearningProgramVersion | None = None
) -> ProgramRead:
    version = version or await queries.get_current_version(db, program.id)
    if version is None:
        raise NotFoundError("learning_program_version_not_found")
    publisher = (
        await identity_api.get_user_by_id(db, version.updated_by)
        if version.published_at is not None and version.updated_by is not None
        else None
    )
    version_out = ProgramVersionRead.model_validate(
        {
            **version.__dict__,
            "published_by": version.updated_by if version.published_at is not None else None,
            "published_by_name": publisher.display_name if publisher is not None else None,
        }
    )
    return ProgramRead.model_validate(
        {
            **program.__dict__,
            "current_version": version_out,
            "paths": await _paths_for_version(db, version.id),
        }
    )


async def get_authoring_options(db: AsyncSession, actor: CurrentUser) -> ProgramAuthoringOptions:
    primary_org = await access_control_api.get_user_primary_org(db, actor.user_id)
    if primary_org is None:
        return ProgramAuthoringOptions()
    faculties, paths, default_faculty_id = await queries.list_program_authoring_options(
        db, organization_id=primary_org.id, actor_id=actor.user_id
    )
    allowed_faculties = [
        faculty
        for faculty in faculties
        if await queries.actor_has_program_role(
            db,
            user_id=actor.user_id,
            organization_id=primary_org.id,
            faculty_id=faculty.id,
            role_codes=("manager", "hod"),
        )
    ]
    if not allowed_faculties:
        raise ForbiddenError("manager_or_faculty_dean_scope_required")
    return ProgramAuthoringOptions(
        faculties=[ProgramOptionRead(id=row.id, name=row.name) for row in allowed_faculties],
        career_paths=[
            ProgramOptionRead(id=row.id, name=row.name, slug=row.slug, description=row.description)
            for row in paths
        ],
        default_faculty_id=default_faculty_id,
    )


async def list_program_versions(
    db: AsyncSession, *, program_id: UUID, actor: CurrentUser
) -> list[ProgramVersionRead]:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    result: list[ProgramVersionRead] = []
    for version in await queries.list_versions(db, program_id):
        publisher = (
            await identity_api.get_user_by_id(db, version.updated_by)
            if version.published_at is not None and version.updated_by is not None
            else None
        )
        result.append(
            ProgramVersionRead.model_validate(
                {
                    **version.__dict__,
                    "published_by": version.updated_by
                    if version.published_at is not None
                    else None,
                    "published_by_name": publisher.display_name
                    if publisher is not None
                    else None,
                }
            )
        )
    return result


async def get_program_version(
    db: AsyncSession, *, program_id: UUID, version_id: UUID, actor: CurrentUser
) -> ProgramRead:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    version = await queries.get_version(db, version_id)
    if version is None or version.learning_program_id != program.id:
        raise NotFoundError("learning_program_version_not_found")
    return await _program_out(db, program, version)


async def create_program(
    db: AsyncSession, payload: ProgramCreate, actor: CurrentUser
) -> ProgramRead:
    primary_org = await access_control_api.get_user_primary_org(db, actor.user_id)
    if primary_org is None:
        raise ForbiddenError("primary_organization_required")
    organization_id = primary_org.id
    if not await queries.faculty_is_valid(db, payload.faculty_id, organization_id):
        raise ConflictError("faculty_must_belong_to_organization")
    probe = LearningProgram(
        organization_id=organization_id,
        faculty_id=payload.faculty_id,
        owner_faculty_dean_id=None,
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
        status="draft",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    await _require_operator(db, actor_id=actor.user_id, program=probe)
    resolved = await queries.resolve_published_path_versions(
        db,
        organization_id=organization_id,
        career_path_ids=payload.career_path_ids,
    )
    if len(resolved) != len(set(payload.career_path_ids)):
        raise ConflictError("all_paths_must_be_published_and_belong_to_the_program_organization")
    if any(path.status == "archived" for path, _version in resolved):
        raise ConflictError("archived_path_cannot_be_added")

    db.add(probe)
    await flush_or_conflict(db)
    version = LearningProgramVersion(
        learning_program_id=probe.id,
        version_no=1,
        status="draft",
        max_path_switches=payload.max_path_switches,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(version)
    await flush_or_conflict(db)
    for position, (path, path_version) in enumerate(resolved, start=1):
        db.add(
            LearningProgramVersionPath(
                program_version_id=version.id,
                career_path_id=path.id,
                career_path_version_id=path_version.id,
                position=position,
            )
        )
    await flush_or_conflict(db)
    return await _program_out(db, probe, version)


async def list_programs(
    db: AsyncSession, *, organization_id: UUID, actor: CurrentUser
) -> list[ProgramRead]:
    result: list[ProgramRead] = []
    for program in await queries.list_programs(db, organization_id):
        try:
            await _require_operator(db, actor_id=actor.user_id, program=program)
        except ForbiddenError:
            continue
        result.append(await _program_out(db, program))
    return result


async def get_program_for_operator(
    db: AsyncSession, *, program_id: UUID, actor: CurrentUser
) -> ProgramRead:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    return await _program_out(db, program)


async def update_program(
    db: AsyncSession, *, program_id: UUID, payload: ProgramUpdate, actor: CurrentUser
) -> ProgramRead:
    program = await queries.get_program(db, program_id, lock=True)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    if program.status == "archived":
        raise ConflictError("archived_program_is_immutable")
    if payload.name is not None:
        program.name = payload.name
    if payload.slug is not None:
        program.slug = payload.slug
    if "description" in payload.model_fields_set:
        program.description = payload.description
    program.updated_by = actor.user_id

    current = await queries.get_current_version(db, program.id)
    if current is None:
        raise NotFoundError("learning_program_version_not_found")
    if current.status == "published":
        source_paths = await queries.list_version_paths(db, current.id)
        draft = LearningProgramVersion(
            learning_program_id=program.id,
            version_no=current.version_no + 1,
            status="draft",
            max_path_switches=payload.max_path_switches
            if payload.max_path_switches is not None
            else current.max_path_switches,
            created_by=actor.user_id,
            updated_by=actor.user_id,
        )
        db.add(draft)
        await flush_or_conflict(db)
        path_ids = payload.career_path_ids
        if path_ids is None:
            for row in source_paths:
                db.add(
                    LearningProgramVersionPath(
                        program_version_id=draft.id,
                        career_path_id=row["career_path_id"],
                        career_path_version_id=row["career_path_version_id"],
                        position=row["position"],
                    )
                )
        else:
            await _replace_draft_paths(db, program, draft, path_ids)
        current = draft
    else:
        if payload.max_path_switches is not None:
            current.max_path_switches = payload.max_path_switches
        current.updated_by = actor.user_id
        if payload.career_path_ids is not None:
            await queries.delete_version_paths(db, current.id)
            await _replace_draft_paths(db, program, current, payload.career_path_ids)
    await flush_or_conflict(db)
    return await _program_out(db, program, current)


async def _replace_draft_paths(
    db: AsyncSession,
    program: LearningProgram,
    version: LearningProgramVersion,
    path_ids: list[UUID],
) -> None:
    if len(path_ids) != len(set(path_ids)):
        raise ConflictError("career_path_ids_must_be_unique")
    resolved = await queries.resolve_published_path_versions(
        db, organization_id=program.organization_id, career_path_ids=path_ids
    )
    if len(resolved) != len(path_ids):
        raise ConflictError("all_paths_must_be_published_and_not_archived")
    for position, (path, path_version) in enumerate(resolved, start=1):
        if path.status == "archived":
            raise ConflictError("archived_path_cannot_be_added")
        db.add(
            LearningProgramVersionPath(
                program_version_id=version.id,
                career_path_id=path.id,
                career_path_version_id=path_version.id,
                position=position,
            )
        )


async def publish_program(db: AsyncSession, *, program_id: UUID, actor: CurrentUser) -> ProgramRead:
    program = await queries.get_program(db, program_id, lock=True)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    if program.status == "archived":
        raise ConflictError("archived_program_cannot_be_published")
    version = await queries.get_current_version(db, program.id)
    if version is None or version.status != "draft":
        raise ConflictError("program_has_no_draft_version")
    paths = await queries.list_version_paths(db, version.id)
    if not paths:
        raise ConflictError("program_requires_at_least_one_path")
    version.status = "published"
    version.published_at = _now()
    version.updated_by = actor.user_id
    program.status = "published"
    program.updated_by = actor.user_id
    await flush_or_conflict(db)
    return await _program_out(db, program, version)


async def archive_program(db: AsyncSession, *, program_id: UUID, actor: CurrentUser) -> ProgramRead:
    program = await queries.get_program(db, program_id, lock=True)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    program.status = "archived"
    program.updated_by = actor.user_id
    await flush_or_conflict(db)
    return await _program_out(db, program)


async def import_students_from_csv(
    db: AsyncSession,
    *,
    program_id: UUID,
    rows: list[dict[str, str]],
    actor: CurrentUser,
) -> ProgramCsvImportResult:
    """Create accounts as needed and enrol a whole roster into the program.

    Deliberately NOT built on :func:`enroll_students`. That one is
    all-or-nothing: a single already-enrolled student or a missing role
    aborts the batch. That is right for a hand-picked selection, and wrong
    for a file — a roster with one duplicate or one typo would import
    nothing, and the manager gets no clue which line was at fault.

    Here every row is validated and applied independently:

    * an unknown email creates a student account in the program's org (same
      admin-invite path a manual invite uses);
    * a known email is reused untouched — a roster file is not authority
      over an existing account's name or role;
    * an already-enrolled student is reported in ``already_enrolled``, not
      failed, because re-uploading last week's file is a normal thing to do;
    * anything else lands in ``failures`` with its row number and reason.

    The concurrent-enrollment cap is enforced per row exactly as the
    hand-picked path enforces it, so an import cannot be used to sidestep it.
    """
    program = await queries.get_program(db, program_id, lock=True)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    if program.status != "published":
        raise ConflictError("only_published_programs_accept_enrollments")
    version = await queries.get_current_version(db, program.id, published_only=True)
    if version is None:
        raise ConflictError("program_has_no_published_version")

    limit = int(
        await resolve_setting(
            db,
            "learning_program.max_concurrent_enrollments",
            organization_id=program.organization_id,
        )
    )

    result = ProgramCsvImportResult()
    seen_emails: set[str] = set()

    for row_number, raw in enumerate(rows, start=1):
        try:
            parsed = ProgramCsvImportRow.model_validate(raw)
        except (ValueError, TypeError) as exc:
            result.failures.append(
                ProgramCsvImportFailure(
                    row_number=row_number,
                    identifier=str(raw.get("email", "")) or None,
                    reason=f"invalid_row: {exc.__class__.__name__}",
                )
            )
            continue

        email = parsed.email.strip().lower()
        # A file that lists the same person twice should import them once,
        # not fail the second line with "already enrolled".
        if email in seen_emails:
            continue
        seen_emails.add(email)

        try:
            student_id, created = await identity_api.find_or_create_student(
                db,
                email=email,
                organization_id=program.organization_id,
                actor_id=actor.user_id,
                given_name=parsed.given_name,
                family_name=parsed.family_name,
                display_name=parsed.display_name,
            )
            if created:
                result.created_users.append(student_id)

            existing = await queries.get_program_enrollment(db, program.id, student_id)
            if existing is not None and existing.status in (
                "awaiting_path",
                "active",
                "completed",
            ):
                result.already_enrolled.append(student_id)
                continue

            concurrent = await queries.count_concurrent_enrollments(
                db, organization_id=program.organization_id, student_id=student_id
            )
            if concurrent >= limit:
                result.failures.append(
                    ProgramCsvImportFailure(
                        row_number=row_number,
                        identifier=email,
                        reason="max_concurrent_enrollments_reached",
                    )
                )
                continue

            # Same row shape the hand-picked path writes, including the
            # re-enrol branch: a withdrawn student re-appearing in a roster
            # file is reinstated onto the current version rather than
            # colliding with their old row.
            if existing is None:
                db.add(
                    ProgramEnrollment(
                        learning_program_id=program.id,
                        program_version_id=version.id,
                        student_id=student_id,
                        status="awaiting_path",
                        created_by=actor.user_id,
                        updated_by=actor.user_id,
                    )
                )
            else:
                existing.program_version_id = version.id
                existing.status = "awaiting_path"
                existing.enrolled_at = _now()
                existing.withdrawn_at = None
                existing.withdrawal_reason = None
                existing.updated_by = actor.user_id
            await flush_or_conflict(db)
            result.enrolled.append(student_id)
        except (ConflictError, NotFoundError, ValueError) as exc:
            result.failures.append(
                ProgramCsvImportFailure(
                    row_number=row_number, identifier=email, reason=str(exc)
                )
            )

    return result


async def enroll_students(
    db: AsyncSession, *, program_id: UUID, student_ids: list[UUID], actor: CurrentUser
) -> list[ProgramEnrollmentRead]:
    program = await queries.get_program(db, program_id, lock=True)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    if program.status != "published":
        raise ConflictError("only_published_programs_accept_enrollments")
    version = await queries.get_current_version(db, program.id, published_only=True)
    if version is None:
        raise ConflictError("program_has_no_published_version")
    roles = await access_control_api.get_role_codes_for_users(db, student_ids)
    bad = [student_id for student_id in student_ids if "student" not in roles.get(student_id, ())]
    if bad:
        raise ConflictError("all_enrollees_must_have_the_student_role")
    limit = int(
        await resolve_setting(
            db,
            "learning_program.max_concurrent_enrollments",
            organization_id=program.organization_id,
        )
    )
    result: list[ProgramEnrollmentRead] = []
    for student_id in dict.fromkeys(student_ids):
        existing = await queries.get_program_enrollment(db, program.id, student_id)
        if existing is not None and existing.status in ("awaiting_path", "active", "completed"):
            raise ConflictError(f"student_already_enrolled_in_program:{student_id}")
        concurrent = await queries.count_concurrent_enrollments(
            db, organization_id=program.organization_id, student_id=student_id
        )
        if concurrent >= limit:
            raise ConflictError(f"concurrent_program_limit_reached:{student_id}:{limit}")
        if existing is None:
            existing = ProgramEnrollment(
                learning_program_id=program.id,
                program_version_id=version.id,
                student_id=student_id,
                status="awaiting_path",
                created_by=actor.user_id,
                updated_by=actor.user_id,
            )
            db.add(existing)
        else:
            existing.program_version_id = version.id
            existing.status = "awaiting_path"
            existing.enrolled_at = _now()
            existing.withdrawn_at = None
            existing.withdrawal_reason = None
            existing.updated_by = actor.user_id
        await flush_or_conflict(db)
        result.append(await _enrollment_out(db, existing))
    return result


async def withdraw_student(
    db: AsyncSession,
    *,
    program_id: UUID,
    student_id: UUID,
    reason: str,
    actor: CurrentUser,
) -> ProgramEnrollmentRead:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    enrollment = await queries.get_program_enrollment(db, program.id, student_id)
    if enrollment is None:
        raise NotFoundError("program_enrollment_not_found")
    if enrollment.status == "completed":
        raise ConflictError("completed_program_cannot_be_withdrawn")
    enrollment.status = "withdrawn"
    enrollment.withdrawn_at = _now()
    enrollment.withdrawal_reason = reason
    enrollment.updated_by = actor.user_id
    attempt = await queries.get_active_attempt(db, enrollment.id, lock=True)
    if attempt is not None:
        attempt.exit_snapshot = await queries.build_exit_snapshot(
            db, student_id=student_id, attempt=attempt
        )
        attempt.status = "cancelled"
        attempt.ended_at = _now()
        attempt.updated_by = actor.user_id
        await queries.revoke_path_entitlements(db, attempt_id=attempt.id)
        if not await queries.count_other_active_path_attempts(
            db,
            student_id=student_id,
            career_path_id=attempt.career_path_id,
            excluding_attempt_id=attempt.id,
        ):
            await career_paths_api.release_program_path_access(
                db,
                student_id=student_id,
                career_path_id=attempt.career_path_id,
                actor_id=actor.user_id,
            )
    pending = await queries.get_pending_request(db, enrollment.id)
    if pending is not None:
        pending.status = "cancelled"
        pending.reviewed_at = _now()
        pending.decision_reason = "program_enrollment_withdrawn"
        pending.updated_by = actor.user_id
    await flush_or_conflict(db)
    return await _enrollment_out(db, enrollment)


async def _enrollment_out(db: AsyncSession, enrollment: ProgramEnrollment) -> ProgramEnrollmentRead:
    program = await queries.get_program(db, enrollment.learning_program_id)
    version = await queries.get_version(db, enrollment.program_version_id)
    if program is None or version is None:
        raise NotFoundError("program_enrollment_parent_not_found")
    attempts = await queries.list_attempts(db, enrollment.id)
    pending = await queries.get_pending_request(db, enrollment.id)
    active_attempt = next((row for row in attempts if row.status == "active"), None)
    progress_rows: list[dict[str, object]] = []
    if active_attempt is not None:
        progress_rows = await career_paths_api.get_version_course_progress_for_user(
            db,
            version_id=active_attempt.career_path_version_id,
            student_id=enrollment.student_id,
        )
    completed_courses = sum(bool(row.get("satisfied")) for row in progress_rows)
    total_courses = len(progress_rows)
    return ProgramEnrollmentRead.model_validate(
        {
            **enrollment.__dict__,
            "program_name": program.name,
            "program_version_no": version.version_no,
            "max_path_switches": version.max_path_switches,
            "approved_switch_count": await queries.count_approved_switches(db, enrollment.id),
            "current_progress_percent": round(
                (completed_courses / total_courses * 100) if total_courses else 0, 2
            ),
            "current_completed_courses": completed_courses,
            "current_total_courses": total_courses,
            "paths": await _paths_for_version(db, version.id),
            "attempts": [PathAttemptRead.model_validate(row) for row in attempts],
            "pending_change_request": (
                PathChangeRequestRead.model_validate(pending).model_dump(mode="json")
                if pending is not None
                else None
            ),
        }
    )


async def list_my_enrollments(db: AsyncSession, student_id: UUID) -> list[ProgramEnrollmentRead]:
    return [
        await _enrollment_out(db, row)
        for row in await queries.list_student_enrollments(db, student_id)
    ]


async def list_roster(
    db: AsyncSession, *, program_id: UUID, actor: CurrentUser
) -> list[ProgramEnrollmentRead]:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    return [
        await _enrollment_out(db, row)
        for row in await queries.list_program_enrollments(db, program_id)
    ]


async def select_path(
    db: AsyncSession, *, enrollment_id: UUID, career_path_id: UUID, student_id: UUID
) -> ProgramEnrollmentRead:
    enrollment = await queries.get_enrollment(db, enrollment_id, lock=True)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("program_enrollment_not_found")
    if enrollment.status != "awaiting_path":
        raise ConflictError("initial_path_can_only_be_selected_once")
    paths = await queries.list_version_paths(db, enrollment.program_version_id)
    target = next((row for row in paths if row["career_path_id"] == career_path_id), None)
    if target is None:
        raise ConflictError("path_is_not_in_the_pinned_program_version")
    if target["status"] == "archived":
        raise ConflictError("archived_path_cannot_be_selected")
    attempt = ProgramPathAttempt(
        program_enrollment_id=enrollment.id,
        career_path_id=career_path_id,
        career_path_version_id=target["career_path_version_id"],
        status="active",
        created_by=student_id,
        updated_by=student_id,
    )
    db.add(attempt)
    enrollment.status = "active"
    enrollment.updated_by = student_id
    await flush_or_conflict(db)
    await career_paths_api.ensure_program_path_access(
        db,
        student_id=student_id,
        career_path_id=career_path_id,
        version_id=cast(UUID, target["career_path_version_id"]),
        actor_id=student_id,
    )
    return await _enrollment_out(db, enrollment)


async def request_path_change(
    db: AsyncSession,
    *,
    enrollment_id: UUID,
    target_path_id: UUID,
    reason: str,
    student_id: UUID,
) -> PathChangeRequestRead:
    enrollment = await queries.get_enrollment(db, enrollment_id, lock=True)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("program_enrollment_not_found")
    if enrollment.status != "active":
        raise ConflictError("only_active_programs_can_change_path")
    attempt = await queries.get_active_attempt(db, enrollment.id, lock=True)
    if attempt is None:
        raise ConflictError("active_path_attempt_not_found")
    if attempt.career_path_id == target_path_id:
        raise ConflictError("target_path_must_differ_from_current_path")
    if await queries.get_pending_request(db, enrollment.id) is not None:
        raise ConflictError("program_already_has_a_pending_path_change")
    version = await queries.get_version(db, enrollment.program_version_id)
    if version is None:
        raise NotFoundError("program_version_not_found")
    if await queries.count_approved_switches(db, enrollment.id) >= version.max_path_switches:
        raise ConflictError("path_switch_limit_reached")
    paths = await queries.list_version_paths(db, enrollment.program_version_id)
    target = next((row for row in paths if row["career_path_id"] == target_path_id), None)
    if target is None:
        raise ConflictError("target_path_is_not_in_the_pinned_program_version")
    if target["status"] == "archived":
        raise ConflictError("target_path_archived")
    request = PathChangeRequest(
        program_enrollment_id=enrollment.id,
        from_attempt_id=attempt.id,
        target_career_path_id=target_path_id,
        target_career_path_version_id=target["career_path_version_id"],
        reason=reason,
        status="pending",
        created_by=student_id,
        updated_by=student_id,
    )
    db.add(request)
    await flush_or_conflict(db)
    return PathChangeRequestRead.model_validate(request)


async def cancel_change_request(
    db: AsyncSession, *, request_id: UUID, student_id: UUID
) -> PathChangeRequestRead:
    request = await queries.get_change_request(db, request_id, lock=True)
    if request is None:
        raise NotFoundError("path_change_request_not_found")
    enrollment = await queries.get_enrollment(db, request.program_enrollment_id)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("path_change_request_not_found")
    if request.status != "pending":
        raise ConflictError("only_pending_requests_can_be_cancelled")
    request.status = "cancelled"
    request.reviewed_at = _now()
    request.updated_by = student_id
    await flush_or_conflict(db)
    return PathChangeRequestRead.model_validate(request)


async def list_change_requests(
    db: AsyncSession, *, program_id: UUID, actor: CurrentUser
) -> list[PathChangeRequestRead]:
    program = await queries.get_program(db, program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_operator(db, actor_id=actor.user_id, program=program)
    return [
        PathChangeRequestRead.model_validate(row)
        for row in await queries.list_program_change_requests(db, program_id)
    ]


async def decide_change_request(  # noqa: C901 - approval is one atomic invariant set
    db: AsyncSession,
    *,
    request_id: UUID,
    approve: bool,
    decision_reason: str | None,
    actor: CurrentUser,
) -> PathChangeRequestRead:
    request = await queries.get_change_request(db, request_id, lock=True)
    if request is None:
        raise NotFoundError("path_change_request_not_found")
    enrollment = await queries.get_enrollment(db, request.program_enrollment_id, lock=True)
    if enrollment is None:
        raise NotFoundError("program_enrollment_not_found")
    program = await queries.get_program(db, enrollment.learning_program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_owner_dean(db, actor_id=actor.user_id, program=program)
    if actor.user_id == enrollment.student_id:
        raise ForbiddenError("self_approval_is_not_allowed")
    if request.status != "pending":
        raise ConflictError("request_is_not_pending")
    if not approve:
        request.status = "rejected"
        request.reviewed_by = actor.user_id
        request.reviewed_at = _now()
        request.decision_reason = decision_reason
        request.updated_by = actor.user_id
        await flush_or_conflict(db)
        return PathChangeRequestRead.model_validate(request)
    if enrollment.status != "active":
        raise ConflictError("program_is_not_active")
    attempt = await queries.get_active_attempt(db, enrollment.id, lock=True)
    if attempt is None or attempt.id != request.from_attempt_id:
        raise ConflictError("active_path_changed_since_request")
    version = await queries.get_version(db, enrollment.program_version_id)
    if version is None:
        raise NotFoundError("program_version_not_found")
    if await queries.count_approved_switches(db, enrollment.id) >= version.max_path_switches:
        raise ConflictError("path_switch_limit_reached")
    paths = await queries.list_version_paths(db, enrollment.program_version_id)
    target = next(
        (row for row in paths if row["career_path_id"] == request.target_career_path_id), None
    )
    if target is None or target["status"] == "archived":
        request.status = "invalidated"
        request.reviewed_by = actor.user_id
        request.reviewed_at = _now()
        request.decision_reason = "target_path_archived"
        request.updated_by = actor.user_id
        await flush_or_conflict(db)
        return PathChangeRequestRead.model_validate(request)

    attempt.exit_snapshot = await queries.build_exit_snapshot(
        db, student_id=enrollment.student_id, attempt=attempt
    )
    attempt.status = "switched_out"
    attempt.ended_at = _now()
    attempt.updated_by = actor.user_id
    new_attempt = ProgramPathAttempt(
        program_enrollment_id=enrollment.id,
        career_path_id=request.target_career_path_id,
        career_path_version_id=request.target_career_path_version_id,
        previous_attempt_id=attempt.id,
        status="active",
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(new_attempt)
    await flush_or_conflict(db)
    await queries.transfer_path_entitlements(
        db,
        old_attempt_id=attempt.id,
        new_attempt_id=new_attempt.id,
        new_path_version_id=new_attempt.career_path_version_id,
        actor_id=actor.user_id,
    )
    request.status = "approved"
    request.reviewed_by = actor.user_id
    request.reviewed_at = _now()
    request.decision_reason = decision_reason
    request.new_attempt_id = new_attempt.id
    request.updated_by = actor.user_id
    await career_paths_api.ensure_program_path_access(
        db,
        student_id=enrollment.student_id,
        career_path_id=new_attempt.career_path_id,
        version_id=new_attempt.career_path_version_id,
        actor_id=actor.user_id,
    )
    if not await queries.count_other_active_path_attempts(
        db,
        student_id=enrollment.student_id,
        career_path_id=attempt.career_path_id,
        excluding_attempt_id=attempt.id,
    ):
        await career_paths_api.release_program_path_access(
            db,
            student_id=enrollment.student_id,
            career_path_id=attempt.career_path_id,
            actor_id=actor.user_id,
        )
    await flush_or_conflict(db)
    return PathChangeRequestRead.model_validate(request)


__all__ = [
    "archive_program",
    "cancel_change_request",
    "create_program",
    "decide_change_request",
    "enroll_students",
    "get_program_for_operator",
    "list_change_requests",
    "list_my_enrollments",
    "list_programs",
    "list_roster",
    "publish_program",
    "request_path_change",
    "select_path",
    "update_program",
    "withdraw_student",
]
