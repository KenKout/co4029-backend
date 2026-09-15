from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict, register_conflict_mappings
from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.core.runtime_settings import resolve_setting
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.models import CareerPath
from abridgeai.features.career_paths.api import public as career_paths_api
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.learning_programs import notify, queries
from abridgeai.features.learning_programs.models import (
    PATH_CHANGE_OPEN_STATUSES,
    PATH_CHANGE_REJECTION_REASON_CODES,
    LearningProgram,
    LearningProgramVersion,
    LearningProgramVersionPath,
    PathChangeRequest,
    ProgramEnrollment,
    ProgramPathAttempt,
)
from abridgeai.features.learning_programs.schemas import (
    CareerPathOptionRead,
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
        "uq_program_path_attempts_active_path": "path_already_selected",
        "uq_path_change_requests_one_pending": "program_already_has_a_pending_path_change",
    }
)


class ProgramConflictError(ConflictError):
    """A 409 that carries a HUMAN sentence plus structured fields.

    The rest of this module raises ``ConflictError("some_machine_code")`` and
    the router serializes ``str(exc)`` into ``detail.error``, which is fine
    for codes the SPA branches on. It stopped being fine once codes started
    carrying identifiers: a manager hitting the concurrency cap was shown
    ``concurrent_program_limit_reached:71acb8a2-…:1`` verbatim in a toast.

    So: ``code`` stays the stable machine string (FE branching, tests),
    ``message`` is the sentence a manager can act on, and ``fields`` carries
    the ids/limits the SPA may want to render itself. The router emits all
    three as ``{"error": code, "message": message, **fields}``.
    """

    def __init__(self, code: str, message: str, **fields: object) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.fields = fields


def _now() -> datetime:
    return datetime.now(UTC)


async def _student_label(db: AsyncSession, student_id: UUID) -> str:
    """Name (or email) for an error sentence; the raw id as a last resort."""
    user = await identity_api.get_user_by_id(db, student_id)
    if user is None:
        return str(student_id)
    return (user.display_name or "").strip() or user.primary_email


async def _notify_owning_deans(
    db: AsyncSession,
    *,
    program: LearningProgram,
    request_id: UUID,
    student_id: UUID,
    subject_path_id: UUID,
    kind: str = "change",
    arq_pool: object | None = None,
) -> None:
    """Fan out the "student filed a path request" notification.

    ``subject_path_id`` is the path the message is about: where the student
    wants to go for a change, and the one they want to end for a drop.

    Pull-based badges on the program card exist, but a filed request is
    exactly the push moment: without it the dean only finds out by
    browsing. Notification failures are swallowed inside ``notify`` — a
    notification must never roll back the request creation.
    """
    dean_ids = await queries.list_owning_deans(
        db,
        organization_id=program.organization_id,
        faculty_id=program.faculty_id,
    )
    student_label = await _student_label(db, student_id)
    subject_path_name = await _target_path_name(db, subject_path_id)
    for dean_id in dean_ids:
        if dean_id == student_id:
            continue
        await notify.notify_dean_path_change_requested(
            db,
            dean_user_id=dean_id,
            request_id=request_id,
            program_id=program.id,
            program_name=program.name,
            student_label=student_label,
            target_path_name=subject_path_name,
            kind=kind,
            arq_pool=arq_pool,
        )


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
    rows = await queries.list_version_paths(db, version_id)
    thumbnail_urls = await career_paths_api.get_career_path_thumbnail_urls(
        db, [cast(UUID, row["career_path_id"]) for row in rows]
    )
    return [
        ProgramPathRead.model_validate(
            {
                **row,
                "thumbnail_url": thumbnail_urls.get(cast(UUID, row["career_path_id"])),
            }
        )
        for row in rows
    ]


async def _target_path_name(db: AsyncSession, career_path_id: UUID) -> str:
    """Career-path display name for student-facing notification copy.

    Falls back to a generic noun instead of raising: a request whose target path
    row has since disappeared should still produce a readable notification, and
    the request record itself keeps the ID.
    """
    return await queries.get_career_path_name(db, career_path_id) or "the requested path"


async def _career_path_limit_ceiling(db: AsyncSession, organization_id: UUID) -> int:
    """Organization guardrail for manager-owned per-program path limits."""
    return int(
        await resolve_setting(
            db,
            "learning_program.max_career_paths_per_enrollment",
            organization_id=organization_id,
        )
    )


async def _require_path_limit_within_ceiling(
    db: AsyncSession, *, organization_id: UUID, requested: int
) -> int:
    ceiling = await _career_path_limit_ceiling(db, organization_id)
    if requested > ceiling:
        raise ProgramConflictError(
            "career_path_limit_exceeds_organization_ceiling",
            f"This organization allows at most {ceiling} career path"
            f"{'s' if ceiling != 1 else ''} per learning program.",
            requested=requested,
            ceiling=ceiling,
        )
    return ceiling


async def _program_out(
    db: AsyncSession, program: LearningProgram, version: LearningProgramVersion | None = None
) -> ProgramRead:
    version = version or await queries.get_current_version(db, program.id)
    if version is None:
        raise NotFoundError("learning_program_version_not_found")
    # Any UPDATE through TimestampMixin expires ``updated_at`` (its onupdate
    # is a server-side NOW()). The __dict__ spreads below read the raw
    # instance dict and never trigger a lazy load, so an expired column
    # silently vanishes and ProgramRead rejects the payload with
    # ``Field required [type=missing]`` (observed as a 500 on
    # POST .../publish right after a successful flush). Refresh re-loads
    # both rows inside the current transaction before serializing.
    await db.refresh(program)
    await db.refresh(version)
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
    # The picker must not offer a path the attach gate would reject. The
    # options query only returns published paths WITH a published version;
    # everything else this org owns is fetched here and surfaced as
    # selectable=False with a reason, so a manager sees "Data Engineer
    # (draft — publish it first)" instead of picking it and getting a 409.
    all_org_paths = await queries.list_all_org_paths(db, organization_id=primary_org.id)
    selectable_ids = {path.id for path in paths}
    career_path_options = [
        ProgramOptionFactory.career_path_option(path, selectable=path.id in selectable_ids)
        for path in all_org_paths
    ]
    return ProgramAuthoringOptions(
        faculties=[ProgramOptionRead(id=row.id, name=row.name) for row in allowed_faculties],
        career_paths=career_path_options,
        default_faculty_id=default_faculty_id,
        max_career_paths_per_program=await _career_path_limit_ceiling(db, primary_org.id),
    )


class ProgramOptionFactory:
    """Builds CareerPathOptionRead rows; kept as a tiny helper so the reason
    string lives in exactly one place."""

    @staticmethod
    def career_path_option(path: CareerPath, *, selectable: bool) -> CareerPathOptionRead:
        return CareerPathOptionRead(
            id=path.id,
            name=path.name,
            slug=path.slug,
            description=path.description,
            selectable=selectable,
            not_selectable_reason=None if selectable else "path_not_published",
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
                    "published_by_name": publisher.display_name if publisher is not None else None,
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
    await _require_path_limit_within_ceiling(
        db,
        organization_id=organization_id,
        requested=payload.max_career_paths_per_enrollment,
    )
    if not await queries.faculty_is_valid(db, payload.faculty_id, organization_id):
        raise ConflictError("faculty_must_belong_to_organization")
    probe = LearningProgram(
        organization_id=organization_id,
        faculty_id=payload.faculty_id,
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
        max_career_paths_per_enrollment=payload.max_career_paths_per_enrollment,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(version)
    await flush_or_conflict(db)
    default_career_path_id = payload.default_career_path_id
    if default_career_path_id is None and len(resolved) == 1:
        default_career_path_id = resolved[0][0].id
    for position, (path, path_version) in enumerate(resolved, start=1):
        db.add(
            LearningProgramVersionPath(
                program_version_id=version.id,
                career_path_id=path.id,
                career_path_version_id=path_version.id,
                position=position,
                is_default=path.id == default_career_path_id,
            )
        )
    await flush_or_conflict(db)
    return await _program_out(db, probe, version)


async def list_programs(
    db: AsyncSession, *, organization_id: UUID, actor: CurrentUser
) -> list[ProgramRead]:
    programs = await queries.list_programs(db, organization_id)
    visible = [
        program for program in programs if await _actor_can_operate(db, actor.user_id, program)
    ]
    if not visible:
        return []
    cards = await queries.list_program_list_cards(db, [p.id for p in visible])
    result: list[ProgramRead] = []
    for program in visible:
        dto = await _program_out(db, program)
        stats = cards.get(program.id, {})
        result.append(
            dto.model_copy(
                update={
                    "student_count": stats.get("student_count", 0),
                    "path_change_request_count": stats.get("path_change_request_count", 0),
                    "has_draft_version": bool(stats.get("has_draft_version", False)),
                }
            )
        )
    return result


async def _actor_can_operate(db: AsyncSession, actor_id: UUID, program: LearningProgram) -> bool:
    try:
        await _require_operator(db, actor_id=actor_id, program=program)
        return True
    except ForbiddenError:
        return False


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
    if payload.max_career_paths_per_enrollment is not None:
        await _require_path_limit_within_ceiling(
            db,
            organization_id=program.organization_id,
            requested=payload.max_career_paths_per_enrollment,
        )
    if current.status == "published":
        source_paths = await queries.list_version_paths(db, current.id)
        draft = LearningProgramVersion(
            learning_program_id=program.id,
            version_no=current.version_no + 1,
            status="draft",
            max_path_switches=payload.max_path_switches
            if payload.max_path_switches is not None
            else current.max_path_switches,
            max_career_paths_per_enrollment=(
                payload.max_career_paths_per_enrollment
                if payload.max_career_paths_per_enrollment is not None
                else current.max_career_paths_per_enrollment
            ),
            created_by=actor.user_id,
            updated_by=actor.user_id,
        )
        db.add(draft)
        await flush_or_conflict(db)
        path_ids = (
            payload.career_path_ids
            if payload.career_path_ids is not None
            else [cast(UUID, row["career_path_id"]) for row in source_paths]
        )
        default_path_id = _resolve_draft_default(
            payload=payload,
            path_ids=path_ids,
            existing_paths=source_paths,
        )
        await _replace_draft_paths(
            db,
            program,
            draft,
            path_ids,
            default_career_path_id=default_path_id,
            existing_paths=source_paths,
        )
        current = draft
    else:
        if payload.max_path_switches is not None:
            current.max_path_switches = payload.max_path_switches
        if payload.max_career_paths_per_enrollment is not None:
            current.max_career_paths_per_enrollment = (
                payload.max_career_paths_per_enrollment
            )
        current.updated_by = actor.user_id
        if (
            payload.career_path_ids is not None
            or "default_career_path_id" in payload.model_fields_set
        ):
            source_paths = await queries.list_version_paths(db, current.id)
            path_ids = (
                payload.career_path_ids
                if payload.career_path_ids is not None
                else [cast(UUID, row["career_path_id"]) for row in source_paths]
            )
            default_path_id = _resolve_draft_default(
                payload=payload,
                path_ids=path_ids,
                existing_paths=source_paths,
            )
            await queries.delete_version_paths(db, current.id)
            await _replace_draft_paths(
                db,
                program,
                current,
                path_ids,
                default_career_path_id=default_path_id,
                existing_paths=source_paths,
            )
    await flush_or_conflict(db)
    return await _program_out(db, program, current)


async def _replace_draft_paths(
    db: AsyncSession,
    program: LearningProgram,
    version: LearningProgramVersion,
    path_ids: list[UUID],
    *,
    default_career_path_id: UUID | None = None,
    existing_paths: list[dict[str, object]] | None = None,
) -> None:
    if len(path_ids) != len(set(path_ids)):
        raise ConflictError("career_path_ids_must_be_unique")
    if default_career_path_id is None and len(path_ids) == 1:
        default_career_path_id = path_ids[0]
    if default_career_path_id is not None and default_career_path_id not in path_ids:
        raise ConflictError("default_path_must_belong_to_program_version")

    existing_by_id = {cast(UUID, row["career_path_id"]): row for row in (existing_paths or [])}
    added_path_ids = [path_id for path_id in path_ids if path_id not in existing_by_id]
    resolved = await queries.resolve_published_path_versions(
        db,
        organization_id=program.organization_id,
        career_path_ids=added_path_ids,
    )
    if len(resolved) != len(added_path_ids):
        raise ConflictError("all_paths_must_be_published_and_not_archived")
    resolved_by_id = {path.id: (path, path_version) for path, path_version in resolved}

    for position, path_id in enumerate(path_ids, start=1):
        existing = existing_by_id.get(path_id)
        if existing is not None:
            db.add(
                LearningProgramVersionPath(
                    program_version_id=version.id,
                    career_path_id=path_id,
                    career_path_version_id=cast(UUID, existing["career_path_version_id"]),
                    position=position,
                    is_default=path_id == default_career_path_id,
                )
            )
            continue

        path, path_version = resolved_by_id[path_id]
        if path.status == "archived":
            raise ConflictError("archived_path_cannot_be_added")
        db.add(
            LearningProgramVersionPath(
                program_version_id=version.id,
                career_path_id=path.id,
                career_path_version_id=path_version.id,
                position=position,
                is_default=path.id == default_career_path_id,
            )
        )


def _resolve_draft_default(
    *,
    payload: ProgramUpdate,
    path_ids: list[UUID],
    existing_paths: list[dict[str, object]],
) -> UUID | None:
    existing_default = next(
        (
            cast(UUID, row["career_path_id"])
            for row in existing_paths
            if bool(row.get("is_default"))
        ),
        None,
    )
    if "default_career_path_id" in payload.model_fields_set:
        requested = payload.default_career_path_id
        if requested is None and existing_default is not None:
            raise ConflictError("default_path_must_be_replaced_before_removal")
        if requested is not None and requested not in path_ids:
            raise ConflictError("default_path_must_belong_to_program_version")
        return requested
    if existing_default is not None and existing_default not in path_ids:
        raise ConflictError("default_path_must_be_replaced_before_removal")
    return existing_default


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
    if version.max_career_paths_per_enrollment > len(paths):
        raise ProgramConflictError(
            "career_path_limit_exceeds_program_paths",
            "The student path limit cannot exceed the number of Career Paths "
            "attached to this program.",
            requested=version.max_career_paths_per_enrollment,
            path_count=len(paths),
        )
    await _require_path_limit_within_ceiling(
        db,
        organization_id=program.organization_id,
        requested=version.max_career_paths_per_enrollment,
    )
    if sum(bool(path["is_default"]) for path in paths) != 1:
        raise ConflictError("program_requires_exactly_one_default_path")
    invalid_path_ids = await queries.list_unpublishable_version_path_ids(
        db,
        version_id=version.id,
        organization_id=program.organization_id,
    )
    if invalid_path_ids:
        raise ConflictError("program_contains_unavailable_paths")
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
    default_path = await _get_enrollable_default_path(db, version.id)

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
                        reason=(
                            f"Already in {concurrent} active learning program(s); "
                            f"this organization allows at most {limit} at a time."
                        ),
                    )
                )
                continue

            # Same row shape the hand-picked path writes, including the
            # re-enrol branch: a withdrawn student re-appearing in a roster
            # file is reinstated onto the current version rather than
            # colliding with their old row.
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
            await _activate_default_path(
                db,
                enrollment=existing,
                actor_id=actor.user_id,
                default_path=default_path,
            )
            result.enrolled.append(student_id)
        except (ConflictError, NotFoundError, ValueError) as exc:
            result.failures.append(
                ProgramCsvImportFailure(row_number=row_number, identifier=email, reason=str(exc))
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
    default_path = await _get_enrollable_default_path(db, version.id)
    roles = await access_control_api.get_role_codes_for_users(db, student_ids)
    bad = [student_id for student_id in student_ids if "student" not in roles.get(student_id, ())]
    if bad:
        names = ", ".join([await _student_label(db, student_id) for student_id in bad[:5]])
        more = f" (and {len(bad) - 5} more)" if len(bad) > 5 else ""
        raise ProgramConflictError(
            "all_enrollees_must_have_the_student_role",
            f"{names}{more} do not have the Student role, so they cannot be enrolled. "
            "Give them the Student role first, or remove them from the selection.",
            student_ids=[str(student_id) for student_id in bad],
        )
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
            raise ProgramConflictError(
                "student_already_enrolled_in_program",
                f"{await _student_label(db, student_id)} is already enrolled in this program. "
                "Remove them from the selection and try again.",
                student_id=str(student_id),
            )
        concurrent = await queries.count_concurrent_enrollments(
            db, organization_id=program.organization_id, student_id=student_id
        )
        if concurrent >= limit:
            raise ProgramConflictError(
                "concurrent_program_limit_reached",
                f"{await _student_label(db, student_id)} is already in {concurrent} active "
                f"learning program(s), and this organization allows at most {limit} at a time. "
                "Withdraw them from a program first, or raise the limit in "
                "Settings → Learning programs.",
                student_id=str(student_id),
                limit=limit,
                current=concurrent,
            )
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
        await _activate_default_path(
            db,
            enrollment=existing,
            actor_id=actor.user_id,
            default_path=default_path,
        )
        result.append(await _enrollment_out(db, existing))
    return result


async def _activate_default_path(
    db: AsyncSession,
    *,
    enrollment: ProgramEnrollment,
    actor_id: UUID,
    default_path: dict[str, object] | None,
) -> bool:
    """Start the pinned version's default path, when it has one.

    A missing default means the enrollment belongs to a legacy published
    version.  It intentionally remains ``awaiting_path`` so existing student
    choice flows continue to work without a data backfill.
    """
    if default_path is None:
        return False

    attempt = ProgramPathAttempt(
        program_enrollment_id=enrollment.id,
        career_path_id=cast(UUID, default_path["career_path_id"]),
        career_path_version_id=cast(UUID, default_path["career_path_version_id"]),
        status="active",
        selection_source="program_default",
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(attempt)
    enrollment.status = "active"
    enrollment.updated_by = actor_id
    await flush_or_conflict(db)
    await career_paths_api.ensure_program_path_access(
        db,
        student_id=enrollment.student_id,
        career_path_id=attempt.career_path_id,
        version_id=attempt.career_path_version_id,
        actor_id=actor_id,
    )
    return True


async def _get_enrollable_default_path(
    db: AsyncSession, version_id: UUID
) -> dict[str, object] | None:
    """Resolve and validate the default before any enrollment rows are written."""
    paths = await queries.list_version_paths(db, version_id)
    default_path = next((row for row in paths if bool(row["is_default"])), None)
    if default_path is not None and default_path["status"] == "archived":
        raise ConflictError("program_default_path_is_archived")
    return default_path


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
    attempts = await queries.list_active_attempts(db, enrollment.id, lock=True)
    for attempt in attempts:
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
    selected_attempts = [row for row in attempts if row.status in ("active", "completed")]
    attempt_outputs: list[PathAttemptRead] = []
    completed_courses = 0
    total_courses = 0
    for attempt in attempts:
        progress_rows: list[dict[str, object]] = []
        if attempt.status in ("active", "completed"):
            progress_rows = await career_paths_api.get_version_course_progress_for_user(
                db,
                version_id=attempt.career_path_version_id,
                student_id=enrollment.student_id,
            )
        attempt_completed = sum(bool(row.get("satisfied")) for row in progress_rows)
        attempt_total = len(progress_rows)
        completed_courses += attempt_completed
        total_courses += attempt_total
        attempt_outputs.append(
            PathAttemptRead.model_validate(
                {
                    **attempt.__dict__,
                    "progress_percent": round(
                        (attempt_completed / attempt_total * 100) if attempt_total else 0,
                        2,
                    ),
                    "completed_courses": attempt_completed,
                    "total_courses": attempt_total,
                }
            )
        )
    # Same expiry hazard as _program_out: a flush after mutating this row
    # (e.g. select_path flipping status) expires server-side columns such as
    # ``completed_at`` / ``withdrawn_at``, which then vanish from ``__dict__``
    # and make ProgramEnrollmentRead reject the payload. Refresh re-loads
    # every column before the dict spread.
    await db.refresh(enrollment)
    return ProgramEnrollmentRead.model_validate(
        {
            **enrollment.__dict__,
            "program_name": program.name,
            "program_version_no": version.version_no,
            "max_path_switches": version.max_path_switches,
            "approved_switch_count": await queries.count_approved_switches(db, enrollment.id),
            "max_career_paths": version.max_career_paths_per_enrollment,
            "selected_path_count": len(selected_attempts),
            "current_progress_percent": round(
                (completed_courses / total_courses * 100) if total_courses else 0, 2
            ),
            "current_completed_courses": completed_courses,
            "current_total_courses": total_courses,
            "paths": await _paths_for_version(db, version.id),
            "attempts": attempt_outputs,
            "pending_change_request": (
                PathChangeRequestRead.model_validate(pending).model_dump(mode="json")
                if pending is not None
                else None
            ),
            "change_request_history": [
                PathChangeRequestRead.model_validate(row)
                for row in await queries.list_enrollment_change_requests(db, enrollment.id)
            ],
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
    if enrollment.status not in ("awaiting_path", "active"):
        raise ConflictError("paths_can_only_be_added_to_an_open_program")
    program = await queries.get_program(db, enrollment.learning_program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    version = await queries.get_version(db, enrollment.program_version_id)
    if version is None:
        raise NotFoundError("learning_program_version_not_found")
    selected = [
        attempt
        for attempt in await queries.list_attempts(db, enrollment.id)
        if attempt.status in ("active", "completed")
    ]
    if any(attempt.career_path_id == career_path_id for attempt in selected):
        raise ConflictError("path_already_selected")
    limit = version.max_career_paths_per_enrollment
    if len(selected) >= limit:
        raise ProgramConflictError(
            "career_path_selection_limit_reached",
            f"This learning program allows at most {limit} selected career path"
            f"{'s' if limit != 1 else ''}.",
            limit=limit,
        )
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
        selection_source="student",
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
    from_attempt_id: UUID | None = None,
    arq_pool: object | None = None,
) -> PathChangeRequestRead:
    enrollment = await queries.get_enrollment(db, enrollment_id, lock=True)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("program_enrollment_not_found")
    if enrollment.status != "active":
        raise ConflictError("only_active_programs_can_change_path")
    active_attempts = await queries.list_active_attempts(db, enrollment.id, lock=True)
    if not active_attempts:
        raise ConflictError("active_path_attempt_not_found")
    if from_attempt_id is None:
        if len(active_attempts) != 1:
            raise ConflictError("source_path_attempt_is_required")
        attempt = active_attempts[0]
    else:
        attempt = next((row for row in active_attempts if row.id == from_attempt_id), None)
        if attempt is None:
            raise ConflictError("source_path_attempt_is_not_active")
    selected_attempts = [
        row
        for row in await queries.list_attempts(db, enrollment.id)
        if row.status in ("active", "completed")
    ]
    if any(row.career_path_id == target_path_id for row in selected_attempts):
        raise ConflictError("path_already_selected")
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

    # Every owning Faculty Dean of the program should learn about the new
    # request (see _notify_owning_deans).
    program = await queries.get_program(db, version.learning_program_id)
    if program is not None:
        await _notify_owning_deans(
            db,
            program=program,
            request_id=request.id,
            student_id=student_id,
            subject_path_id=target_path_id,
            arq_pool=arq_pool,
        )

    return PathChangeRequestRead.model_validate(request)


async def _subject_path_name(db: AsyncSession, request: PathChangeRequest) -> str:
    """The path a notification about this request should name.

    A change is about where the student is going; a drop is about the path
    they are ending, which is the only path it mentions at all.
    """
    if request.kind == "drop":
        attempt = await queries.get_attempt(db, request.from_attempt_id)
        if attempt is None:
            return "the requested path"
        return await _target_path_name(db, attempt.career_path_id)
    return await _target_path_name(db, cast(UUID, request.target_career_path_id))


async def request_path_drop(
    db: AsyncSession,
    *,
    enrollment_id: UUID,
    from_attempt_id: UUID,
    reason: str,
    student_id: UUID,
    arq_pool: object | None = None,
) -> PathChangeRequestRead:
    """File a request to end one Career Path without taking another.

    Same review queue, same one-open-request slot and same switch budget as a
    change: an approved drop consumes a switch and is never refunded, so a
    student cannot use drop-then-add to dodge the dean.

    The rule this adds is that a student must always be sitting at least one
    active path. Dropping the last one is leaving the program, which is
    withdrawal — an operator action with its own record — not a path edit.
    """
    enrollment = await queries.get_enrollment(db, enrollment_id, lock=True)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("program_enrollment_not_found")
    if enrollment.status != "active":
        raise ConflictError("only_active_programs_can_drop_path")
    active_attempts = await queries.list_active_attempts(db, enrollment.id, lock=True)
    if not active_attempts:
        raise ConflictError("active_path_attempt_not_found")
    if len(active_attempts) < 2:
        raise ProgramConflictError(
            "at_least_one_path_must_remain",
            "This is your only active career path, so it cannot be dropped. "
            "Ask your program manager to withdraw you from the program instead.",
            active_path_count=len(active_attempts),
        )
    attempt = next((row for row in active_attempts if row.id == from_attempt_id), None)
    if attempt is None:
        raise ConflictError("source_path_attempt_is_not_active")
    if await queries.get_pending_request(db, enrollment.id) is not None:
        raise ConflictError("program_already_has_a_pending_path_change")
    version = await queries.get_version(db, enrollment.program_version_id)
    if version is None:
        raise NotFoundError("program_version_not_found")
    if await queries.count_approved_switches(db, enrollment.id) >= version.max_path_switches:
        raise ConflictError("path_switch_limit_reached")

    request = PathChangeRequest(
        program_enrollment_id=enrollment.id,
        from_attempt_id=attempt.id,
        kind="drop",
        target_career_path_id=None,
        target_career_path_version_id=None,
        reason=reason,
        status="pending",
        created_by=student_id,
        updated_by=student_id,
    )
    db.add(request)
    await flush_or_conflict(db)

    program = await queries.get_program(db, version.learning_program_id)
    if program is not None:
        await _notify_owning_deans(
            db,
            program=program,
            request_id=request.id,
            student_id=student_id,
            subject_path_id=attempt.career_path_id,
            kind="drop",
            arq_pool=arq_pool,
        )

    return PathChangeRequestRead.model_validate(request)


async def cancel_change_request(
    db: AsyncSession, *, request_id: UUID, student_id: UUID
) -> PathChangeRequestRead:
    """Student withdraws their own request.

    Allowed while the request is OPEN — including ``in_progress``. A student who
    changed their mind should not be forced to wait for a decision just because
    a dean opened the request, and a rejection costs no switch budget, so there
    is nothing to game by cancelling late. The cancellation stays in the
    request history either way.
    """
    request = await queries.get_change_request(db, request_id, lock=True)
    if request is None:
        raise NotFoundError("path_change_request_not_found")
    enrollment = await queries.get_enrollment(db, request.program_enrollment_id)
    if enrollment is None or enrollment.student_id != student_id:
        raise NotFoundError("path_change_request_not_found")
    if request.status not in PATH_CHANGE_OPEN_STATUSES:
        raise ConflictError("only_open_requests_can_be_cancelled")
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
    decision_reason_code: str | None = None,
    decision_note: str | None = None,
    arq_pool: object | None = None,
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
    # Both OPEN statuses are decidable: a dean can approve/reject straight from
    # the queue, or acknowledge first and decide after checking the data. Only
    # terminal states are refused.
    if request.status not in PATH_CHANGE_OPEN_STATUSES:
        raise ConflictError("request_is_not_open")
    target_path_name = await _subject_path_name(db, request)
    if not approve:
        if decision_reason_code is None:
            raise ConflictError("rejection_reason_code_is_required")
        if decision_reason_code not in PATH_CHANGE_REJECTION_REASON_CODES:
            raise ConflictError("unknown_rejection_reason_code")
        # 'other' is the escape hatch from the fixed list, so it has to carry
        # the words the list could not express — otherwise the student receives
        # a rejection whose reason is literally "other".
        custom_reason = (decision_reason or "").strip() or None
        note = (decision_note or "").strip() or None
        if decision_reason_code == "other" and custom_reason is None:
            raise ConflictError("rejection_reason_is_required_when_code_is_other")
        if decision_reason_code != "other" and note is None:
            note = custom_reason
        if decision_reason_code != "other":
            custom_reason = None
        request.status = "rejected"
        request.reviewed_by = actor.user_id
        request.reviewed_at = _now()
        request.decision_reason_code = decision_reason_code
        request.decision_reason = custom_reason
        request.decision_note = note
        request.updated_by = actor.user_id
        await flush_or_conflict(db)
        await notify.notify_path_change_rejected(
            db,
            student_user_id=enrollment.student_id,
            request_id=request.id,
            kind=request.kind,
            program_name=program.name,
            target_path_name=target_path_name,
            reason_code=decision_reason_code,
            reason_detail=custom_reason,
            note=note,
            arq_pool=arq_pool,
        )
        return PathChangeRequestRead.model_validate(request)
    if enrollment.status != "active":
        raise ConflictError("program_is_not_active")
    attempt = await queries.get_attempt(db, request.from_attempt_id, lock=True)
    if (
        attempt is None
        or attempt.program_enrollment_id != enrollment.id
        or attempt.status != "active"
    ):
        raise ConflictError("active_path_changed_since_request")
    if request.kind == "drop":
        return await _approve_path_drop(
            db,
            request=request,
            enrollment=enrollment,
            attempt=attempt,
            program_name=program.name,
            dropped_path_name=target_path_name,
            decision_reason=decision_reason,
            decision_note=decision_note,
            actor_id=actor.user_id,
            arq_pool=arq_pool,
        )
    selected_attempts = [
        row
        for row in await queries.list_attempts(db, enrollment.id)
        if row.status in ("active", "completed")
    ]
    if any(
        row.id != attempt.id and row.career_path_id == request.target_career_path_id
        for row in selected_attempts
    ):
        raise ConflictError("path_already_selected")
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
        selection_source="path_change",
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
    request.decision_note = (decision_note or "").strip() or None
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
    await notify.notify_path_change_approved(
        db,
        student_user_id=enrollment.student_id,
        request_id=request.id,
        kind="change",
        program_name=program.name,
        target_path_name=target_path_name,
        note=request.decision_note,
        arq_pool=arq_pool,
    )
    return PathChangeRequestRead.model_validate(request)


async def _approve_path_drop(
    db: AsyncSession,
    *,
    request: PathChangeRequest,
    enrollment: ProgramEnrollment,
    attempt: ProgramPathAttempt,
    program_name: str,
    dropped_path_name: str,
    decision_reason: str | None,
    decision_note: str | None,
    actor_id: UUID,
    arq_pool: object | None,
) -> PathChangeRequestRead:
    """End one attempt, grant nothing in its place.

    The "at least one active path" rule is re-checked HERE, not only when the
    request was filed. Between filing and decision the student's other path
    can have been switched out, completed, or dropped by an earlier approval,
    and approving blindly would leave an active enrolment with no active path
    — a state nothing else in the feature can produce or recover from.

    No ``new_attempt_id`` is written: the request is terminal on its own, and
    a NULL there is what distinguishes an approved drop from an approved
    change in the history the student reads.
    """
    version = await queries.get_version(db, enrollment.program_version_id)
    if version is None:
        raise NotFoundError("program_version_not_found")
    if await queries.count_approved_switches(db, enrollment.id) >= version.max_path_switches:
        raise ConflictError("path_switch_limit_reached")

    remaining = [
        row
        for row in await queries.list_active_attempts(db, enrollment.id, lock=True)
        if row.id != attempt.id
    ]
    if not remaining:
        raise ProgramConflictError(
            "at_least_one_path_must_remain",
            f"Approving this would leave {await _student_label(db, enrollment.student_id)} "
            "with no active career path. Withdraw them from the program instead.",
            active_path_count=1,
        )

    attempt.exit_snapshot = await queries.build_exit_snapshot(
        db, student_id=enrollment.student_id, attempt=attempt
    )
    attempt.status = "cancelled"
    attempt.ended_at = _now()
    attempt.updated_by = actor_id
    await queries.revoke_path_entitlements(db, attempt_id=attempt.id)
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
            actor_id=actor_id,
        )

    request.status = "approved"
    request.reviewed_by = actor_id
    request.reviewed_at = _now()
    request.decision_reason = decision_reason
    request.decision_note = (decision_note or "").strip() or None
    request.updated_by = actor_id
    enrollment.updated_by = actor_id
    await flush_or_conflict(db)
    await notify.notify_path_change_approved(
        db,
        student_user_id=enrollment.student_id,
        request_id=request.id,
        kind="drop",
        program_name=program_name,
        target_path_name=dropped_path_name,
        note=request.decision_note,
        arq_pool=arq_pool,
    )
    return PathChangeRequestRead.model_validate(request)


async def mark_change_request_in_progress(
    db: AsyncSession,
    *,
    request_id: UUID,
    actor: CurrentUser,
    arq_pool: object | None = None,
) -> PathChangeRequestRead:
    """Acknowledge a request: the dean has seen it and is checking the data.

    Deliberately NOT a decision. Nothing about the student's enrolment, attempt,
    or switch budget moves, and every approval-time recheck still runs later —
    this only records that the request has an owner and lets the student stop
    wondering whether it was received.

    Idempotent by design: re-acknowledging an already ``in_progress`` request is
    a no-op rather than a 409, because two deans opening the same queue is
    normal and the second one should not see an error. The notification is
    therefore sent once, on the pending → in_progress edge only.
    """
    request = await queries.get_change_request(db, request_id, lock=True)
    if request is None:
        raise NotFoundError("path_change_request_not_found")
    enrollment = await queries.get_enrollment(db, request.program_enrollment_id)
    if enrollment is None:
        raise NotFoundError("program_enrollment_not_found")
    program = await queries.get_program(db, enrollment.learning_program_id)
    if program is None:
        raise NotFoundError("learning_program_not_found")
    await _require_owner_dean(db, actor_id=actor.user_id, program=program)
    if actor.user_id == enrollment.student_id:
        raise ForbiddenError("self_approval_is_not_allowed")
    if request.status == "in_progress":
        return PathChangeRequestRead.model_validate(request)
    if request.status != "pending":
        raise ConflictError("only_pending_requests_can_be_marked_in_progress")

    request.status = "in_progress"
    request.in_progress_at = _now()
    request.in_progress_by = actor.user_id
    request.updated_by = actor.user_id
    await flush_or_conflict(db)
    await notify.notify_path_change_in_progress(
        db,
        student_user_id=enrollment.student_id,
        request_id=request.id,
        kind=request.kind,
        program_name=program.name,
        target_path_name=await _subject_path_name(db, request),
        arq_pool=arq_pool,
    )
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
    "mark_change_request_in_progress",
    "publish_program",
    "request_path_change",
    "request_path_drop",
    "select_path",
    "update_program",
    "withdraw_student",
]
