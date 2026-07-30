from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.enrollments.models import Enrollment, InvitationCode
from abridgeai.features.enrollments.queries import authoring as authoring_queries
from abridgeai.features.enrollments.schemas import (
    BulkEnrollFailure,
    BulkEnrollRequest,
    BulkEnrollResult,
    CSVImportFailure,
    CSVImportResult,
    CSVImportRow,
    EnrollmentAuthoring,
    EnrollmentPatch,
    InvitationCodeAuthoring,
    InvitationCodeCreate,
    InvitationCodePatch,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser


register_conflict_mappings(
    {
        "course_enrollments_course_id_student_id_key": "enrollment_already_exists: this student is already enrolled in this course",  # noqa: E501
        "course_invitation_codes_code_key": "invitation_code_taken: an invitation code with this value already exists",  # noqa: E501
    }
)


def _to_authoring(enrollment: Enrollment) -> EnrollmentAuthoring:
    return EnrollmentAuthoring.model_validate(
        {
            "id": enrollment.id,
            "course_id": enrollment.course_id,
            "student_id": enrollment.student_id,
            "status": enrollment.status,
            "source": enrollment.source,
            "invitation_code_id": enrollment.invitation_code_id,
            "enrolled_at": enrollment.enrolled_at,
            "completed_at": enrollment.completed_at,
            "dropped_at": enrollment.dropped_at,
            "created_at": enrollment.created_at,
            "updated_at": enrollment.updated_at,
            "created_by": enrollment.created_by,
            "updated_by": enrollment.updated_by,
        }
    )


def _row_to_authoring(row: dict[str, Any]) -> EnrollmentAuthoring:
    """Same shape as :func:`_to_authoring` plus the joined identity fields."""
    return EnrollmentAuthoring.model_validate(row)


def _to_code_authoring(code: InvitationCode) -> InvitationCodeAuthoring:
    return InvitationCodeAuthoring.model_validate(
        {
            "id": code.id,
            "course_id": code.course_id,
            "organization_id": code.organization_id,
            "code": code.code,
            "expires_at": code.expires_at,
            "max_uses": code.max_uses,
            "current_uses": code.current_uses,
            "is_active": code.is_active,
            "created_at": code.created_at,
            "updated_at": code.updated_at,
            "created_by": code.created_by,
            "updated_by": code.updated_by,
            "deleted_at": code.deleted_at,
            "deleted_by": code.deleted_by,
        }
    )


async def list_enrollments_for_course(
    db: AsyncSession, course_id: UUID
) -> list[EnrollmentAuthoring]:
    """Manager/HOD roster tab -- includes ``primary_email``/``display_name``.

    Was reading the plain ``Enrollment`` ORM rows with no identity join,
    so the roster tab rendered a raw ``student_id`` UUID with no name.
    """
    rows = await authoring_queries.list_enrollments_for_course_with_identity(db, course_id)
    return [_row_to_authoring(row) for row in rows]


async def _resolve_user_ids(
    db: AsyncSession, payload: BulkEnrollRequest
) -> tuple[list[UUID], list[BulkEnrollFailure]]:
    resolved: list[UUID] = list(payload.user_ids)
    failures: list[BulkEnrollFailure] = []
    if payload.emails:
        emails = [str(email) for email in payload.emails]
        rows = await authoring_queries.lookup_users_by_email(db, emails)
        found_by_email = {str(row["primary_email"]): UUID(str(row["id"])) for row in rows}
        for email in emails:
            user_id = found_by_email.get(email)
            if user_id is None:
                failures.append(BulkEnrollFailure(identifier=email, reason="user_not_found"))
                continue
            if user_id not in resolved:
                resolved.append(user_id)
    return resolved, failures


async def _create_enrollment(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_id: UUID,
    actor_id: UUID,
    source: str,
    invitation_code_id: UUID | None = None,
) -> Enrollment:
    enrollment = Enrollment(
        course_id=course_id,
        student_id=student_id,
        status="active",
        source=source,
        invitation_code_id=invitation_code_id,
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(enrollment)
    await flush_or_conflict(db)
    return enrollment


async def _notify_enrolled_if_published(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_ids: list[UUID],
    arq_pool: object | None,
) -> None:
    """Notify newly-enrolled students, but only if the course is published.

    Enrolling into a draft is not yet actionable (the student can't open a
    course they can't see); they are notified when the course publishes. Best-
    effort: a course lookup miss or empty list is simply a no-op. Individual
    send failures are swallowed inside ``notify_student_enrolled``.
    """
    if not student_ids:
        return
    course = await courses_api.get_course_by_id(db, course_id)
    if course is None or course.status != "published":
        return
    for student_id in student_ids:
        await courses_api.notify_student_enrolled(
            db,
            student_user_id=student_id,
            course_id=course_id,
            course_title=course.title,
            course_slug=course.slug,
            arq_pool=arq_pool,
        )


async def ensure_enrollment(
    db: AsyncSession,
    *,
    course_id: UUID,
    student_id: UUID,
    actor_id: UUID,
    source: str = "manager_bulk",
) -> Enrollment:
    """Idempotently guarantee an active enrollment for ``(course, student)``.

    A "make it so" primitive used by cross-feature callers (career-path
    auto-enroll, via :func:`enrollments.api.public.ensure_course_enrollment`).
    No-op when a non-dropped row already exists; reactivates a ``dropped``
    row; otherwise creates one. Unlike :func:`bulk_enroll_students` it never
    raises ``already_enrolled`` — re-running is safe.

    ``source`` defaults to ``"manager_bulk"`` (career enrollment is a
    manager action; the ``course_enrollments.source`` CHECK has no
    dedicated ``career_path`` value and we avoid a migration per plan).
    Caller owns the transaction — no commit here.
    """
    existing = await authoring_queries.find_enrollment(db, course_id, student_id)
    if existing is not None and existing.status != "dropped":
        return existing
    if existing is not None and existing.status == "dropped":
        existing.status = "active"
        existing.dropped_at = None
        existing.updated_by = actor_id
        await flush_or_conflict(db)
        return existing
    return await _create_enrollment(
        db,
        course_id=course_id,
        student_id=student_id,
        actor_id=actor_id,
        source=source,
    )


async def bulk_enroll_students(
    db: AsyncSession,
    course_id: UUID,
    payload: BulkEnrollRequest,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> BulkEnrollResult:
    """Manager-side: enroll many students in one call.

    Already-enrolled students are reported as failures with reason
    ``already_enrolled``. Email lookups that miss are reported with
    reason ``user_not_found``. The transaction is left open for the
    router to commit.

    Newly-enrolled students are notified with a deep-link to the course when
    the course is published (best-effort; never blocks the enrolment).
    """
    candidate_ids, failures = await _resolve_user_ids(db, payload)
    enrolled: list[UUID] = []

    for user_id in candidate_ids:
        existing = await authoring_queries.find_enrollment(db, course_id, user_id)
        if existing is not None and existing.status != "dropped":
            failures.append(BulkEnrollFailure(identifier=str(user_id), reason="already_enrolled"))
            continue
        if existing is not None and existing.status == "dropped":
            existing.status = "active"
            existing.dropped_at = None
            existing.updated_by = actor.user_id
            await flush_or_conflict(db)
            enrolled.append(user_id)
            continue
        await _create_enrollment(
            db,
            course_id=course_id,
            student_id=user_id,
            actor_id=actor.user_id,
            source="manager_bulk",
        )
        enrolled.append(user_id)

    await _notify_enrolled_if_published(
        db, course_id=course_id, student_ids=enrolled, arq_pool=arq_pool
    )

    return BulkEnrollResult(enrolled=enrolled, failures=failures)


async def unenroll_student(
    db: AsyncSession,
    course_id: UUID,
    user_id: UUID,
    actor: CurrentUser,
) -> EnrollmentAuthoring:
    enrollment = await authoring_queries.find_enrollment(db, course_id, user_id)
    if enrollment is None:
        raise NotFoundError(f"No enrollment for course={course_id} user={user_id}")
    enrollment.status = "dropped"
    enrollment.dropped_at = datetime.now(UTC)
    enrollment.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(enrollment)
    return _to_authoring(enrollment)


async def patch_enrollment(
    db: AsyncSession,
    enrollment_id: UUID,
    payload: EnrollmentPatch,
    actor: CurrentUser,
) -> EnrollmentAuthoring:
    """Update a single enrollment row by id (teacher / manager surface).

    Powers the SPA's drop / reactivate buttons on the
    course-student-detail page. The schema allowlist (status,
    completed_at, dropped_at) keeps identity (course_id, student_id,
    source) immutable. ``status='dropped'`` auto-stamps ``dropped_at``
    if the caller didn't supply one; ``status='active'`` clears
    ``dropped_at`` so the row reflects re-activation.
    """
    enrollment = await db.get(Enrollment, enrollment_id)
    if enrollment is None:
        raise NotFoundError(f"Enrollment {enrollment_id} not found")
    data = payload.model_dump(exclude_unset=True)
    if (
        data.get("status") == "dropped"
        and enrollment.dropped_at is None
        and "dropped_at" not in data
    ):
        data["dropped_at"] = datetime.now(UTC)
    if data.get("status") == "active" and "dropped_at" not in data:
        enrollment.dropped_at = None
    for key, value in data.items():
        setattr(enrollment, key, value)
    enrollment.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(enrollment)
    return _to_authoring(enrollment)


async def bulk_import_students_from_csv(
    db: AsyncSession,
    course_id: UUID,
    rows: list[dict[str, Any]],
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> CSVImportResult:
    """Parse CSV-like rows, create users for new emails, enroll all.

    Each row is validated independently; bad rows do not abort the
    batch. Returns per-row error details so the Manager UI can show
    which rows failed and why.
    """
    enrolled: list[UUID] = []
    created_users: list[UUID] = []
    failures: list[CSVImportFailure] = []

    for row_number, raw in enumerate(rows, start=1):
        try:
            parsed = CSVImportRow.model_validate(raw)
        except (ValueError, TypeError) as exc:
            identifier = str(raw.get("email", "")) if isinstance(raw, dict) else None
            failures.append(
                CSVImportFailure(
                    row_number=row_number,
                    identifier=identifier,
                    reason=f"invalid_row: {exc.__class__.__name__}",
                )
            )
            continue

        email = str(parsed.email)
        existing_users = await authoring_queries.lookup_users_by_email(db, [email])
        if existing_users:
            user_id = UUID(str(existing_users[0]["id"]))
        else:
            user_id = uuid4()
            await authoring_queries.insert_user_with_profile(
                db,
                user_id=user_id,
                primary_email=email,
                given_name=parsed.given_name,
                family_name=parsed.family_name,
                display_name=parsed.display_name,
            )
            created_users.append(user_id)

        existing = await authoring_queries.find_enrollment(db, course_id, user_id)
        if existing is not None and existing.status != "dropped":
            failures.append(
                CSVImportFailure(
                    row_number=row_number,
                    identifier=email,
                    reason="already_enrolled",
                )
            )
            continue
        if existing is not None:
            existing.status = "active"
            existing.dropped_at = None
            existing.updated_by = actor.user_id
            await flush_or_conflict(db)
        else:
            await _create_enrollment(
                db,
                course_id=course_id,
                student_id=user_id,
                actor_id=actor.user_id,
                source="admin_bulk",
            )
        enrolled.append(user_id)

    await _notify_enrolled_if_published(
        db, course_id=course_id, student_ids=enrolled, arq_pool=arq_pool
    )

    return CSVImportResult(enrolled=enrolled, created_users=created_users, failures=failures)


async def list_invitation_codes_for_course(
    db: AsyncSession, course_id: UUID
) -> list[InvitationCodeAuthoring]:
    rows = await authoring_queries.list_invitation_codes_for_course(db, course_id)
    return [_to_code_authoring(row) for row in rows]


async def create_invitation_code(
    db: AsyncSession,
    course_id: UUID,
    payload: InvitationCodeCreate,
    actor: CurrentUser,
) -> InvitationCodeAuthoring:
    organization_id = await authoring_queries.get_course_organization_id(db, course_id)
    if organization_id is None:
        raise NotFoundError(f"Course {course_id} not found")

    existing = await authoring_queries.find_invitation_code_by_string(db, payload.code)
    if existing is not None:
        raise ConflictError(
            f"invitation_code_taken: invitation code {payload.code!r} already exists"
        )

    code = InvitationCode(
        course_id=course_id,
        organization_id=organization_id,
        code=payload.code,
        max_uses=payload.max_uses,
        expires_at=payload.expires_at,
        current_uses=0,
        is_active=True,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(code)
    await flush_or_conflict(db)
    return _to_code_authoring(code)


async def update_invitation_code(
    db: AsyncSession,
    code_id: UUID,
    payload: InvitationCodePatch,
    actor: CurrentUser,
) -> InvitationCodeAuthoring:
    code = await authoring_queries.get_invitation_code(db, code_id)
    if code is None:
        raise NotFoundError(f"Invitation code {code_id} not found")
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(code, key, value)
    code.updated_by = actor.user_id
    await flush_or_conflict(db)
    return _to_code_authoring(code)


async def delete_invitation_code(db: AsyncSession, code_id: UUID, actor: CurrentUser) -> None:
    code = await authoring_queries.get_invitation_code(db, code_id)
    if code is None:
        raise NotFoundError(f"Invitation code {code_id} not found")
    code.is_active = False
    code.updated_by = actor.user_id
    await soft_delete_cascade(db, code, actor_id=actor.user_id)


__all__ = [
    "bulk_enroll_students",
    "bulk_import_students_from_csv",
    "create_invitation_code",
    "delete_invitation_code",
    "list_enrollments_for_course",
    "list_invitation_codes_for_course",
    "unenroll_student",
    "update_invitation_code",
]
