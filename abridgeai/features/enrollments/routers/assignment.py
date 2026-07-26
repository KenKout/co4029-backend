"""Manager + HOD enrollment-assignment surface (T7.1).

NO student-facing self-enroll route exists in this feature. Per the
locked plan decision, students cannot self-enroll and there is no
invitation-code redemption endpoint — codes are Manager tracking
artefacts handed out via email.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
    require_permission,
)
from abridgeai.features.enrollments.queries import authoring as authoring_queries
from abridgeai.features.enrollments.schemas import (
    BulkEnrollRequest,
    BulkEnrollResult,
    CSVImportResult,
    EnrollmentAuthoring,
    EnrollmentPatch,
    InvitationCodeAuthoring,
    InvitationCodeCreate,
    InvitationCodePatch,
)
from abridgeai.features.enrollments.services import manager as manager_service

dept_router = APIRouter(prefix="/dept", tags=["enrollments-assignment"])
management_router = APIRouter(prefix="/management", tags=["enrollments-assignment"])
teacher_router = APIRouter(prefix="/teacher", tags=["enrollments-assignment"])


class CSVImportPayload(BaseModel):
    csv_text: str | None = None
    csv_base64: str | None = None

    model_config = ConfigDict(extra="forbid")


_REQUIRE_ENROLLMENT_READ = require_any_permission("course.enrollment.read", "system.administer")
_REQUIRE_COURSE_ENROLLMENT_READ = require_course_permission(
    "course_id", "course.enrollment.read", "system.administer"
)
_REQUIRE_COURSE_ENROLLMENT_CREATE = require_course_permission(
    "course_id", "course.enrollment.create", "system.administer"
)
_REQUIRE_COURSE_ENROLLMENT_REMOVE = require_course_permission(
    "course_id", "course.enrollment.remove", "system.administer"
)
_REQUIRE_INVITATION_CODE_MANAGE = require_permission("course.enrollment.create")
_REQUIRE_TEACHER_ENROLLMENT_PATCH = require_any_permission(
    "course.enrollment.create",
    "course.enrollment.remove",
    "course.update",
    "system.administer",
)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "message": detail},
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "bad_request", "message": detail},
    )


@dept_router.get(
    "/courses/{course_id}/enrollments",
    response_model=list[EnrollmentAuthoring],
)
async def list_dept_course_enrollments(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[EnrollmentAuthoring]:
    return await manager_service.list_enrollments_for_course(db, course_id)


@management_router.post(
    "/courses/{course_id}/enrollments/bulk",
    response_model=BulkEnrollResult,
    status_code=status.HTTP_200_OK,
)
async def manager_bulk_enroll(
    course_id: UUID,
    payload: BulkEnrollRequest,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BulkEnrollResult:
    result = await manager_service.bulk_enroll_students(db, course_id, payload, current_user)
    await db.commit()
    return result


@management_router.delete(
    "/courses/{course_id}/enrollments/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def manager_unenroll(
    course_id: UUID,
    user_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_REMOVE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await manager_service.unenroll_student(db, course_id, user_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


def _decode_csv_text(payload: CSVImportPayload) -> str:
    if payload.csv_text is not None:
        return payload.csv_text
    if payload.csv_base64 is not None:
        try:
            return base64.b64decode(payload.csv_base64).decode("utf-8-sig")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise _bad_request(f"invalid_csv_base64: {exc.__class__.__name__}") from exc
    raise _bad_request("csv_text or csv_base64 required")


def _parse_csv_text(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


@management_router.post(
    "/courses/{course_id}/enrollments/import-csv",
    response_model=CSVImportResult,
    status_code=status.HTTP_200_OK,
)
async def manager_csv_import(
    course_id: UUID,
    payload: CSVImportPayload,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CSVImportResult:
    csv_text = _decode_csv_text(payload)
    try:
        rows = _parse_csv_text(csv_text)
    except csv.Error as exc:
        raise _bad_request(f"invalid_csv: {exc.__class__.__name__}") from exc

    result = await manager_service.bulk_import_students_from_csv(db, course_id, rows, current_user)
    await db.commit()
    return result


@management_router.get(
    "/courses/{course_id}/invitation-codes",
    response_model=list[InvitationCodeAuthoring],
)
async def list_invitation_codes(
    course_id: UUID,
    _current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_READ)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[InvitationCodeAuthoring]:
    return await manager_service.list_invitation_codes_for_course(db, course_id)


@management_router.post(
    "/courses/{course_id}/invitation-codes",
    response_model=InvitationCodeAuthoring,
    status_code=status.HTTP_201_CREATED,
)
async def create_invitation_code(
    course_id: UUID,
    payload: InvitationCodeCreate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_COURSE_ENROLLMENT_CREATE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InvitationCodeAuthoring:
    try:
        result = await manager_service.create_invitation_code(db, course_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "conflict", "message": str(exc)},
        ) from exc
    await db.commit()
    return result


async def _ensure_caller_in_code_org(
    db: AsyncSession, current_user: CurrentUser, code_id: UUID
) -> None:
    """Require the caller to belong to the invitation code's organization.

    Unlike the course-scoped endpoints above, these two resolve a code by its
    own id, so ``require_course_permission`` cannot be used — there is no
    course in the path to scope against. ``_REQUIRE_INVITATION_CODE_MANAGE``
    is a flat permission check, and the flat set ignores ``scope_kind`` (see
    ``access_control/api/public.py::_ACTIVE_PERMISSIONS_SQL``), so a manager
    granted ``course.enrollment.create`` within org B satisfies it while
    editing org A's code.

    That matters more here than the shape suggests: an invitation code is a
    self-service enrolment credential. Editing another org's code — extending
    its expiry, raising its use limit, or reactivating a revoked one — is a
    way into that org's courses, and deleting one is a denial of enrolment.

    404s rather than 403s on failure, matching the resource-not-found shape
    used elsewhere so the endpoint does not confirm the code exists.
    """
    code = await authoring_queries.get_invitation_code(db, code_id)
    if code is None:
        raise _not_found(f"Invitation code {code_id} not found")
    if await access_control_api.is_user_member_of_org(
        db, user_id=current_user.user_id, org_id=code.organization_id
    ):
        return
    permissions = await access_control_api.get_active_permissions(db, current_user.user_id)
    if any(p.code == "system.administer" for p in permissions):
        return
    raise _not_found(f"Invitation code {code_id} not found")


@management_router.patch(
    "/invitation-codes/{code_id}",
    response_model=InvitationCodeAuthoring,
)
async def update_invitation_code(
    code_id: UUID,
    payload: InvitationCodePatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_INVITATION_CODE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InvitationCodeAuthoring:
    try:
        await _ensure_caller_in_code_org(db, current_user, code_id)
        result = await manager_service.update_invitation_code(db, code_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return result


@management_router.delete(
    "/invitation-codes/{code_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_invitation_code(
    code_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_INVITATION_CODE_MANAGE)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    try:
        await _ensure_caller_in_code_org(db, current_user, code_id)
        await manager_service.delete_invitation_code(db, code_id, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()


@teacher_router.patch(
    "/course-enrollments/{enrollment_id}",
    response_model=EnrollmentAuthoring,
)
async def patch_enrollment(
    enrollment_id: UUID,
    payload: EnrollmentPatch,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_TEACHER_ENROLLMENT_PATCH)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EnrollmentAuthoring:
    """Drop / reactivate an individual enrollment from the teacher's roster.

    Powers the SPA's drop / reactivate buttons on the
    course-student-detail page. The schema allowlist (status,
    completed_at, dropped_at) keeps identity (course_id, student_id,
    source) immutable.
    """
    try:
        result = await manager_service.patch_enrollment(db, enrollment_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found(str(exc)) from exc
    await db.commit()
    return result


__all__ = ["dept_router", "management_router", "teacher_router"]
