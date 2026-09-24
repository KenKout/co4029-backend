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
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import (
    require_any_permission,
    require_course_permission,
)
from abridgeai.features.enrollments.schemas import (
    BulkEnrollRequest,
    BulkEnrollResult,
    CSVImportResult,
    EnrollmentAuthoring,
    EnrollmentPatch,
)
from abridgeai.features.enrollments.services import manager as manager_service

dept_router = APIRouter(prefix="/dept", tags=["enrollments-assignment"])
management_router = APIRouter(prefix="/management", tags=["enrollments-assignment"])
teacher_router = APIRouter(prefix="/teacher", tags=["enrollments-assignment"])


async def get_arq_pool() -> object | None:
    """ARQ Redis pool dependency (email dispatch for enrolment notifications).

    Returns ``None`` until the app factory overrides it; the notification path
    accepts ``None`` and writes the in-app row without enqueuing email. Mirrors
    the identical dependency in the materials / quizzes / courses routers.
    """
    return None


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
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> BulkEnrollResult:
    result = await manager_service.bulk_enroll_students(
        db, course_id, payload, current_user, arq_pool=arq_pool
    )
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
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> CSVImportResult:
    csv_text = _decode_csv_text(payload)
    try:
        rows = _parse_csv_text(csv_text)
    except csv.Error as exc:
        raise _bad_request(f"invalid_csv: {exc.__class__.__name__}") from exc

    result = await manager_service.bulk_import_students_from_csv(
        db, course_id, rows, current_user, arq_pool=arq_pool
    )
    await db.commit()
    return result


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
