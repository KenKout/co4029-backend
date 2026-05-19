"""Materials authoring router (T4.5).

10 endpoints under prefix ``/teacher`` covering the direct-upload
lifecycle (single + multipart), version reprocess, soft-delete, and
authoring reads. Composes :mod:`features.materials.services.authoring`
(routers→services boundary, T0.4 import-linter contract).

Security perimeter (FIX-SEC-1, Reconciliation §A9 + §E4)
--------------------------------------------------------
Every endpoint enforces a course-scoped permission check. Endpoints
with a ``lesson_id`` path parameter use T3.7's
:func:`abridgeai.features.courses.routers._deps.require_lesson_authoring_access`,
which walks ``lesson_id → module_id → course_id``. Endpoints with a
``material_id`` path parameter use the local
:func:`require_material_authoring_access` factory, which walks
``material_id → lesson_id → module_id → course_id`` and then runs the
same owner-or-grant check semantics as
:func:`abridgeai.features.access_control.policies.require_course_permission`.

No bare ``Depends(get_current_user)`` appears on any write endpoint
(verified by the source-grep test
``test_no_bare_get_current_user_on_write_endpoints``).

Service-layer exceptions are mapped to HTTP errors locally — services
stay HTTP-agnostic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser, get_current_user
from abridgeai.features.access_control.policies import can_manage_course
from abridgeai.features.courses.api import public as courses_api
from abridgeai.features.courses.routers._deps import require_lesson_authoring_access
from abridgeai.features.materials.api import public as materials_api
from abridgeai.features.materials.models import LearningMaterialVersion
from abridgeai.features.materials.schemas import (
    MaterialAuthoring,
    MaterialUpdate,
    MaterialUploadComplete,
    MaterialUploadInit,
    ProcessingProgress,
)
from abridgeai.features.materials.schemas.public import MaterialStreamUrl
from abridgeai.features.materials.schemas.status import LessonProcessingSummary
from abridgeai.features.materials.services import authoring as authoring_service
from abridgeai.features.materials.services.authoring import (
    CompletedPartIn,
    ConcurrentReprocessError,
    HeadVerificationError,
)

router = APIRouter(prefix="/teacher", tags=["materials-authoring"])

_DEFAULT_AUTHORING_PERM: tuple[str, ...] = ("course.update",)


# ---------------------------------------------------------------------------
# Material-scoped permission wrapper (FIX-SEC-1 perimeter)
# ---------------------------------------------------------------------------


async def _resolve_material_to_course(
    db: AsyncSession, material_id: UUID
) -> tuple[UUID, UUID] | None:
    ctx = await materials_api.get_material_with_lesson_context(db, material_id)
    if ctx is None:
        return None
    course = await courses_api.get_course_by_id(db, ctx.course_id)
    if course is None:
        return None
    return ctx.course_id, course.owner_user_id


async def _resolve_version_to_course(
    db: AsyncSession, version_id: UUID
) -> tuple[UUID, UUID, UUID] | None:
    stmt = select(LearningMaterialVersion.material_id).where(
        LearningMaterialVersion.id == version_id
    )
    material_id = (await db.execute(stmt)).scalar_one_or_none()
    if material_id is None:
        return None
    resolved = await _resolve_material_to_course(db, material_id)
    if resolved is None:
        return None
    course_id, owner_user_id = resolved
    return course_id, material_id, owner_user_id


def _not_found(resource: str, resource_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "not_found", "resource": resource, "id": str(resource_id)},
    )


def _permission_denied(*, codes: tuple[str, ...], course_id: UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": "permission_denied",
            "required": list(codes),
            "scope": "course",
            "course_id": str(course_id),
        },
    )


async def _enforce_course_permission(
    db: AsyncSession,
    current_user: CurrentUser,
    course_id: UUID,
    owner_user_id: UUID,
    codes: tuple[str, ...],
) -> CurrentUser:
    if owner_user_id == current_user.user_id:
        return current_user
    for code in codes:
        if await can_manage_course(db, current_user.user_id, course_id, manage_perm=code):
            return current_user
    raise _permission_denied(codes=codes, course_id=course_id)


def require_material_authoring_access(
    *perm_codes: str,
) -> Callable[..., Awaitable[CurrentUser]]:
    """Build a dependency that walks ``material_id → course_id`` and enforces course perms.

    Closes the FIX-SEC-1 gap for material-scoped paths the same way
    :mod:`abridgeai.features.courses.routers._deps` does for lessons /
    modules / resources. The path parameter MUST be named
    ``material_id``.
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERM

    async def dependency(
        material_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await _resolve_material_to_course(db, material_id)
        if resolved is None:
            raise _not_found("material", material_id)
        course_id, owner_user_id = resolved
        return await _enforce_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


def require_version_authoring_access(
    *perm_codes: str,
) -> Callable[..., Awaitable[CurrentUser]]:
    """Walks ``version_id → material → lesson → module → course`` and enforces perms.

    The path parameter MUST be named ``version_id``. ``material_id``
    in the same path is verified to match the version's parent — a
    request that smuggles a foreign material_id is rejected with 404
    (no information leak).
    """
    codes = perm_codes or _DEFAULT_AUTHORING_PERM

    async def dependency(
        request: Request,
        version_id: UUID,
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> CurrentUser:
        resolved = await _resolve_version_to_course(db, version_id)
        if resolved is None:
            raise _not_found("material_version", version_id)
        course_id, material_id, owner_user_id = resolved
        path_material = request.path_params.get("material_id")
        if path_material is not None and str(path_material) != str(material_id):
            raise _not_found("material_version", version_id)
        return await _enforce_course_permission(db, current_user, course_id, owner_user_id, codes)

    return dependency


_REQUIRE_LESSON = require_lesson_authoring_access()
_REQUIRE_MATERIAL = require_material_authoring_access()
_REQUIRE_VERSION = require_version_authoring_access()


# ---------------------------------------------------------------------------
# ARQ pool dependency (overridable in tests)
# ---------------------------------------------------------------------------


async def get_arq_pool() -> object | None:
    """ARQ Redis pool dependency.

    Returns ``None`` until T1.10's app factory wires a real
    ``ArqRedis`` pool via ``app.dependency_overrides``. The service
    layer accepts ``None`` and skips the enqueue (useful for tests
    that exercise the DB writes without spinning up Redis +
    ``ArqRedis``); production wiring is a single-line override in the
    FastAPI factory.
    """
    return None


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class _MultipartPartOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_number: int = Field(ge=1)
    url: str


class MaterialUploadInitOut(BaseModel):
    """Discriminated upload-init response (single or multipart)."""

    model_config = ConfigDict(extra="forbid")

    material_id: UUID
    version_id: UUID
    storage_object_id: UUID
    mode: Literal["single", "multipart"]
    expires_at: str
    upload_url: str | None = None
    upload_id: str | None = None
    part_count: int | None = None
    part_size_bytes: int | None = None
    parts: list[_MultipartPartOut] | None = None


class MultipartPartsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[_MultipartPartOut]
    expires_at: str


class MultipartCompleteIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(min_length=1)
    parts: list[_CompletedPartIn]


class _CompletedPartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_number: int = Field(ge=1)
    etag: str = Field(min_length=1)


class MultipartAbortIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str = Field(min_length=1)


class UploadCompleteOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


class ReprocessOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


MultipartCompleteIn.model_rebuild()


# ---------------------------------------------------------------------------
# Endpoints — direct-upload flow
# ---------------------------------------------------------------------------


@router.post(
    "/lessons/{lesson_id}/materials/init-upload",
    response_model=MaterialUploadInitOut,
    status_code=status.HTTP_201_CREATED,
)
async def init_upload(
    lesson_id: UUID,
    payload: MaterialUploadInit,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialUploadInitOut:
    """Plan §4925-4929 — create draft material + version + presigned URL(s).

    Branches on ``size_bytes`` (>100MB → multipart, else single-shot
    PUT). The router commits after a successful service flush so the
    client's subsequent /complete call sees the version row.
    """
    init = await authoring_service.init_upload(db, lesson_id, payload, current_user)
    await db.commit()
    return MaterialUploadInitOut(
        material_id=init.material_id,
        version_id=init.version_id,
        storage_object_id=init.storage_object_id,
        mode=init.mode,
        expires_at=init.expires_at.isoformat(),
        upload_url=init.upload_url,
        upload_id=init.upload_id,
        part_count=init.part_count,
        part_size_bytes=init.part_size_bytes,
        parts=(
            [_MultipartPartOut(part_number=p.part_number, url=p.url) for p in init.parts]
            if init.parts is not None
            else None
        ),
    )


@router.post(
    "/materials/{material_id}/versions/{version_id}/multipart/parts",
    response_model=MultipartPartsOut,
)
async def fetch_multipart_parts(
    material_id: UUID,
    version_id: UUID,
    upload_id: Annotated[str, Query(min_length=1)],
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_VERSION)],
    db: Annotated[AsyncSession, Depends(get_db)],
    part_from: Annotated[int, Query(alias="from", ge=1)] = 1,
    count: Annotated[int, Query(ge=1, le=100)] = 100,
) -> MultipartPartsOut:
    """Fetch additional presigned URLs for a multipart upload."""
    del current_user
    try:
        batch = await authoring_service.fetch_multipart_parts(
            db,
            material_id,
            version_id,
            upload_id,
            part_from=part_from,
            part_count=count,
        )
    except NotFoundError as exc:
        raise _not_found("material_version", version_id) from exc
    return MultipartPartsOut(
        parts=[_MultipartPartOut(part_number=p.part_number, url=p.url) for p in batch.parts],
        expires_at=batch.expires_at.isoformat(),
    )


@router.post(
    "/materials/{material_id}/versions/{version_id}/multipart/complete",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def complete_multipart(
    material_id: UUID,
    version_id: UUID,
    payload: MultipartCompleteIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_VERSION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Finalise multipart upload (S3 ``CompleteMultipartUpload``)."""
    del current_user
    completed = [CompletedPartIn(part_number=p.part_number, etag=p.etag) for p in payload.parts]
    try:
        await authoring_service.complete_multipart(
            db, material_id, version_id, payload.upload_id, completed
        )
    except NotFoundError as exc:
        raise _not_found("material_version", version_id) from exc
    await db.commit()


@router.post(
    "/materials/{material_id}/versions/{version_id}/multipart/abort",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def abort_multipart(
    material_id: UUID,
    version_id: UUID,
    payload: MultipartAbortIn,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_VERSION)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Cancel a multipart upload + mark the version cancelled."""
    del current_user
    try:
        await authoring_service.abort_multipart(db, material_id, version_id, payload.upload_id)
    except NotFoundError as exc:
        raise _not_found("material_version", version_id) from exc
    await db.commit()


@router.post(
    "/materials/{material_id}/versions/{version_id}/complete",
    response_model=UploadCompleteOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def complete_upload(
    material_id: UUID,
    version_id: UUID,
    payload: MaterialUploadComplete,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_VERSION)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> UploadCompleteOut:
    """Plan §4933-4939 — head-verify the upload, register, enqueue ARQ.

    ``head_object`` is the security-critical step: a phantom-complete
    request that never PUTed bytes is rejected with 404, and a
    zero-byte upload is rejected with 400. Only on a successful
    verification does the ARQ ``ingest_material_version_task`` get
    enqueued.
    """
    try:
        result = await authoring_service.complete_upload(
            db, material_id, version_id, payload, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("material_version", version_id) from exc
    except HeadVerificationError as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "upload_not_found", "message": str(exc)},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "upload_invalid", "message": str(exc)},
        ) from exc
    await db.commit()
    return UploadCompleteOut(
        material_id=result.material_id,
        version_id=result.version_id,
        processing_job_id=result.processing_job_id,
        pipeline_run_id=result.pipeline_run_id,
    )


# ---------------------------------------------------------------------------
# Endpoints — reads + reprocess + delete
# ---------------------------------------------------------------------------


@router.get("/lessons/{lesson_id}/materials", response_model=list[MaterialAuthoring])
async def list_materials(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[MaterialAuthoring]:
    """List materials (any state) on a lesson — teacher view."""
    del current_user
    return await authoring_service.list_authoring_materials(db, lesson_id)


@router.get(
    "/lessons/{lesson_id}/processing-summary",
    response_model=LessonProcessingSummary,
)
async def lesson_processing_summary(
    lesson_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_LESSON)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LessonProcessingSummary:
    """Roll-up of processing-status counts across the lesson's materials.

    Lets the teacher's lesson-manage page render one badge instead of
    polling every material's individual progress endpoint.
    """
    del current_user
    return await authoring_service.get_lesson_processing_summary_view(db, lesson_id)


@router.get("/materials/{material_id}", response_model=MaterialAuthoring)
async def get_material(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialAuthoring:
    del current_user
    material = await authoring_service.get_authoring_material(db, material_id)
    if material is None:
        raise _not_found("material", material_id)
    return material


@router.get(
    "/materials/{material_id}/processing-summary",
    response_model=ProcessingProgress,
)
async def processing_summary(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProcessingProgress:
    del current_user
    progress = await authoring_service.get_processing_progress(db, material_id)
    if progress is None:
        raise _not_found("material", material_id)
    return progress


@router.get(
    "/materials/{material_id}/stream-url",
    response_model=MaterialStreamUrl,
)
async def get_material_stream_url(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialStreamUrl:
    """Mint a presigned GET URL for the teacher's preview pane.

    Authoring sibling of ``GET /materials/{material_id}/stream-url`` —
    skips the learner ``visible_to_students`` and ``processing_status='ready'``
    gates so a teacher can preview hidden / mid-pipeline materials. Auth
    same as the other ``/teacher/materials/{id}`` endpoints (course
    authoring scope).
    """
    del current_user
    stream = await authoring_service.get_authoring_stream_url(db, material_id)
    if stream is None:
        raise _not_found("material", material_id)
    return stream


@router.patch("/materials/{material_id}", response_model=MaterialAuthoring)
async def update_material(
    material_id: UUID,
    payload: MaterialUpdate,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MaterialAuthoring:
    try:
        material = await authoring_service.update_material(db, material_id, payload, current_user)
    except NotFoundError as exc:
        raise _not_found("material", material_id) from exc
    await db.commit()
    return material


@router.post(
    "/materials/{material_id}/reprocess",
    response_model=ReprocessOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_material(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
    arq_pool: Annotated[object | None, Depends(get_arq_pool)],
) -> ReprocessOut:
    """Reconciliation §C13 — clear chunks + enqueue fresh ingest.

    Returns 409 if a previous ingest job is still ``pending`` /
    ``running`` (concurrency guard).
    """
    try:
        result = await authoring_service.reprocess_material(
            db, material_id, current_user, arq_pool=arq_pool
        )
    except NotFoundError as exc:
        raise _not_found("material", material_id) from exc
    except ConcurrentReprocessError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "concurrent_reprocess", "message": str(exc)},
        ) from exc
    await db.commit()
    return ReprocessOut(
        material_id=result.material_id,
        version_id=result.version_id,
        processing_job_id=result.processing_job_id,
        pipeline_run_id=result.pipeline_run_id,
    )


@router.delete("/materials/{material_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_material(
    material_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_MATERIAL)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Soft-delete (plan §4946 + §4954) — does NOT touch S3."""
    try:
        await authoring_service.soft_delete_material(db, material_id, current_user)
    except NotFoundError as exc:
        raise _not_found("material", material_id) from exc
    await db.commit()


__all__ = [
    "get_arq_pool",
    "require_material_authoring_access",
    "require_version_authoring_access",
    "router",
]
