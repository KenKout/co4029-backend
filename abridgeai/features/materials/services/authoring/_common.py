"""Shared types, constants, and internal helpers for the authoring service.

Lives under :mod:`features.materials.services.authoring` (T4.5). The
package is split for the 800 LOC review cap (DoD §136); this module
holds the response DTOs, MIME→type fallback table, and the small
``_require_*`` / ``_resolve_*`` helpers shared by upload + version flows.

Architectural rules (mirrors the parent package docstring):

* ``services -> sqlalchemy`` import-linter contract — :class:`AsyncSession`
  is referenced only under :data:`TYPE_CHECKING`. Raw-SQL escape hatches
  used by storage helpers stay in :mod:`._storage`; reads go through
  ``queries.*``.
* No HTTP types here; routers map service exceptions themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
from abridgeai.features.materials.queries import get_material_for_authoring
from abridgeai.features.materials.schemas import MaterialVersionAuthoring
from abridgeai.features.materials.schemas.public import MaterialTypeLiteral

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Plan §4927 — single-shot threshold. Multipart for anything bigger.
MULTIPART_THRESHOLD_BYTES: int = 100 * 1024 * 1024
MULTIPART_PART_BYTES: int = 10 * 1024 * 1024  # 10 MB part size (S3 minimum is 5 MB).
MULTIPART_FIRST_BATCH_CAP: int = 100  # mint up to N URLs in init; client paginates.

# Plan §4935 — zero-byte uploads are rejected.
MIN_ACCEPTABLE_BYTES: int = 1
# Plan §4935 — accept ±1% slack vs the size declared at init. Network
# round-tripping + S3's accounting can produce a few-byte delta on the
# tail block; declare the tolerance up front so reviewers don't argue
# over magic numbers.
SIZE_TOLERANCE_FRACTION: float = 0.01

# MIME → material_type fallback when the client omits material_type at init.
_MIME_TO_TYPE: dict[str, MaterialTypeLiteral] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/plain": "text",
    "text/markdown": "text",
}
_MIME_PREFIX_TO_TYPE: tuple[tuple[str, MaterialTypeLiteral], ...] = (
    ("video/", "video"),
    ("audio/", "audio"),
    ("image/", "image"),
)


# ---------------------------------------------------------------------------
# Response DTOs (router serialises these directly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MultipartPart:
    part_number: int
    url: str


@dataclass(frozen=True)
class MaterialUploadInitResponse:
    """Discriminated response returned by ``init_upload``.

    ``mode='single'`` carries ``upload_url``; ``mode='multipart'``
    carries ``upload_id`` + the first batch of presigned part URLs.
    The router serialises this dataclass directly — Pydantic was kept
    out of the service layer to avoid a circular import dance with
    schemas (request schemas already import from public schemas).
    """

    material_id: UUID
    version_id: UUID
    storage_object_id: UUID
    mode: Literal["single", "multipart"]
    expires_at: datetime
    upload_url: str | None = None
    upload_id: str | None = None
    part_count: int | None = None
    part_size_bytes: int | None = None
    parts: tuple[_MultipartPart, ...] | None = None


@dataclass(frozen=True)
class MultipartPartsResponse:
    parts: tuple[_MultipartPart, ...]
    expires_at: datetime


@dataclass(frozen=True)
class CompletedPartIn:
    part_number: int
    etag: str


@dataclass(frozen=True)
class UploadCompleteResponse:
    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


@dataclass(frozen=True)
class ReprocessResponse:
    material_id: UUID
    version_id: UUID
    processing_job_id: UUID
    pipeline_run_id: UUID


@dataclass
class _StorageView:
    """Duck-typed S3 ``StorageObject`` Protocol fit (only bucket+object_key).

    Non-frozen so the writable Protocol on
    :class:`abridgeai.infrastructure.s3.StorageObject` accepts the type
    (matches the pattern from
    :class:`abridgeai.features.materials.queries.chunks.MaterialStreamTarget`
    and :class:`abridgeai.features.materials.workers.ingest._StorageView`).
    The service never mutates the fields.
    """

    bucket: str
    object_key: str


# ---------------------------------------------------------------------------
# Internal helpers (shared between upload + version flows)
# ---------------------------------------------------------------------------


def resolve_material_type(
    declared: MaterialTypeLiteral | None, content_type: str
) -> MaterialTypeLiteral:
    """Return ``declared`` if present, otherwise infer from MIME.

    Falls back to ``'text'`` when no rule matches; the ingest pipeline
    still runs for unknown types (it picks the appropriate extractor
    or fails loudly) so the catch-all is safer than a hard reject at
    init time.
    """
    if declared is not None:
        return declared
    lowered = content_type.lower()
    if lowered in _MIME_TO_TYPE:
        return _MIME_TO_TYPE[lowered]
    for prefix, mtype in _MIME_PREFIX_TO_TYPE:
        if lowered.startswith(prefix):
            return mtype
    return "text"


def compute_part_count(size_bytes: int) -> int:
    return max(1, math.ceil(size_bytes / MULTIPART_PART_BYTES))


def present_version(version: LearningMaterialVersion) -> MaterialVersionAuthoring:
    return MaterialVersionAuthoring.model_validate(version)


async def require_material(db: AsyncSession, material_id: UUID) -> LearningMaterial:
    material = await get_material_for_authoring(db, material_id)
    if material is None:
        raise NotFoundError(f"Material {material_id} not found")
    return material


async def require_version(db: AsyncSession, version_id: UUID) -> LearningMaterialVersion:
    # Avoid the soft-delete listener gating on a write path: the router
    # explicitly wants the row even if soft-deleted hasn't been purged
    # yet. ``db.get`` participates in the listener anyway, so we use it
    # for read-side guard only; reprocess / complete flows reach
    # additionally-fresh rows.
    version = await db.get(LearningMaterialVersion, version_id)
    if version is None:
        raise NotFoundError(f"Material version {version_id} not found")
    return version


async def resolve_storage_view(db: AsyncSession, version: LearningMaterialVersion) -> _StorageView:
    """Return ``(bucket, object_key)`` for the version's storage row.

    Raw SQL keeps us off the identity feature's ORM (the import-linter
    "Features are independent" contract) — only ``bucket`` /
    ``object_key`` are needed for the S3 helpers.
    """
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    row = (
        await db.execute(
            text(
                "SELECT bucket, object_key FROM storage_objects "
                "WHERE id = :id AND deleted_at IS NULL"
            ),
            {"id": version.storage_object_id},
        )
    ).first()
    if row is None:
        raise NotFoundError(
            f"Storage object {version.storage_object_id} for version {version.id} not found"
        )
    return _StorageView(bucket=row.bucket, object_key=row.object_key)


async def resolve_course_id_for_material(db: AsyncSession, material_id: UUID) -> UUID | None:
    """Walk ``material → lesson → module → course``.

    Used by the router to derive the ``course_id`` for the
    ``require_course_permission("course.update")`` guard on
    material-scoped endpoints (FIX-SEC-1 perimeter).
    Returns ``None`` if the chain is broken or any row is soft-deleted.
    """
    from sqlalchemy import text  # noqa: PLC0415  -- localised raw-SQL escape hatch

    row = (
        await db.execute(
            text(
                """
                SELECT m.course_id AS course_id
                FROM learning_materials lm
                JOIN lessons l ON l.id = lm.lesson_id
                JOIN modules m ON m.id = l.module_id
                JOIN courses c ON c.id = m.course_id
                WHERE lm.id = :id
                  AND lm.deleted_at IS NULL
                  AND l.deleted_at IS NULL
                  AND m.deleted_at IS NULL
                  AND c.deleted_at IS NULL
                """
            ),
            {"id": material_id},
        )
    ).first()
    return None if row is None else row.course_id


# ---------------------------------------------------------------------------
# Service-layer exceptions (HTTP-agnostic; routers map to status codes)
# ---------------------------------------------------------------------------


from abridgeai.core.exceptions import AppError  # noqa: E402  -- keep with the exceptions block


class HeadVerificationError(AppError):
    """``head_object`` rejected the upload (missing / zero-byte / size mismatch)."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ConcurrentReprocessError(AppError):
    """A previous ingest job for this version is still in flight (409)."""
