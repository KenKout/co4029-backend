"""Direct-upload init + multipart finalisation (plan §4925-4929).

Splits the upload lifecycle out of the monolithic ``authoring.py`` for
the 800 LOC review cap. Covers:

* :func:`init_upload` — single vs multipart routing, pre-mints the
  first batch of presigned URLs.
* :func:`fetch_multipart_parts` — pages further presigned part URLs.
* :func:`complete_multipart` / :func:`abort_multipart` — wraps
  :mod:`abridgeai.infrastructure.s3` calls.

The ``head_object``-verified ``complete_upload`` (which finalises the
version row + enqueues ARQ) lives in :mod:`._versions` because it
shares state with reprocess. ``get_settings`` is looked up via the
parent package so test monkeypatches at
``abridgeai.features.materials.services.authoring.get_settings``
remain effective.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import (
    flush_or_conflict,
    register_conflict_mappings,
)
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
from abridgeai.features.materials.schemas import MaterialUploadInit
from abridgeai.features.materials.services.authoring import _common
from abridgeai.features.materials.services.authoring._common import (
    CompletedPartIn,
    MaterialUploadInitResponse,
    MultipartPartsResponse,
    _MultipartPart,
    _StorageView,
    compute_part_count,
    require_version,
    resolve_material_type,
    resolve_storage_view,
)
from abridgeai.features.materials.services.authoring._storage import (
    create_storage_object,
    set_storage_object_key,
)
from abridgeai.infrastructure.s3 import (
    CompletedPart as S3CompletedPart,
)
from abridgeai.infrastructure.s3 import (
    abort_multipart_upload,
    complete_multipart_upload,
    create_multipart_upload,
    create_upload_url,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


register_conflict_mappings(
    {
        "learning_material_versions_material_id_version_no_key": "material_version_taken: another version with this number already exists for the material",  # noqa: E501
        "uq_learning_material_versions_material_version_no": "material_version_taken: another version with this number already exists for the material",  # noqa: E501
        "uq_storage_objects_bucket_key": "storage_object_key_taken: another storage object with this bucket+key already exists",  # noqa: E501
    }
)


def _get_settings() -> object:
    """Indirection so ``monkeypatch.setattr("...authoring.get_settings", ...)`` wins.

    The test suite patches ``get_settings`` on the parent package
    (``abridgeai.features.materials.services.authoring``); resolving via
    the package attribute keeps the override visible from this submodule.
    """
    from abridgeai.features.materials.services import authoring as _pkg  # noqa: PLC0415

    return _pkg.get_settings()


async def init_upload(
    db: AsyncSession,
    lesson_id: UUID,
    payload: MaterialUploadInit,
    actor: CurrentUser,
) -> MaterialUploadInitResponse:
    """Create material + version (pending) + storage row, mint upload URL(s)."""
    settings = _get_settings()
    material_type = resolve_material_type(payload.material_type, payload.content_type)

    material = LearningMaterial(
        lesson_id=lesson_id,
        title=payload.title,
        material_type=material_type,
        ai_processing_enabled=True,
        visible_to_students=False,
    )
    db.add(material)
    await flush_or_conflict(db)
    await db.refresh(material)

    storage_object_id = await create_storage_object(
        db,
        bucket=settings.s3_bucket_name,  # type: ignore[attr-defined]
        # ``materials/{material_id}/{version_id_placeholder}/{filename}`` —
        # version_id is back-filled below once the version row has an id.
        object_key=None,
        size_bytes=max(0, int(payload.size_bytes)),
        content_type=payload.content_type,
        original_filename=payload.filename,
        uploaded_by=actor.user_id,
    )

    version = LearningMaterialVersion(
        material_id=material.id,
        storage_object_id=storage_object_id,
        version_no=1,
        is_current=False,
        processing_status="pending",
        uploaded_by=actor.user_id,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(version)
    await flush_or_conflict(db)
    await db.refresh(version)

    object_key = f"materials/{material.id}/{version.id}/{payload.filename}"
    await set_storage_object_key(db, storage_object_id, object_key)

    storage_view = _StorageView(bucket=settings.s3_bucket_name, object_key=object_key)  # type: ignore[attr-defined]

    if payload.size_bytes > _common.MULTIPART_THRESHOLD_BYTES:
        part_count = compute_part_count(payload.size_bytes)
        first_batch = min(part_count, _common.MULTIPART_FIRST_BATCH_CAP)
        init = await create_multipart_upload(
            storage_view, part_count=first_batch, content_type=payload.content_type
        )
        expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)  # type: ignore[attr-defined]
        return MaterialUploadInitResponse(
            material_id=material.id,
            version_id=version.id,
            storage_object_id=storage_object_id,
            mode="multipart",
            expires_at=expires_at,
            upload_id=init.upload_id,
            part_count=part_count,
            part_size_bytes=_common.MULTIPART_PART_BYTES,
            parts=tuple(_MultipartPart(part_number=p.part_number, url=p.url) for p in init.parts),
        )

    upload_url, expires_at = await create_upload_url(
        storage_view, content_type=payload.content_type
    )
    return MaterialUploadInitResponse(
        material_id=material.id,
        version_id=version.id,
        storage_object_id=storage_object_id,
        mode="single",
        expires_at=expires_at,
        upload_url=upload_url,
    )


async def fetch_multipart_parts(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
    *,
    part_from: int,
    part_count: int,
) -> MultipartPartsResponse:
    """Mint a follow-up batch of presigned part URLs.

    Defensive validation only — caller must already have passed the
    course-perm check at the router. ``part_from`` is 1-indexed (S3
    convention); ``part_count`` is capped at
    :data:`_common.MULTIPART_FIRST_BATCH_CAP`.
    """
    if part_from < 1:
        raise AppError("part_from must be >= 1")
    if part_count < 1 or part_count > _common.MULTIPART_FIRST_BATCH_CAP:
        raise AppError(f"part_count must be in [1, {_common.MULTIPART_FIRST_BATCH_CAP}]")

    version = await require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await resolve_storage_view(db, version)

    parts = await _presign_existing_multipart(
        storage_view, upload_id=upload_id, start=part_from, count=part_count
    )
    settings = _get_settings()
    expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)  # type: ignore[attr-defined]
    return MultipartPartsResponse(
        parts=tuple(_MultipartPart(part_number=pn, url=u) for pn, u in parts),
        expires_at=expires_at,
    )


async def _presign_existing_multipart(
    storage_view: _StorageView,
    *,
    upload_id: str,
    start: int,
    count: int,
) -> list[tuple[int, str]]:
    """Presign ``count`` consecutive part URLs starting at ``start`` for ``upload_id``.

    T2.3's ``s3.py`` only exposes ``create_multipart_upload`` (which
    initialises a new upload). For the second batch we presign
    ``upload_part`` directly via aioboto3 — a one-off escape hatch
    documented for a follow-up ``s3.presign_multipart_parts`` helper.
    """
    import aioboto3  # noqa: PLC0415  -- localised at the call site

    settings = _get_settings()
    endpoint = settings.aws_public_endpoint_url or settings.aws_endpoint_url  # type: ignore[attr-defined]
    access_key = settings.aws_access_key_id  # type: ignore[attr-defined]
    secret_key = settings.aws_secret_access_key  # type: ignore[attr-defined]
    if access_key is None or secret_key is None:
        from abridgeai.infrastructure.errors import S3NotConfiguredError  # noqa: PLC0415

        raise S3NotConfiguredError("S3 credentials are not configured")
    expires_in = settings.s3_url_ttl_seconds  # type: ignore[attr-defined]

    session = aioboto3.Session()
    parts: list[tuple[int, str]] = []
    async with session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key.get_secret_value(),
        aws_secret_access_key=secret_key.get_secret_value(),
        region_name=settings.aws_region,  # type: ignore[attr-defined]
    ) as client:
        for offset in range(count):
            part_number = start + offset
            url = await client.generate_presigned_url(
                ClientMethod="upload_part",
                Params={
                    "Bucket": storage_view.bucket,
                    "Key": storage_view.object_key,
                    "UploadId": upload_id,
                    "PartNumber": part_number,
                },
                ExpiresIn=expires_in,
            )
            parts.append((part_number, url))
    return parts


async def complete_multipart(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
    completed_parts: list[CompletedPartIn],
) -> None:
    """Wrap S3 ``CompleteMultipartUpload``. Caller follows up with ``complete_upload``."""
    version = await require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await resolve_storage_view(db, version)
    s3_parts = [S3CompletedPart(part_number=p.part_number, etag=p.etag) for p in completed_parts]
    await complete_multipart_upload(storage_view, upload_id, s3_parts)


async def abort_multipart(
    db: AsyncSession,
    material_id: UUID,
    version_id: UUID,
    upload_id: str,
) -> None:
    """Cancel a multipart upload + mark the version cancelled."""
    version = await require_version(db, version_id)
    if version.material_id != material_id:
        raise NotFoundError(f"Version {version_id} does not belong to material {material_id}")
    storage_view = await resolve_storage_view(db, version)
    await abort_multipart_upload(storage_view, upload_id)
    version.processing_status = "cancelled"
    await flush_or_conflict(db)
