"""Authoring read-side: list / get / update / processing-progress.

Composed by ``GET`` endpoints in
:mod:`features.materials.routers.authoring`. Mutating writes
(``init_upload``, ``complete_upload``, ``reprocess_material``,
``soft_delete_material``) live in :mod:`._upload` and :mod:`._versions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
from abridgeai.features.materials.queries import (
    get_authoring_stream_target_for_material,
    get_latest_processing_job,
    get_lesson_processing_summary,
    get_material_for_authoring,
    list_all_materials,
)
from abridgeai.features.materials.schemas import (
    LessonProcessingSummary,
    MaterialAuthoring,
    MaterialLinkExisting,
    MaterialStreamUrl,
    MaterialUpdate,
    ProcessingProgress,
)
from abridgeai.features.materials.services.authoring._common import (
    present_version,
    require_material,
)
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def list_authoring_materials(
    db: AsyncSession, lesson_id: UUID, *, include_archived: bool = False
) -> list[MaterialAuthoring]:
    materials = await list_all_materials(db, lesson_id, include_archived=include_archived)
    return [await _present_material(db, m) for m in materials]


async def get_authoring_material(db: AsyncSession, material_id: UUID) -> MaterialAuthoring | None:
    material = await get_material_for_authoring(db, material_id)
    if material is None:
        return None
    return await _present_material(db, material)


async def _present_material(db: AsyncSession, material: LearningMaterial) -> MaterialAuthoring:
    """Hydrate :class:`MaterialAuthoring` (latest version + counts)."""
    from sqlalchemy import func, select  # noqa: PLC0415  -- localised raw escape hatch

    version_count = (
        await db.execute(
            select(func.count(LearningMaterialVersion.id)).where(
                LearningMaterialVersion.material_id == material.id
            )
        )
    ).scalar_one()
    latest_version: LearningMaterialVersion | None = None
    if material.current_version_id is not None:
        latest_version = await db.get(LearningMaterialVersion, material.current_version_id)

    return MaterialAuthoring.model_validate(material).model_copy(
        update={
            "version_count": int(version_count),
            "latest_version": present_version(latest_version)
            if latest_version is not None
            else None,
        }
    )


async def get_processing_progress(db: AsyncSession, material_id: UUID) -> ProcessingProgress | None:
    """Return the latest version's progress slice (or ``None``)."""
    material = await get_material_for_authoring(db, material_id)
    if material is None or material.current_version_id is None:
        return None
    version = await db.get(LearningMaterialVersion, material.current_version_id)
    if version is None:
        return None
    job = await get_latest_processing_job(db, version.id)
    return ProcessingProgress(
        material_id=material.id,
        version_id=version.id,
        processing_status=version.processing_status,
        progress_percent=int(job.progress_percent) if job is not None else 0,
        latest_log_line=None,
        error_message=version.processing_error,
    )


async def update_material(
    db: AsyncSession,
    material_id: UUID,
    payload: MaterialUpdate,
    actor: CurrentUser,
) -> MaterialAuthoring:
    del actor  # audit listener writes updated_by
    material = await require_material(db, material_id)
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(material, key, value)
    await flush_or_conflict(db)
    await db.refresh(material)
    return await _present_material(db, material)


async def get_authoring_stream_url(db: AsyncSession, material_id: UUID) -> MaterialStreamUrl | None:
    """Mint a presigned GET URL for a teacher previewing a material.

    Authoring sibling of
    :func:`features.materials.services.catalog.get_stream_url_for_material`.
    Skips the learner ``visible_to_students`` and ``processing_status='ready'``
    gates so a teacher can review hidden / mid-pipeline materials. Returns
    ``None`` (router maps to 404) when the material is missing, soft-deleted,
    or has no current version with a resolvable storage object.
    """
    target = await get_authoring_stream_target_for_material(db, material_id)
    if target is None:
        return None
    settings = get_settings()
    safe_title = target.title.replace('"', "")
    url, _ = await create_stream_url(
        target,
        response_headers={"Content-Disposition": f'attachment; filename="{safe_title}"'},
    )
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    expires_at = datetime.now(tz=UTC) + timedelta(seconds=settings.s3_url_ttl_seconds)
    return MaterialStreamUrl(
        url=url,
        expires_at=expires_at,
        material_version_id=target.material_version_id,
    )


async def get_lesson_processing_summary_view(
    db: AsyncSession, lesson_id: UUID
) -> LessonProcessingSummary:
    """Aggregate processing-status counts across every material under a lesson.

    Wraps :func:`get_lesson_processing_summary` with a typed DTO. Returns
    a row of zeroes for a lesson with no materials so the SPA can render
    the empty-state without an extra null check.
    """
    counts = await get_lesson_processing_summary(db, lesson_id)
    return LessonProcessingSummary(lesson_id=lesson_id, **counts)


async def link_existing_material(
    db: AsyncSession,
    lesson_id: UUID,
    payload: MaterialLinkExisting,
    actor: CurrentUser,
) -> MaterialAuthoring:
    """Create a material record linked to an already-uploaded storage object.

    No upload flow, no AI processing enqueued. The material appears in
    the AI Hub as a draft that the teacher can later enable processing on.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    material_type = payload.material_type or "other"

    material = LearningMaterial(
        lesson_id=lesson_id,
        title=payload.title,
        material_type=material_type,
        ai_processing_enabled=payload.ai_processing_enabled,
        visible_to_students=payload.visible_to_students,
    )
    db.add(material)
    await flush_or_conflict(db)
    await db.refresh(material)

    version = LearningMaterialVersion(
        material_id=material.id,
        storage_object_id=payload.storage_object_id,
        version_no=1,
        is_current=True,
        processing_status="pending",
        uploaded_by=actor.user_id,
        uploaded_at=datetime.now(tz=UTC),
    )
    db.add(version)
    await flush_or_conflict(db)
    await db.refresh(version)

    material.current_version_id = version.id
    await flush_or_conflict(db)
    await db.refresh(material)

    return await _present_material(db, material)
