"""Authoring read-side: list / get / update / processing-progress.

Composed by ``GET`` endpoints in
:mod:`features.materials.routers.authoring`. Mutating writes
(``init_upload``, ``complete_upload``, ``reprocess_material``,
``soft_delete_material``) live in :mod:`._upload` and :mod:`._versions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.security import CurrentUser
from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion
from abridgeai.features.materials.queries import (
    get_latest_processing_job,
    get_material_for_authoring,
    list_all_materials,
)
from abridgeai.features.materials.schemas import (
    MaterialAuthoring,
    MaterialUpdate,
    ProcessingProgress,
)
from abridgeai.features.materials.services.authoring._common import (
    present_version,
    require_material,
)

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
