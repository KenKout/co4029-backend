"""Authoring material queries (teacher / admin surface).

Plan §4761-4764. Returns ORM models in any state — drafts, processing,
failed, ready, cancelled. The visibility predicate of ``published.py``
does NOT apply here.

``include_archived`` toggles the ``status='archived'`` filter on
``LearningMaterial``. Soft-delete is auto-filtered by T0.7's
``with_loader_criteria`` listener.

``get_material_with_versions`` returns the material plus its full version
history (newest first) as a tuple. The ORM declares no
``LearningMaterial.versions`` collection (T4.1 kept relationships
minimal), so we issue two queries and bundle the result. Cheap — O(1)
round-trips, no lazy-load risk in async context.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.models import (
    LearningMaterial,
    LearningMaterialVersion,
)


async def list_all_materials(
    db: AsyncSession,
    lesson_id: UUID,
    *,
    include_archived: bool = False,
) -> list[LearningMaterial]:
    """All materials on ``lesson_id`` (any visibility / processing state).

    Mirrors the legacy ``backend/app/queries/orm/materials.py:list_lesson_materials``
    but adds an ``include_archived`` flag (default FALSE). Archive is
    expressed via the current version's ``processing_status='cancelled'``
    pipeline — for the materials feature an archived material is one whose
    current version was cancelled. When ``include_archived=False`` we
    filter rows whose joined current-version status is ``cancelled``.

    With no current version yet (fresh upload mid-pipeline) the row is
    still returned — the teacher needs to see in-flight uploads.
    """
    stmt = select(LearningMaterial).where(LearningMaterial.lesson_id == lesson_id)
    if not include_archived:
        cancelled_subq = (
            select(LearningMaterialVersion.id)
            .where(
                LearningMaterialVersion.id == LearningMaterial.current_version_id,
                LearningMaterialVersion.processing_status == "cancelled",
            )
            .exists()
        )
        stmt = stmt.where(~cancelled_subq)
    stmt = stmt.order_by(LearningMaterial.created_at)
    return list((await db.execute(stmt)).scalars().all())


async def get_material_for_authoring(
    db: AsyncSession, material_id: UUID
) -> LearningMaterial | None:
    """Material by id without visibility / processing filters.

    Returns soft-deleted-IS-still-filtered (T0.7) but everything else
    surfaces — drafts, failed pipelines, cancelled versions, hidden.
    """
    return await db.get(LearningMaterial, material_id)


async def get_material_with_versions(
    db: AsyncSession, material_id: UUID
) -> tuple[LearningMaterial, list[LearningMaterialVersion]] | None:
    """Material + every non-deleted version, newest first.

    Returns ``None`` when the material is missing or soft-deleted.
    Two-query bundle (no ORM ``versions`` relationship declared on
    :class:`LearningMaterial` per T4.1) — equivalent to a ``selectinload``
    in cost, no lazy-load risk in async context.
    """
    material = await db.get(LearningMaterial, material_id)
    if material is None:
        return None
    versions_stmt = (
        select(LearningMaterialVersion)
        .where(LearningMaterialVersion.material_id == material_id)
        .order_by(LearningMaterialVersion.version_no.desc())
    )
    versions = list((await db.execute(versions_stmt)).scalars().all())
    return material, versions


__all__ = [
    "get_material_for_authoring",
    "get_material_with_versions",
    "list_all_materials",
]
