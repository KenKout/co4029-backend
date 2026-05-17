"""Visibility-filtered material queries (learner / public surface).

Plan §4753-4760, §4773. Returns ORM models; services serialize.

Visibility predicate (Reconciliation §C10 + DRAFT_VISIBILITY draft):

* ``LearningMaterial.visible_to_students`` is TRUE
* The current version (``learning_materials.current_version_id`` →
  ``learning_material_versions.id``) is in ``processing_status='ready'``

Soft-delete is filtered automatically by the T0.7 ``with_loader_criteria``
listener — every ``select(LearningMaterial)`` / ``select(LearningMaterialVersion)``
emits ``WHERE deleted_at IS NULL`` for free.

Materials in flight (``pending`` / ``extracting`` / ``chunking`` /
``enriching`` / ``embedding`` / ``building_kg``) are filtered OUT — never
returned to learners. Same for ``failed`` and ``cancelled``.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.models import (
    LearningMaterial,
    LearningMaterialVersion,
)


async def list_visible_materials(db: AsyncSession, lesson_id: UUID) -> list[LearningMaterial]:
    """Return materials visible to students for ``lesson_id``.

    Filter: ``visible_to_students=TRUE`` AND the current version's
    ``processing_status='ready'``. Sorted by ``created_at`` (oldest first)
    — ``LearningMaterial`` does not carry a ``position`` column so created
    order is the canonical authoring order.
    """
    stmt = (
        select(LearningMaterial)
        .join(
            LearningMaterialVersion,
            LearningMaterial.current_version_id == LearningMaterialVersion.id,
        )
        .where(
            LearningMaterial.lesson_id == lesson_id,
            LearningMaterial.visible_to_students.is_(True),
            LearningMaterialVersion.processing_status == "ready",
        )
        .order_by(LearningMaterial.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_visible_material(db: AsyncSession, material_id: UUID) -> LearningMaterial | None:
    """Single-material lookup with the same visibility predicate.

    Returns ``None`` (router maps to 404) when the material is invisible,
    soft-deleted, draft, or its current version is not ``ready``. Existence
    is therefore not leaked.
    """
    stmt = (
        select(LearningMaterial)
        .join(
            LearningMaterialVersion,
            LearningMaterial.current_version_id == LearningMaterialVersion.id,
        )
        .where(
            LearningMaterial.id == material_id,
            LearningMaterial.visible_to_students.is_(True),
            LearningMaterialVersion.processing_status == "ready",
        )
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_latest_ready_version(
    db: AsyncSession, material_id: UUID
) -> LearningMaterialVersion | None:
    """The current ``ready`` version for ``material_id``, else ``None``.

    Pairs ``processing_status='ready'`` with ``is_current=TRUE`` so the
    historical-but-stale ready versions never surface. Reconciliation
    §C15: dual-source invariant (``LearningMaterial.current_version_id``
    + ``LearningMaterialVersion.is_current``) is owned by the
    upload-complete service (T4.5); this query intentionally trusts
    ``is_current`` without dereferencing the FK.
    """
    stmt = select(LearningMaterialVersion).where(
        LearningMaterialVersion.material_id == material_id,
        LearningMaterialVersion.processing_status == "ready",
        LearningMaterialVersion.is_current.is_(True),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


__all__ = [
    "get_latest_ready_version",
    "get_visible_material",
    "list_visible_materials",
]
