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

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.materials.models import (
    LearningMaterial,
    LearningMaterialVersion,
    LessonKnowledgeGraphCurated,
)


async def get_published_curated_kg(
    db: AsyncSession, lesson_id: UUID
) -> LessonKnowledgeGraphCurated | None:
    """Return the curated KG row for a lesson IF it has a published snapshot.

    Returns ``None`` when no curated graph exists or it has never been
    published (``published_json IS NULL``) — the learner UI then hides the
    knowledge-map panel. Soft-delete is filtered by the global listener.
    """
    stmt = select(LessonKnowledgeGraphCurated).where(
        LessonKnowledgeGraphCurated.lesson_id == lesson_id,
        LessonKnowledgeGraphCurated.published_json.isnot(None),
    )
    return (await db.execute(stmt)).scalar_one_or_none()


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


# Raw cross-feature FK walks (material/lesson → course). Deliberately
# plain ``text()`` SQL: the target tables are owned by the courses feature
# and importing its models here would cross the ``Features are independent``
# contract. Same pattern as ``spaced_repetition``'s raw ``_CARDS_DUE_SQL``.

_RESOLVE_MATERIAL_COURSE_SQL = text(
    """
    SELECT c.id
    FROM learning_materials lm
    JOIN lessons l ON l.id = lm.lesson_id
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE lm.id = :material_id
      AND lm.deleted_at IS NULL
    """
)


async def resolve_material_course(db: AsyncSession, material_id: UUID) -> UUID | None:
    """Owning ``course_id`` for a material, or ``None`` (missing / deleted).

    Feeds the tenant gate on learner material reads: the caller must be
    able to see the owning course before visibility alone grants access.
    """
    return (
        await db.execute(
            _RESOLVE_MATERIAL_COURSE_SQL, {"material_id": material_id}
        )
    ).scalar_one_or_none()


_RESOLVE_LESSON_COURSE_SQL = text(
    """
    SELECT c.id
    FROM lessons l
    JOIN modules m ON m.id = l.module_id
    JOIN courses c ON c.id = m.course_id
    WHERE l.id = :lesson_id
      AND l.deleted_at IS NULL
    """
)


async def resolve_lesson_course(db: AsyncSession, lesson_id: UUID) -> UUID | None:
    """Owning ``course_id`` for a lesson, or ``None`` (missing / deleted)."""
    return (
        await db.execute(_RESOLVE_LESSON_COURSE_SQL, {"lesson_id": lesson_id})
    ).scalar_one_or_none()


__all__ = [
    "get_latest_ready_version",
    "get_visible_material",
    "list_visible_materials",
    "resolve_lesson_course",
    "resolve_material_course",
]
