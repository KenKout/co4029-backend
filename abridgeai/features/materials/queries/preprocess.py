"""Read/write queries for the preprocessing quarantine.

Raw SQL to match the sibling query modules in this feature. Every statement
is scoped by ``material_version_id`` or ``course_id``; the router resolves
those from a permission-checked material, so a quarantine id alone can never
address another course's rows.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def list_quarantine_for_version(
    db: AsyncSession,
    material_version_id: UUID,
    *,
    include_confirmed: bool = True,
) -> list[dict[str, Any]]:
    """Every recorded preprocessing decision for one version, in page order."""
    clause = (
        ""
        if include_confirmed
        else "AND (teacher_action IS NULL OR teacher_action <> 'confirm')"
    )
    rows = (
        await db.execute(
            text(
                f"""
                SELECT id, unit_kind, page_number, ordinal, content, occurrences,
                       rule_name, reason_code, action, rule_score, detector_stage,
                       teacher_action, teacher_action_at, created_at
                FROM material_preprocess_quarantine
                WHERE material_version_id = :vid {clause}
                ORDER BY page_number NULLS LAST, ordinal
                """  # noqa: S608 -- clause is a fixed literal, not user input
            ),
            {"vid": str(material_version_id)},
        )
    ).mappings()
    return [dict(row) for row in rows]


async def get_quarantine_row(
    db: AsyncSession,
    quarantine_id: UUID,
) -> dict[str, Any] | None:
    """Fetch one row plus the version/course it belongs to (for permissions)."""
    row = (
        await db.execute(
            text(
                """
                SELECT q.id, q.material_version_id, q.course_id, q.content,
                       q.reason_code, q.teacher_action,
                       lmv.material_id AS material_id
                FROM material_preprocess_quarantine q
                JOIN learning_material_versions lmv ON lmv.id = q.material_version_id
                WHERE q.id = :qid
                  AND lmv.deleted_at IS NULL
                """
            ),
            {"qid": str(quarantine_id)},
        )
    ).mappings().first()
    return dict(row) if row else None


async def set_teacher_action(
    db: AsyncSession,
    quarantine_id: UUID,
    *,
    action: str,
    user_id: UUID,
) -> bool:
    """Record ``restore`` or ``confirm``. Returns False when the row is gone.

    The decision is persisted rather than applied immediately: the cascade
    reads it on the NEXT reprocess. Applying it in place would mean rewriting
    already-embedded chunks, which is the re-index this whole design exists to
    avoid.
    """
    result = await db.execute(
        text(
            """
            UPDATE material_preprocess_quarantine
            SET teacher_action = :action,
                teacher_action_by = CAST(:uid AS uuid),
                teacher_action_at = now(),
                updated_at = now()
            WHERE id = :qid
            """
        ),
        {"qid": str(quarantine_id), "action": action, "uid": str(user_id)},
    )
    return bool(result.rowcount)


async def list_restored_pages(
    db: AsyncSession,
    material_version_id: UUID,
) -> set[int]:
    """Pages the teacher restored, so the next run leaves them alone.

    Keyed on ``page_number``, not ``ordinal``: ordinals are positions in the
    decision log and shift the moment any earlier rule fires differently,
    whereas a page number is what the teacher actually saw and acted on.

    Page-level only. A line-level restore (an individual stripped footer) is
    recorded and visible in the report but is not yet re-injected — the units
    teachers actually dispute are whole pages.
    """
    rows = (
        await db.execute(
            text(
                """
                SELECT DISTINCT page_number
                FROM material_preprocess_quarantine
                WHERE material_version_id = :vid
                  AND teacher_action = 'restore'
                  AND unit_kind = 'page'
                  AND page_number IS NOT NULL
                """
            ),
            {"vid": str(material_version_id)},
        )
    ).all()
    return {int(row.page_number) for row in rows}


async def get_current_version_and_mode(
    db: AsyncSession,
    material_id: UUID,
) -> tuple[UUID, str, dict[str, Any]] | None:
    """``(current_version_id, preprocess_mode, extracted_metadata)`` or None.

    One round trip for everything the report view needs from Postgres.
    """
    row = (
        await db.execute(
            text(
                """
                SELECT lm.current_version_id AS version_id,
                       lm.preprocess_mode    AS preprocess_mode,
                       lmv.extracted_metadata AS extracted_metadata
                FROM learning_materials lm
                JOIN learning_material_versions lmv ON lmv.id = lm.current_version_id
                WHERE lm.id = :mid
                  AND lm.deleted_at IS NULL
                  AND lmv.deleted_at IS NULL
                """
            ),
            {"mid": str(material_id)},
        )
    ).mappings().first()
    if row is None or row["version_id"] is None:
        return None
    extracted = row["extracted_metadata"]
    return (
        UUID(str(row["version_id"])),
        str(row["preprocess_mode"] or "full"),
        dict(extracted) if isinstance(extracted, dict) else {},
    )


async def update_preprocess_mode(
    db: AsyncSession,
    material_id: UUID,
    *,
    mode: str,
) -> bool:
    """Set ``learning_materials.preprocess_mode``. False when no such row."""
    result = await db.execute(
        text(
            """
            UPDATE learning_materials
            SET preprocess_mode = :mode, updated_at = now()
            WHERE id = :mid
            """
        ),
        {"mid": str(material_id), "mode": mode},
    )
    return bool(result.rowcount)


async def course_filter_summary(
    db: AsyncSession,
    course_id: UUID,
) -> list[dict[str, Any]]:
    """Per-reason counts across a course — the filter-precision audit view."""
    rows = (
        await db.execute(
            text(
                """
                SELECT reason_code,
                       count(*) AS unit_count,
                       sum(occurrences) AS occurrence_count,
                       count(*) FILTER (WHERE teacher_action = 'restore') AS restored,
                       count(*) FILTER (WHERE teacher_action = 'confirm') AS confirmed
                FROM material_preprocess_quarantine
                WHERE course_id = :cid
                GROUP BY reason_code
                ORDER BY occurrence_count DESC
                """
            ),
            {"cid": str(course_id)},
        )
    ).mappings()
    return [dict(row) for row in rows]


__all__ = [
    "course_filter_summary",
    "get_quarantine_row",
    "list_quarantine_for_version",
    "list_restored_pages",
    "set_teacher_action",
]
