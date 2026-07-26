"""Teacher-facing preprocessing report + override service.

The cascade in ``ai/preprocessing`` removes or de-prioritizes noise between
extraction and chunking, and records every decision in
``material_preprocess_quarantine``. This module is the teacher's window into
those decisions and the lever to overturn them:

* the REPORT shows what was dropped/tagged, per page, with the removed text —
  a teacher cannot sensibly override what they cannot see;
* RESTORE/CONFIRM stamps the row; the cascade reads restores on the next
  reprocess (``list_restored_pages``) and re-injects those pages;
* the per-material ``preprocess_mode`` column is the kill switch when the
  filters misread a whole document.

An override never edits chunks in place. Re-deriving the corpus is what
``reprocess_material`` is for, and the report response says so
(``requires_reprocess``) so the UI can offer the button.

All SQL lives in ``queries/preprocess.py`` per the services-don't-touch-
SQLAlchemy contract; this layer is orchestration + shaping only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.materials.queries.preprocess import (
    course_filter_summary,
    get_current_version_and_mode,
    get_quarantine_row,
    list_quarantine_for_version,
    set_teacher_action,
    update_preprocess_mode,
)
from abridgeai.features.materials.schemas.preprocess import (
    CourseFilterSummaryRow,
    PreprocessReportView,
    QuarantinedUnit,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def get_preprocess_report(
    db: AsyncSession,
    material_id: UUID,
) -> PreprocessReportView:
    """The current version's preprocessing outcome + quarantine rows."""
    resolved = await get_current_version_and_mode(db, material_id)
    if resolved is None:
        raise NotFoundError(f"material {material_id} has no current version")
    version_id, mode, extracted = resolved

    summary = extracted.get("preprocess")
    units = await list_quarantine_for_version(db, version_id)
    return PreprocessReportView(
        material_version_id=version_id,
        preprocess_mode=mode,  # type: ignore[arg-type]  # DB CHECK constrains the value set
        summary=summary if isinstance(summary, dict) else None,
        units=[QuarantinedUnit.model_validate(u) for u in units],
    )


async def apply_teacher_action(
    db: AsyncSession,
    material_id: UUID,
    quarantine_id: UUID,
    *,
    action: Literal["restore", "confirm"],
    user_id: UUID,
) -> bool:
    """Stamp a quarantine row, verifying it belongs to ``material_id``.

    The ownership check matters: the router's permission dependency guards
    the MATERIAL path parameter, so accepting a bare quarantine id here would
    let a caller act on another course's rows through their own material's
    URL. Returns True when the row was updated.
    """
    row = await get_quarantine_row(db, quarantine_id)
    if row is None or row["material_id"] != material_id:
        raise NotFoundError(f"quarantine unit {quarantine_id} not found for this material")
    return await set_teacher_action(db, quarantine_id, action=action, user_id=user_id)


async def set_preprocess_mode(
    db: AsyncSession,
    material_id: UUID,
    *,
    mode: Literal["full", "normalize_only", "off"],
) -> str:
    """Set the per-material cascade mode. Takes effect on the next reprocess."""
    updated = await update_preprocess_mode(db, material_id, mode=mode)
    if not updated:
        raise NotFoundError(f"material {material_id} not found")
    return mode


async def get_course_filter_summary(
    db: AsyncSession,
    course_id: UUID,
) -> list[CourseFilterSummaryRow]:
    """Per-reason counts across the course — the filter-precision audit."""
    rows = await course_filter_summary(db, course_id)
    return [CourseFilterSummaryRow.model_validate(r) for r in rows]


__all__ = [
    "apply_teacher_action",
    "get_course_filter_summary",
    "get_preprocess_report",
    "set_preprocess_mode",
]
