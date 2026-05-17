"""IT Admin-side queries for the courses aggregate (T3.5 administration).

These joins span ``ai_model_calls`` + ``processing_jobs`` + ``generation_runs``
to produce a per-course processing audit. The cross-feature reach is
intentional and read-only: services consuming this module are themselves
the IT-Admin-scoped course administration layer.

The pattern mirrors :mod:`abridgeai.features.access_control.queries.admin` —
INSERT/DELETE flows use raw SQL (``text()``) because the ORM unit-of-work
walks ``Base.metadata.sorted_tables`` and triggers cross-feature FK
resolution; SELECTs use ``select()`` for type-safe ORM hydration.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.courses.models import Course
from abridgeai.features.courses.queries._pagination import (
    CursorPage,
)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _clamp(limit: int) -> int:
    return min(max(limit, 1), _MAX_LIMIT)


def _encode_admin_cursor(created_at: datetime, course_id: UUID) -> str:
    """Composite cursor for the admin list ``(created_at, id)`` order.

    Admin lists newest first so freshly-inserted rows surface on page 1
    regardless of UUID lexical order. The cursor encodes both keys so
    pagination is stable when many rows share a ``created_at`` value.
    """
    payload = json.dumps([created_at.isoformat(), str(course_id)], separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_admin_cursor(cursor: str) -> tuple[datetime, UUID]:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode()).decode()
    created_iso, course_id = json.loads(raw)
    return datetime.fromisoformat(created_iso), UUID(course_id)


async def list_all_courses_admin(
    db: AsyncSession,
    *,
    include_deleted: bool = False,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
) -> CursorPage[Course]:
    """Cursor-paginated admin view of every course (newest first).

    Ordered by ``(created_at DESC, id DESC)`` so the most recently
    created rows surface first — operationally useful (admins reach for
    the freshest course) and resilient to schemas that already carry
    hundreds of legacy seed rows ahead of newly tombstoned ones.

    When ``include_deleted=True`` the T0.7 global soft-delete filter is
    bypassed via ``execution_options(include_deleted=True)`` so admin
    sees both live and tombstoned rows.
    """
    capped = _clamp(limit)
    after = _decode_admin_cursor(cursor) if cursor else None
    stmt = select(Course).order_by(Course.created_at.desc(), Course.id.desc()).limit(capped)
    if after is not None:
        after_created_at, after_id = after
        stmt = stmt.where(
            or_(
                Course.created_at < after_created_at,
                and_(Course.created_at == after_created_at, Course.id < after_id),
            )
        )
    if include_deleted:
        stmt = stmt.execution_options(include_deleted=True)
    rows = list((await db.execute(stmt)).scalars().all())
    next_cursor = (
        _encode_admin_cursor(rows[-1].created_at, rows[-1].id) if len(rows) == capped else None
    )
    return CursorPage(items=rows, next_cursor=next_cursor)


async def get_course_including_deleted(db: AsyncSession, course_id: UUID) -> Course | None:
    """Course-by-id lookup that ignores the soft-delete loader filter.

    Required by :func:`restore_soft_deleted_course` because by definition
    the row we want to restore is already soft-deleted.
    """
    stmt = select(Course).where(Course.id == course_id).execution_options(include_deleted=True)
    return (await db.execute(stmt)).scalar_one_or_none()


_RESTORE_SOFT_DELETED_SQL = text(
    """
    UPDATE courses
       SET deleted_at = NULL,
           deleted_by = NULL
     WHERE id = :course_id
       AND deleted_at IS NOT NULL
    """
)


async def restore_soft_deleted_course(db: AsyncSession, course_id: UUID) -> bool:
    """Clear ``deleted_at`` / ``deleted_by`` on the course row.

    Returns True if a soft-deleted row was found and restored.
    Children (modules, lessons, ...) are left in their current state —
    administrators who need full-tree restore must escalate via DB ops
    (out of scope for the MVP per plan §4216).
    """
    result = await db.execute(_RESTORE_SOFT_DELETED_SQL, {"course_id": course_id})
    await db.flush()
    return (getattr(result, "rowcount", 0) or 0) > 0


_PROCESSING_AUDIT_SQL = text(
    """
    SELECT
        COALESCE(SUM(amc.estimated_cost_usd), 0) AS total_cost_usd,
        COALESCE(SUM(amc.input_tokens), 0)       AS total_input_tokens,
        COALESCE(SUM(amc.output_tokens), 0)      AS total_output_tokens,
        COUNT(amc.id)                             AS total_calls,
        COUNT(DISTINCT amc.pipeline_run_id)       AS pipeline_runs,
        COUNT(DISTINCT pj.id)                     AS processing_jobs,
        COUNT(DISTINCT gr.id)                     AS generation_runs,
        MIN(amc.called_at)                        AS first_call_at,
        MAX(amc.called_at)                        AS last_call_at
    FROM ai_model_calls amc
    LEFT JOIN processing_jobs pj   ON pj.id = amc.processing_job_id
    LEFT JOIN generation_runs gr   ON gr.id = amc.generation_run_id
    WHERE gr.course_id = :course_id
    """
)


async def get_course_processing_audit(db: AsyncSession, course_id: UUID) -> dict[str, Any]:
    """Aggregate AI processing costs / counts scoped to ``course_id``.

    Joins ``ai_model_calls`` to the two parent FKs that may carry course
    context (``processing_jobs.course_id`` or ``generation_runs.course_id``).
    Returns a single-row aggregate; values are zero / None when no AI work
    has touched the course.
    """
    result = await db.execute(_PROCESSING_AUDIT_SQL, {"course_id": course_id})
    row: dict[str, Any] = dict(result.mappings().one_or_none() or {})
    return {
        "course_id": course_id,
        "total_cost_usd": float(row.get("total_cost_usd") or 0),
        "total_input_tokens": int(row.get("total_input_tokens") or 0),
        "total_output_tokens": int(row.get("total_output_tokens") or 0),
        "total_calls": int(row.get("total_calls") or 0),
        "pipeline_runs": int(row.get("pipeline_runs") or 0),
        "processing_jobs": int(row.get("processing_jobs") or 0),
        "generation_runs": int(row.get("generation_runs") or 0),
        "first_call_at": _normalize_dt(row.get("first_call_at")),
        "last_call_at": _normalize_dt(row.get("last_call_at")),
    }


def _normalize_dt(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


__all__ = [
    "get_course_including_deleted",
    "get_course_processing_audit",
    "list_all_courses_admin",
    "restore_soft_deleted_course",
]
