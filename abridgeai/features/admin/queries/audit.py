"""Audit dashboard queries (T7.5).

T0.23 (HTTP audit log middleware) is deferred. :func:`http_audit_table_exists`
probes the schema at request time so callers can return 503 gracefully.
"""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

_SQL_DIR = resources.files("abridgeai.features.admin.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_ROLE_CHANGES_SQL = _load("audit/role_changes.sql")
_HTTP_AUDIT_SQL = _load("audit/http_audit.sql")
_HTTP_AUDIT_PROBE_SQL = _load("audit/http_audit_table_exists.sql")

# Data-change lookups, keyed by the ``table`` query param. Every statement
# projects the same uniform shape (entity_id / title / status / created_by /
# updated_by / deleted_by / created_at / updated_at / deleted_at /
# organization_id) plus optional entity-specific columns; the router flattens
# whatever the row carries.
_DATA_CHANGES_SQL: dict[str, TextClause] = {
    "courses": _load("audit/data_changes_courses.sql"),
    "materials": _load("audit/data_changes_materials.sql"),
    "users": _load("audit/data_changes_users.sql"),
    "role_assignments": _load("audit/data_changes_role_assignments.sql"),
}

# LIST variants — same projection, but every row updated within a window
# instead of one row by id. Kept in a parallel map so the single-lookup
# statements stay untouched.
_DATA_CHANGES_LIST_SQL: dict[str, TextClause] = {
    "courses": _load("audit/data_changes_courses_list.sql"),
    "materials": _load("audit/data_changes_materials_list.sql"),
    "users": _load("audit/data_changes_users_list.sql"),
    "role_assignments": _load("audit/data_changes_role_assignments_list.sql"),
}

# Public tuple of the tables ``data_changes`` accepts — the router validates
# against this so a bad ``table`` value 400s before touching the DB.
SUPPORTED_DATA_CHANGE_TABLES: tuple[str, ...] = tuple(_DATA_CHANGES_SQL)


async def role_changes(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None,
    organization_id: UUID | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _ROLE_CHANGES_SQL,
            {
                "since": since,
                "until": until,
                "organization_id": organization_id,
                "limit": limit,
            },
        )
    ).mappings()
    return [dict(r) for r in rows]


async def http_audit_table_exists(db: AsyncSession) -> bool:
    row = (await db.execute(_HTTP_AUDIT_PROBE_SQL)).first()
    return row is not None


async def http_audit_search(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None,
    user_id: UUID | None,
    path_pattern: str | None,
    event_kind: str | None = None,
    request_id: UUID | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            _HTTP_AUDIT_SQL,
            {
                "since": since,
                "until": until,
                "user_id": user_id,
                "path_pattern": path_pattern,
                "event_kind": event_kind,
                "request_id": request_id,
                "limit": limit,
            },
        )
    ).mappings()
    return [dict(r) for r in rows]


async def data_changes(db: AsyncSession, *, table: str, entity_id: UUID) -> dict[str, Any] | None:
    """Look up the audit trail for a single row in ``table``.

    ``table`` must be one of :data:`SUPPORTED_DATA_CHANGE_TABLES`; callers
    are expected to validate before calling (the router 400s on an
    unsupported value). Returns ``None`` when no matching row exists.
    """
    stmt = _DATA_CHANGES_SQL[table]
    row = (await db.execute(stmt, {"entity_id": entity_id})).mappings().one_or_none()
    return dict(row) if row is not None else None


async def data_changes_list(
    db: AsyncSession,
    *,
    table: str,
    since: datetime,
    until: datetime | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Every row in ``table`` updated within ``[since, until)``, newest first."""
    rows = (
        await db.execute(
            _DATA_CHANGES_LIST_SQL[table],
            {"since": since, "until": until, "limit": limit},
        )
    ).mappings()
    return [dict(r) for r in rows]


__all__ = [
    "SUPPORTED_DATA_CHANGE_TABLES",
    "data_changes",
    "data_changes_list",
    "http_audit_search",
    "http_audit_table_exists",
    "role_changes",
]
