"""Audit service -- composes :mod:`features.admin.queries.audit`.

T0.23 graceful-degradation: :func:`http_audit_search` raises
:class:`HttpAuditUnavailable` when the table is absent so the router can
return 503 without coupling the test harness to T0.23's deployment.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.admin.queries import audit as audit_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class HttpAuditUnavailableError(RuntimeError):
    """Raised when the http_audit_log table (T0.23) is not deployed yet."""


async def role_changes(
    db: AsyncSession,
    *,
    since: datetime,
    organization_id: UUID | None,
    limit: int,
) -> list[dict[str, Any]]:
    return await audit_queries.role_changes(
        db, since=since, organization_id=organization_id, limit=limit
    )


async def http_audit_search(
    db: AsyncSession,
    *,
    since: datetime,
    user_id: UUID | None,
    path_pattern: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if not await audit_queries.http_audit_table_exists(db):
        raise HttpAuditUnavailableError("http_audit_log table (T0.23) not deployed yet")
    return await audit_queries.http_audit_search(
        db,
        since=since,
        user_id=user_id,
        path_pattern=path_pattern,
        limit=limit,
    )


SUPPORTED_DATA_CHANGE_TABLES = audit_queries.SUPPORTED_DATA_CHANGE_TABLES


async def data_changes(db: AsyncSession, *, table: str, entity_id: UUID) -> dict[str, Any] | None:
    """Audit trail for a single row in ``table`` (FR-6.7).

    ``table`` is validated at the router edge against
    :data:`SUPPORTED_DATA_CHANGE_TABLES`; here we trust it and delegate.
    """
    return await audit_queries.data_changes(db, table=table, entity_id=entity_id)


__all__ = [
    "SUPPORTED_DATA_CHANGE_TABLES",
    "HttpAuditUnavailableError",
    "data_changes",
    "http_audit_search",
    "role_changes",
]
