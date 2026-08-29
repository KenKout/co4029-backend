"""Audit service -- composes :mod:`features.admin.queries.audit`.

T0.23 graceful-degradation: :func:`http_audit_search` raises
:class:`HttpAuditUnavailable` when the table is absent so the router can
return 503 without coupling the test harness to T0.23's deployment.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.security.pii import mask_email, mask_ip
from abridgeai.features.admin.queries import audit as audit_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class HttpAuditUnavailableError(RuntimeError):
    """Raised when the http_audit_log table (T0.23) is not deployed yet."""


# Fields carrying personal data in the aggregate projections. Masked on the way
# out of the service, so an unmasked value never enters a list response body —
# see ``core/security/pii`` for why a client-side mask is not a mask.
_MASKED_EMAIL_FIELDS = ("primary_email", "actor_email")
_MASKED_IP_FIELDS = ("ip_address",)


def _mask_row(row: dict[str, Any]) -> dict[str, Any]:
    """Redact personal fields in one aggregate row, leaving the rest intact."""
    masked = dict(row)
    for field in _MASKED_EMAIL_FIELDS:
        if field in masked:
            masked[field] = mask_email(masked[field])
    for field in _MASKED_IP_FIELDS:
        if field in masked:
            masked[field] = mask_ip(masked[field])
    return masked


async def role_changes(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    organization_id: UUID | None,
    limit: int,
) -> list[dict[str, Any]]:
    return await audit_queries.role_changes(
        db, since=since, until=until, organization_id=organization_id, limit=limit
    )


async def http_audit_search(
    db: AsyncSession,
    *,
    reveal: bool = False,
    since: datetime,
    until: datetime | None = None,
    user_id: UUID | None,
    path_pattern: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Request log search. IPs are masked unless ``reveal`` is set.

    ``reveal`` is a deliberate act, gated on ``system.administer`` at the
    router, and the reveal request is itself written to this same log — so
    unmasking is recorded rather than ambient. Masked is the default because
    this endpoint's normal use is scanning for patterns, which a /16 answers.
    """
    if not await audit_queries.http_audit_table_exists(db):
        raise HttpAuditUnavailableError("http_audit_log table (T0.23) not deployed yet")
    rows = await audit_queries.http_audit_search(
        db,
        since=since,
        until=until,
        user_id=user_id,
        path_pattern=path_pattern,
        limit=limit,
    )
    return rows if reveal else [_mask_row(row) for row in rows]


SUPPORTED_DATA_CHANGE_TABLES = audit_queries.SUPPORTED_DATA_CHANGE_TABLES


async def data_changes(db: AsyncSession, *, table: str, entity_id: UUID) -> dict[str, Any] | None:
    """Audit trail for a single row in ``table`` (FR-6.7).

    ``table`` is validated at the router edge against
    :data:`SUPPORTED_DATA_CHANGE_TABLES`; here we trust it and delegate.
    """
    return await audit_queries.data_changes(db, table=table, entity_id=entity_id)


async def data_changes_list(
    db: AsyncSession,
    *,
    table: str,
    since: datetime,
    until: datetime | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    """Every row in ``table`` changed since ``since``, newest first.

    Personal fields are masked (ADM-024). This is the scan-for-patterns view;
    a partial address is enough to spot "all of these are one tenant" or to
    match a row against an address the operator already has. The single-entity
    ``data_changes`` lookup returns the full value, because naming one entity
    is the stated purpose that entitles the caller to it.
    """
    rows = await audit_queries.data_changes_list(
        db, table=table, since=since, until=until, limit=limit
    )
    return [_mask_row(row) for row in rows]


__all__ = [
    "SUPPORTED_DATA_CHANGE_TABLES",
    "HttpAuditUnavailableError",
    "data_changes",
    "data_changes_list",
    "http_audit_search",
    "role_changes",
]
