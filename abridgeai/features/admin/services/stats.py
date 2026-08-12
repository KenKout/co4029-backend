"""Stats service -- composes :mod:`features.admin.queries.stats`.

Org-scope rule (Reconciliation §A1): when the caller's permission set contains
``system.administer`` they bypass org filtering (global view); otherwise they
must hold ``system.stats.read`` AND :func:`resolve_caller_organization` returns
the actor's org. The actor must have at least one non-deleted org membership;
otherwise an empty result is returned (no leak across tenants).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.admin.queries import stats as stats_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def overview(db: AsyncSession, *, organization_id: UUID | None) -> dict[str, int]:
    return await stats_queries.overview_counts(db, organization_id=organization_id)


async def active_users(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    now: datetime | None = None,
) -> dict[str, int]:
    return await stats_queries.active_users(
        db,
        organization_id=organization_id,
        now=now or datetime.now(tz=UTC),
    )


async def active_users_trend(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    days: int,
    now: datetime | None = None,
) -> list[tuple[date, int]]:
    return await stats_queries.active_users_trend(
        db,
        organization_id=organization_id,
        days=days,
        now=now or datetime.now(tz=UTC),
    )


async def content_breakdown(
    db: AsyncSession, *, organization_id: UUID | None
) -> dict[str, list[dict[str, Any]]]:
    return await stats_queries.content_breakdown(db, organization_id=organization_id)


async def health(db: AsyncSession, *, since: datetime) -> dict[str, int]:
    return await stats_queries.health_snapshot(db, since=since)


async def operator_dashboard(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Single-response operator metrics rollup for the admin dashboard."""
    return await stats_queries.operator_dashboard(
        db,
        organization_id=organization_id,
        now=now or datetime.now(tz=UTC),
    )


__all__ = [
    "active_users",
    "content_breakdown",
    "health",
    "operator_dashboard",
    "overview",
]
