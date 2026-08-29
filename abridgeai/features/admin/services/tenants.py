"""Tenant operations service (PRD ADM-042).

Composes the tenant's own counts with the two figures that live elsewhere:
its AI spend (derived, with a coverage caveat) and its inactivity state (the
shared definition the dashboard counts). Assembling them here is the point of
the feature — organization detail was an identity record, and an operator's
questions about a tenant are operational.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.admin.queries import tenants as tenant_queries
from abridgeai.features.admin.services import ai_costs as ai_costs_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_WINDOW_DAYS = 30

#: Matches the dashboard's tenant-anomaly threshold so a tenant flagged
#: inactive there reads as inactive here too.
INACTIVE_ORG_DAYS = 30


@dataclass(frozen=True)
class TenantOperations:
    """One tenant's operational picture."""

    organization_id: UUID
    as_of: datetime
    window_days: int
    # people
    active_members: int
    members_active_in_window: int
    # inventory
    course_count: int
    published_course_count: int
    material_count: int
    storage_bytes: int
    # background work — org-scoped here, unlike the platform-wide job
    # aggregates, because a single tenant's entity set can be walked directly.
    jobs_terminal_window: int
    jobs_failed_window: int
    jobs_in_flight: int
    #: ``None`` when no job reached a terminal state in the window. Not 0% —
    #: the same contract every other rate in this console follows.
    job_failure_rate_pct: float | None
    # cost
    spend_window_usd: float
    #: Share of PLATFORM spend attributable to any tenant in this window. It
    #: travels with the per-tenant figure because a tenant's spend is only as
    #: meaningful as the attribution behind it; low coverage means this number
    #: understates what the tenant actually cost.
    spend_coverage_pct: float | None
    # configuration
    config_overrides: int
    # tenant state
    is_inactive: bool
    last_activity_at: datetime | None
    days_quiet: int | None


async def operations(
    db: AsyncSession,
    *,
    organization_id: UUID,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
) -> TenantOperations:
    evaluated_at = now or datetime.now(tz=UTC)
    row = await tenant_queries.operations_summary(
        db,
        organization_id=organization_id,
        now=evaluated_at,
        window_days=window_days,
    )

    costs = await ai_costs_service.by_organization(
        db,
        since=evaluated_at - timedelta(days=window_days),
        until=evaluated_at,
        limit=500,
    )
    tenant_spend = next(
        (
            item["spend_usd"]
            for item in costs["items"]
            if item["organization_id"] == organization_id
        ),
        0.0,
    )

    inactive = await access_control_api.list_inactive_organizations(
        db,
        days=INACTIVE_ORG_DAYS,
        now=evaluated_at,
        organization_id=organization_id,
    )
    quiet = inactive[0] if inactive else None

    terminal = int(row["jobs_terminal_window"] or 0)
    failed = int(row["jobs_failed_window"] or 0)

    return TenantOperations(
        organization_id=organization_id,
        as_of=row["as_of"],
        window_days=window_days,
        active_members=int(row["active_members"] or 0),
        members_active_in_window=int(row["members_active_in_window"] or 0),
        course_count=int(row["course_count"] or 0),
        published_course_count=int(row["published_course_count"] or 0),
        material_count=int(row["material_count"] or 0),
        storage_bytes=int(row["storage_bytes"] or 0),
        jobs_terminal_window=terminal,
        jobs_failed_window=failed,
        jobs_in_flight=int(row["jobs_in_flight"] or 0),
        job_failure_rate_pct=(
            None if terminal <= 0 else round(100.0 * failed / terminal, 2)
        ),
        spend_window_usd=float(tenant_spend),
        spend_coverage_pct=costs["coverage_pct"],
        config_overrides=int(row["config_overrides"] or 0),
        is_inactive=quiet is not None,
        last_activity_at=quiet.last_activity_at if quiet else None,
        days_quiet=quiet.days_quiet if quiet else None,
    )


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "INACTIVE_ORG_DAYS",
    "TenantOperations",
    "operations",
]
