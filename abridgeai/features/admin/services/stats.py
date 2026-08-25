"""Stats service -- composes :mod:`features.admin.queries.stats`.

Org-scope rule (Reconciliation §A1): when the caller's permission set contains
``system.administer`` they bypass org filtering (global view); otherwise they
must hold ``system.stats.read`` AND :func:`resolve_caller_organization` returns
the actor's org. The actor must have at least one non-deleted org membership;
otherwise an empty result is returned (no leak across tenants).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.ttl_cache import TTLCache
from abridgeai.features.admin.queries import stats as stats_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# The operator dashboard is a single ~30-scan SQL statement over
# ``ai_model_calls`` / ``processing_jobs`` (percentiles, top-driver GROUP BY,
# multi-window counts) — the heaviest recurring read in the system. It feeds
# an admin dashboard whose numbers are freshness-tolerant by definition, so a
# 60s process-local cache turns every dashboard load after the first into a
# dict hit. Keyed by org scope; ``now`` is pinned inside the cached value's
# TTL window (windows are relative to the evaluation instant, and 60s of
# drift on a "7d" window is noise).
_DASHBOARD_TTL_SECONDS = 60.0
_DASHBOARD_CACHE = TTLCache(max_entries=64, ttl_seconds=_DASHBOARD_TTL_SECONDS)


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
    """Single-response operator metrics rollup for the admin dashboard.

    Cached for 60s per org scope (see ``_DASHBOARD_CACHE`` above). Tests that
    pin ``now`` bypass the cache entirely — determinism beats reuse there.
    """
    if now is not None:
        return await stats_queries.operator_dashboard(
            db, organization_id=organization_id, now=now
        )
    cache_key = organization_id
    cached = _DASHBOARD_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    result = await stats_queries.operator_dashboard(
        db,
        organization_id=organization_id,
        now=now or datetime.now(tz=UTC),
    )
    # json round-trip freezes the copy (dates/Decimals -> plain types) so a
    # caller mutating the returned dict can't poison the cache.
    _DASHBOARD_CACHE.put(cache_key, json.loads(json.dumps(result, default=str)))
    return result


__all__ = [
    "active_users",
    "content_breakdown",
    "health",
    "operator_dashboard",
    "overview",
]
