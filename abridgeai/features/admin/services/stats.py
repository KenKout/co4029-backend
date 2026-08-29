"""Stats service -- composes :mod:`features.admin.queries.stats`.

Org-scope rule (Reconciliation §A1): when the caller's permission set contains
``system.administer`` they bypass org filtering (global view); otherwise they
must hold ``system.stats.read`` AND :func:`resolve_caller_organization` returns
the actor's org. The actor must have at least one non-deleted org membership;
otherwise an empty result is returned (no leak across tenants).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.ttl_cache import TTLCache
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.admin.queries import stats as stats_queries
from abridgeai.features.admin.services import job_metrics

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
# Default dashboard window. Callers may widen or narrow it; every windowed
# metric in the response moves together so the tiles stay comparable.
DEFAULT_WINDOW_DAYS = 7

#: Inactivity threshold for the tenant-anomaly count. Fixed at 30 days rather
#: than following the dashboard window: a tenant being quiet for a 1-day window
#: is a weekend, not an anomaly, and letting the window drive it would make the
#: tile mean something different at every setting.
INACTIVE_ORG_DAYS = 30

_DASHBOARD_TTL_SECONDS = 60.0
_DASHBOARD_CACHE = TTLCache(max_entries=64, ttl_seconds=_DASHBOARD_TTL_SECONDS)


def _rate_pct(numerator: int, denominator: int) -> float | None:
    """Percentage, or ``None`` when the denominator is empty.

    PRD section 5: "0 of 0" must never render as 0% -- a quiet window and a
    clean window are different states, and a tile that conflates them stops
    being worth reading.
    """
    if denominator <= 0:
        return None
    return round(100.0 * numerator / denominator, 2)


def _opt_int(value: Any) -> int | None:
    """Round a nullable numeric (percentile) to int, preserving ``None``."""
    return None if value is None else int(round(float(value)))


def _window_bounds(
    now: datetime,
    *,
    window_days: int,
    window_from: date | None = None,
    window_to: date | None = None,
) -> tuple[datetime, datetime, datetime]:
    """Resolve the rollup window to explicit datetime bounds.

    Two shapes, both inclusive of full days:

    * days-mode: ``window_days`` given -> ``[now - days, now)``, the leading
      edge of the window ending at the evaluation instant.
    * range-mode: ``window_from`` / ``window_to`` given -> ``[from 00:00,
      to + 1d 00:00)``, so a picker range like Aug 1 - Aug 29 covers all of
      Aug 29 and the labels the UI prints match the rows counted
      (the label == the calculation rule).

    Returns ``(window_start, window_end, previous_start)`` where the previous
    window has the same length and sits immediately before the current one.
    """
    if window_from is not None and window_to is not None:
        start = datetime.combine(window_from, time.min, tzinfo=UTC)
        end = datetime.combine(window_to + timedelta(days=1), time.min, tzinfo=UTC)
        length = end - start
        previous_start = start - length
        return start, end, previous_start
    start = now - timedelta(days=window_days)
    return start, now, start - timedelta(days=window_days)


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


async def api_latency_trend(
    db: AsyncSession,
    *,
    days: int,
    now: datetime | None = None,
) -> list[tuple[date, int, int | None, int | None]]:
    raw = await stats_queries.api_latency_trend(
        db,
        days=days,
        now=now or datetime.now(tz=UTC),
    )
    return [(day, requests, _opt_int(p50), _opt_int(p95)) for day, requests, p50, p95 in raw]


async def operator_dashboard(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    window_from: date | None = None,
    window_to: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Single-response operator metrics rollup for the admin dashboard.

    Composes three sources that each own their own contract:

    * ``stats_queries.operator_dashboard`` -- cost, usage, tenant signals.
    * ``job_metrics`` -- the canonical job failure rate and queue reading,
      shared with the processing surface so the two can never disagree
      (PRD ADM-004).
    * ``stats_queries.api_reliability`` -- request error rate and latency.

    Every rate in the result is ``None`` rather than ``0.0`` when its
    denominator was empty, and ``as_of`` / ``window_days`` / the per-family
    ``*_scope`` keys travel with the numbers so the client never has to guess
    what a tile is measuring (PRD ADM-004, section 5).

    The window is either ``window_days`` (last N days ending now) or the
    explicit ``window_from`` / ``window_to`` date range; both shapes resolve
    to the SAME datetime bounds for every component so the tiles can never
    describe different spans. In range mode the response also carries
    ``window_from`` / ``window_to`` so the UI labels match the rows counted.

    Cached for 60s per (org scope, window). Tests that pin ``now`` bypass the
    cache entirely -- determinism beats reuse there.
    """
    if now is not None:
        return await _operator_dashboard_uncached(
            db,
            organization_id=organization_id,
            window_days=window_days,
            window_from=window_from,
            window_to=window_to,
            now=now,
        )
    cache_key = (organization_id, window_days, window_from, window_to)
    cached = _DASHBOARD_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    result = await _operator_dashboard_uncached(
        db,
        organization_id=organization_id,
        window_days=window_days,
        window_from=window_from,
        window_to=window_to,
        now=datetime.now(tz=UTC),
    )
    # json round-trip freezes the copy (dates/Decimals -> plain types) so a
    # caller mutating the returned dict can't poison the cache.
    _DASHBOARD_CACHE.put(cache_key, json.loads(json.dumps(result, default=str)))
    return result


async def _operator_dashboard_uncached(
    db: AsyncSession,
    *,
    organization_id: UUID | None,
    window_days: int,
    window_from: date | None,
    window_to: date | None,
    now: datetime,
) -> dict[str, Any]:
    window_start, window_end, previous_start = _window_bounds(
        now,
        window_days=window_days,
        window_from=window_from,
        window_to=window_to,
    )
    rollup = await stats_queries.operator_dashboard(
        db,
        organization_id=organization_id,
        as_of=now,
        window_start=window_start,
        window_end=window_end,
        previous_start=previous_start,
    )
    jobs = await job_metrics.job_outcomes(
        db,
        window_days=(window_end - window_start).days,
        now=now,
        current_start=window_start,
        current_end=window_end,
        previous_start=previous_start,
    )
    queue = await job_metrics.queue_state(db, now=now)
    api = await stats_queries.api_reliability(
        db, as_of=now, window_start=window_start, window_end=window_end
    )
    # Same rows the Organizations list filters to, so the count and its
    # destination cannot drift (ADM-045).
    inactive_orgs = await access_control_api.list_inactive_organizations(
        db, days=INACTIVE_ORG_DAYS, now=now, organization_id=organization_id
    )

    requests_total = int(api["requests_total"] or 0)
    requests_5xx = int(api["requests_5xx"] or 0)
    requests_4xx = int(api["requests_4xx"] or 0)
    ai_calls = int(rollup["ai_calls_window"] or 0)
    failed_ai_calls = int(rollup["failed_ai_calls_window"] or 0)

    return {
        # -- envelope: what these numbers measure -----------------------
        "as_of": rollup["as_of"],
        # range-mode windows report the exact dates the picker applied; the
        # UI echoes them so the label it prints == the rows counted.
        "window_from": window_from,
        "window_to": window_to,
        "window_days": window_days,
        "organization_id": organization_id,
        # Which families the organization filter actually reached. Job, cost
        # and API metrics have no organization edge in the schema, so an
        # org-scoped caller sees global numbers for them and is told so
        # rather than shown a tenant-looking figure.
        "usage_scope": "organization" if organization_id else "global",
        "tenant_scope": "organization" if organization_id else "global",
        "job_scope": jobs.scope,
        "cost_scope": "global",
        "api_scope": "global",
        # -- reliability & throughput -----------------------------------
        "job_failure_rate_pct": jobs.failure_rate_pct,
        "job_failure_rate_prev_pct": jobs.prev_failure_rate_pct,
        "jobs_terminal_window": jobs.terminal_total,
        "jobs_failed_window": jobs.terminal_failed,
        "jobs_terminal_prev_window": jobs.prev_terminal_total,
        "jobs_failed_prev_window": jobs.prev_terminal_failed,
        "queue_depth": queue.queue_depth,
        "queue_pending": queue.pending_count,
        "queue_running": queue.running_count,
        "queue_oldest_age_seconds": queue.oldest_age_seconds,
        "requests_window": requests_total,
        "requests_5xx_window": requests_5xx,
        "requests_4xx_window": requests_4xx,
        "api_error_rate_pct": _rate_pct(requests_5xx, requests_total),
        "api_client_error_rate_pct": _rate_pct(requests_4xx, requests_total),
        "api_p50_latency_ms": _opt_int(api["p50_latency_ms"]),
        "api_p95_latency_ms": _opt_int(api["p95_latency_ms"]),
        # -- cost & capacity --------------------------------------------
        "spend_window_usd": rollup["spend_window_usd"],
        "spend_prev_window_usd": rollup["spend_prev_window_usd"],
        "projected_month_end_usd": rollup["projected_month_end_usd"],
        "tokens_window": rollup["tokens_window"],
        "ai_calls_window": ai_calls,
        "failed_ai_calls_window": failed_ai_calls,
        "ai_failure_rate_pct": _rate_pct(failed_ai_calls, ai_calls),
        "top_cost_driver": rollup["top_cost_driver"],
        "top_cost_driver_usd": rollup["top_cost_driver_usd"],
        "slowest_model": rollup["slowest_model"],
        "slowest_model_p95_ms": rollup["slowest_model_p95_ms"],
        # -- usage -------------------------------------------------------
        "active_users_today": rollup["active_users_today"],
        "active_users_window": rollup["active_users_window"],
        "total_users": rollup["total_users"],
        "materials_ingested_window": rollup["materials_ingested_window"],
        # -- tenant anomalies --------------------------------------------
        "orgs_total": rollup["orgs_total"],
        "orgs_inactive_30d": len(inactive_orgs),
    }


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "INACTIVE_ORG_DAYS",
    "active_users",
    "content_breakdown",
    "operator_dashboard",
    "overview",
]
