"""AI cost dashboard service (T0.27) — admin observability ONLY.

Composes :mod:`features.admin.queries.ai_costs`. Per the user decision
recorded in ``backend-restructure.md`` T0.27, this service does **not**
implement any hard rate limit / refusal — admins inspect spend; they do not
block users on cost.

A soft warning is emitted (log only, never blocks) when a user's daily spend
crosses an optional threshold configured via
``settings.ai_daily_user_spend_warn_usd``. The warning is best-effort and
emitted only on the today-window path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from abridgeai.core.config import get_settings
from abridgeai.core.observability import get_logger
from abridgeai.core.ttl_cache import TTLCache
from abridgeai.features.admin.queries import ai_costs as ai_costs_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

VALID_PERIODS = ("day", "week", "month")

# The cost dashboards full-scan ``ai_model_calls`` (aggregations, per-model
# p95, top-driver GROUP BY) on every admin page load. Admin observability is
# freshness-tolerant — a 60s-stale spend number is fine by definition — while
# the scan is the heaviest recurring read in the system once call volume
# grows. Cache key includes every query parameter that can change the result.
# Only the DEFAULT window (no explicit ``since``) is cached: an explicit
# since is a bespoke slice whose reuse pattern doesn't justify entries.
_AI_COSTS_TTL_SECONDS = 60.0
_AI_COSTS_CACHE = TTLCache(max_entries=256, ttl_seconds=_AI_COSTS_TTL_SECONDS)


def _cacheable_since(since: datetime) -> bool:
    return since == default_since(30)


def _to_float(value: Decimal | float | int | str | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_int(value: int | float | Decimal | str | None) -> int:
    if value is None:
        return 0
    return int(value)


def _normalise_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "totals": {
            "tokens": _to_int(row.get("total_tokens")),
            "input_tokens": _to_int(row.get("input_tokens")),
            "output_tokens": _to_int(row.get("output_tokens")),
            "cached_tokens": _to_int(row.get("cached_tokens")),
            "usd": _to_float(row.get("total_usd")),
            "call_count": _to_int(row.get("call_count")),
        },
        "failed": {
            "call_count": _to_int(row.get("failed_call_count")),
            "usd": _to_float(row.get("failed_usd")),
        },
        "by_role": [
            {
                "role": str(item.get("role") or "unknown"),
                "tokens": _to_int(item.get("tokens")),
                "usd": _to_float(item.get("usd")),
            }
            for item in (row.get("by_role") or [])
        ],
        "by_stage": [
            {
                "stage_name": str(item.get("stage_name") or "unknown"),
                "tokens": _to_int(item.get("tokens")),
                "usd": _to_float(item.get("usd")),
            }
            for item in (row.get("by_stage") or [])
        ],
        "buckets": [
            {
                "bucket_start_ts": item.get("bucket_start_ts"),
                "tokens": _to_int(item.get("tokens")),
                "usd": _to_float(item.get("usd")),
            }
            for item in (row.get("buckets") or [])
        ],
    }


async def summary(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    period: str,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    cacheable = _cacheable_since(since) and until is None
    cache_key: tuple[Any, ...] | None = (
        ("summary", since, period, model, role, operation, status) if cacheable else None
    )
    if cache_key is not None:
        cached = _AI_COSTS_CACHE.get(cache_key)
        if cached is not None:
            return dict(cached)
    row = await ai_costs_queries.summary(
        db,
        since=since,
        until=until,
        period=period,
        model=model,
        role=role,
        operation=operation,
        status=status,
    )
    normalised = _normalise_summary(row)
    if cache_key is not None:
        # Store a deep-frozen copy so later callers mutating the returned
        # dict can't poison the cache. json round-trip also converts any
        # Decimal/datetime that survived normalisation into plain JSON types,
        # which is exactly what the response model wants back.
        _AI_COSTS_CACHE.put(cache_key, json.loads(json.dumps(normalised, default=str)))
    return normalised


async def by_organization(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Per-tenant spend with an explicit coverage figure (PRD ADM-040).

    ``coverage_pct`` is the share of window spend this breakdown can actually
    attribute to a tenant. It ships alongside the rows rather than being left
    for the reader to work out, because a per-organization cost table that
    explains 40% of the bill and does not say so is worse than no table: it
    invites chargeback decisions the data does not support.

    ``None`` coverage means there was no spend at all in the window — not 0%,
    which would read as "nothing could be attributed".
    """
    rows = await ai_costs_queries.by_organization(
        db,
        since=since,
        until=until or datetime.now(tz=UTC),
        limit=limit,
    )
    items = [
        {
            "organization_id": r["organization_id"],
            "organization_name": str(r.get("organization_name") or ""),
            "call_count": _to_int(r.get("call_count")),
            "failed_count": _to_int(r.get("failed_count")),
            "tokens": _to_int(r.get("tokens")),
            "spend_usd": _to_float(r.get("spend_usd")),
        }
        for r in rows
    ]
    total = sum(item["spend_usd"] for item in items)
    attributed = sum(
        item["spend_usd"] for item in items if item["organization_id"] is not None
    )
    return {
        "items": items,
        "total_spend_usd": total,
        "attributed_spend_usd": attributed,
        "coverage_pct": (
            None if total <= 0 else round(100.0 * attributed / total, 2)
        ),
    }


async def by_user(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    top_n: int,
    is_today_window: bool,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_user(db, since=since, until=until, top_n=top_n)
    result = [
        {
            "user_id": r["user_id"],
            "display_name": str(r.get("display_name") or ""),
            "call_count": _to_int(r.get("call_count")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "total_usd": _to_float(r.get("total_usd")),
        }
        for r in rows
    ]
    if is_today_window:
        await _maybe_warn_threshold_exceeded(result)
    return result


async def by_category(
    db: AsyncSession,
    *,
    dimension: str,
    since: datetime,
    until: datetime | None = None,
    top_n: int,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_category(
        db,
        dimension=dimension,
        since=since,
        until=until,
        top_n=top_n,
        model=model,
        role=role,
        operation=operation,
        status=status,
    )
    return [
        {
            "dimension_value": str(r.get("dimension_value") or "unknown"),
            "call_count": _to_int(r.get("call_count")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "input_tokens": _to_int(r.get("input_tokens")),
            "output_tokens": _to_int(r.get("output_tokens")),
            "cached_tokens": _to_int(r.get("cached_tokens")),
            "total_usd": _to_float(r.get("total_usd")),
        }
        for r in rows
    ]


async def by_pipeline(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_pipeline(
        db, since=since, until=until, top_n=top_n
    )
    return [
        {
            "pipeline_run_id": r["pipeline_run_id"],
            "generation_type": r.get("generation_type"),
            "course_id": r.get("course_id"),
            "started_at": r.get("started_at"),
            "call_count": _to_int(r.get("call_count")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "total_usd": _to_float(r.get("total_usd")),
            "stages_breakdown": [
                {
                    "stage_name": str(item.get("stage_name") or "unknown"),
                    "tokens": _to_int(item.get("tokens")),
                    "usd": _to_float(item.get("usd")),
                }
                for item in (r.get("stages_breakdown") or [])
            ],
        }
        for r in rows
    ]


async def recent(db: AsyncSession, *, limit: int) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.recent(db, limit=limit)
    return [
        {
            "id": r["id"],
            "role": r.get("role"),
            "tier": r.get("tier"),
            "stage_name": r.get("stage_name"),
            "model": r.get("model"),
            "tokens": _to_int(r.get("tokens")),
            "input_tokens": _to_int(r.get("input_tokens")),
            "output_tokens": _to_int(r.get("output_tokens")),
            "cached_tokens": _to_int(r.get("cached_tokens")),
            "usd": _to_float(r.get("usd")),
            "latency_ms": r.get("latency_ms"),
            "status": r.get("status"),
            "created_at": r.get("called_at"),
            "pipeline_run_id": r.get("pipeline_run_id"),
        }
        for r in rows
    ]


async def by_model(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    top_n: int,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_model(
        db,
        since=since,
        until=until,
        top_n=top_n,
        model=model,
        role=role,
        operation=operation,
        status=status,
    )
    return [
        {
            "model_name": str(r.get("model_name") or "unknown"),
            "call_count": _to_int(r.get("call_count")),
            "total_tokens": _to_int(r.get("total_tokens")),
            "total_usd": _to_float(r.get("total_usd")),
            "latency_p50_ms": _to_int(r.get("latency_p50_ms")),
            "latency_p95_ms": _to_int(r.get("latency_p95_ms")),
            "usd_per_1m_tokens": _to_float(r.get("usd_per_1m_tokens")),
        }
        for r in rows
    ]


async def _maybe_warn_threshold_exceeded(rows: list[dict[str, Any]]) -> None:
    threshold = _read_threshold()
    if threshold is None:
        return
    for r in rows:
        spend = r.get("total_usd", 0.0)
        if spend > threshold:
            uid = r.get("user_id")
            _logger.warning(
                "ai.cost.user_threshold_exceeded",
                user_id=str(uid) if uid is not None else None,
                daily_spend_usd=spend,
                threshold_usd=threshold,
            )


def _read_threshold() -> float | None:
    settings = get_settings()
    raw = getattr(settings, "ai_daily_user_spend_warn_usd", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def is_today_window(since: datetime) -> bool:
    """Return True when ``since`` is at-or-after the start of today (UTC)."""
    now = datetime.now(tz=UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    aware = since if since.tzinfo is not None else since.replace(tzinfo=UTC)
    return aware >= today_start


def default_since(days: int = 30) -> datetime:
    """Return ``NOW() - days`` truncated to the day boundary in UTC."""
    now = datetime.now(tz=UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return today_start - timedelta(days=days)


VALID_CATEGORY_DIMENSIONS = ai_costs_queries.VALID_CATEGORY_DIMENSIONS


__all__ = [
    "VALID_CATEGORY_DIMENSIONS",
    "VALID_PERIODS",
    "by_category",
    "by_model",
    "by_pipeline",
    "by_user",
    "default_since",
    "is_today_window",
    "recent",
    "summary",
]
