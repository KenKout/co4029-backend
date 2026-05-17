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

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from abridgeai.core.config import get_settings
from abridgeai.core.observability import get_logger
from abridgeai.features.admin.queries import ai_costs as ai_costs_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

VALID_PERIODS = ("day", "week", "month")


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
            "usd": _to_float(row.get("total_usd")),
            "call_count": _to_int(row.get("call_count")),
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
    period: str,
) -> dict[str, Any]:
    row = await ai_costs_queries.summary(db, since=since, period=period)
    return _normalise_summary(row)


async def by_user(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
    is_today_window: bool,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_user(db, since=since, top_n=top_n)
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


async def by_pipeline(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = await ai_costs_queries.by_pipeline(db, since=since, top_n=top_n)
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
            "usd": _to_float(r.get("usd")),
            "latency_ms": r.get("latency_ms"),
            "created_at": r.get("called_at"),
            "pipeline_run_id": r.get("pipeline_run_id"),
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


__all__ = [
    "VALID_PERIODS",
    "by_pipeline",
    "by_user",
    "default_since",
    "is_today_window",
    "recent",
    "summary",
]
