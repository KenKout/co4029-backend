"""AI cost dashboard queries (T0.27).

Wraps four SQL aggregations over ``ai_model_calls`` (audit table populated by
``ai.llm.audit.write_ai_model_call``). All queries are bounded by ``:since``
to prevent unbounded scans on large audit tables.
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

_VALID_PERIODS = frozenset({"day", "week", "month"})


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_SUMMARY_SQL = _load("ai_costs/summary.sql")
_BY_USER_SQL = _load("ai_costs/by_user.sql")
_BY_PIPELINE_SQL = _load("ai_costs/by_pipeline.sql")
_RECENT_SQL = _load("ai_costs/recent.sql")


async def summary(
    db: AsyncSession,
    *,
    since: datetime,
    period: str,
) -> dict[str, Any]:
    if period not in _VALID_PERIODS:
        raise ValueError(f"invalid period {period!r}; expected one of {sorted(_VALID_PERIODS)}")
    row = (await db.execute(_SUMMARY_SQL, {"since": since, "period": period})).mappings().one()
    return dict(row)


async def by_user(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = (await db.execute(_BY_USER_SQL, {"since": since, "top_n": top_n})).mappings()
    return [dict(r) for r in rows]


async def by_pipeline(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = (await db.execute(_BY_PIPELINE_SQL, {"since": since, "top_n": top_n})).mappings()
    return [dict(r) for r in rows]


async def recent(db: AsyncSession, *, limit: int) -> list[dict[str, Any]]:
    rows = (await db.execute(_RECENT_SQL, {"limit": limit})).mappings()
    return [dict(r) for r in rows]


async def daily_user_spend(
    db: AsyncSession,
    *,
    user_id: UUID,
    day_start: datetime,
) -> float:
    """Return today-so-far USD spend for one user (used by soft-warning check).

    Uses both attribution paths (direct ``generation_run_id`` and
    ``processing_jobs.entity_id`` indirection) to match :func:`by_user`.
    """
    row = (
        (
            await db.execute(
                text(
                    """
                WITH bounded AS (
                    SELECT
                        amc.estimated_cost_usd,
                        COALESCE(gr_direct.requested_by, gr_via_job.requested_by) AS user_id
                    FROM ai_model_calls amc
                    LEFT JOIN generation_runs gr_direct
                           ON gr_direct.id = amc.generation_run_id
                    LEFT JOIN processing_jobs pj
                           ON pj.id = amc.processing_job_id
                          AND pj.entity_type = 'generation_run'
                    LEFT JOIN generation_runs gr_via_job
                           ON gr_via_job.id = pj.entity_id
                    WHERE amc.called_at >= CAST(:day_start AS timestamptz)
                )
                SELECT COALESCE(SUM(estimated_cost_usd), 0)::numeric(18, 6) AS total_usd
                FROM bounded
                WHERE user_id = CAST(:user_id AS uuid)
                """
                ),
                {"user_id": user_id, "day_start": day_start},
            )
        )
        .mappings()
        .one()
    )
    return float(row["total_usd"] or 0)


__all__ = [
    "by_pipeline",
    "by_user",
    "daily_user_spend",
    "recent",
    "summary",
]
