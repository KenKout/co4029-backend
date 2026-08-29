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

# Allowlist mapping a caller-facing dimension name -> the real, trusted column
# expression substituted into by_category.sql. The column is NOT bindable
# (Postgres cannot bind an identifier), so this dict is the ONLY sanctioned
# source of column text — never interpolate raw caller input.
_CATEGORY_DIMENSIONS: dict[str, str] = {
    "operation": "amc.operation",
    "role": "amc.role",
    "tier": "amc.tier",
    "stage_name": "amc.stage_name",
    "model_name": "amc.model_name",
    "status": "amc.status",
}

VALID_CATEGORY_DIMENSIONS = frozenset(_CATEGORY_DIMENSIONS)


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


def _load_raw(name: str) -> str:
    return _SQL_DIR.joinpath(name).read_text(encoding="utf-8")


_SUMMARY_SQL = _load("ai_costs/summary.sql")
_BY_USER_SQL = _load("ai_costs/by_user.sql")
_BY_PIPELINE_SQL = _load("ai_costs/by_pipeline.sql")
_RECENT_SQL = _load("ai_costs/recent.sql")
_BY_MODEL_SQL = _load("ai_costs/by_model.sql")
_BY_ORGANIZATION_SQL = _load("ai_costs/by_organization.sql")
_BY_CATEGORY_TEMPLATE = _load_raw("ai_costs/by_category.sql")


def _filter_binds(
    *,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> dict[str, str | None]:
    """Return the four optional NULL-safe filter binds shared by the
    summary and by_category queries. A ``None`` value disables that filter
    (``:f_x IS NULL OR col = :f_x`` short-circuits to TRUE)."""
    return {
        "f_model": model,
        "f_role": role,
        "f_operation": operation,
        "f_status": status,
    }


async def summary(
    db: AsyncSession,
    *,
    since: datetime,
    period: str,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    if period not in _VALID_PERIODS:
        raise ValueError(f"invalid period {period!r}; expected one of {sorted(_VALID_PERIODS)}")
    params: dict[str, Any] = {"since": since, "period": period}
    params.update(_filter_binds(model=model, role=role, operation=operation, status=status))
    row = (await db.execute(_SUMMARY_SQL, params)).mappings().one()
    return dict(row)


async def by_user(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = (await db.execute(_BY_USER_SQL, {"since": since, "top_n": top_n})).mappings()
    return [dict(r) for r in rows]


async def by_organization(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Spend per tenant, plus an ``organization_id IS NULL`` bucket.

    That NULL row is not an error case -- it is the spend from calls with no
    derivable tenant (session-runtime calls attribute via stage name alone).
    The caller keeps it so the per-organization rows still sum to the platform
    total.
    """
    rows = (
        await db.execute(
            _BY_ORGANIZATION_SQL,
            {"since": since, "until": until, "limit": limit},
        )
    ).mappings()
    return [dict(r) for r in rows]


async def by_pipeline(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
) -> list[dict[str, Any]]:
    rows = (await db.execute(_BY_PIPELINE_SQL, {"since": since, "top_n": top_n})).mappings()
    return [dict(r) for r in rows]


async def by_category(
    db: AsyncSession,
    *,
    dimension: str,
    since: datetime,
    top_n: int,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Group spend by one caller-chosen dimension (operation/role/tier/etc).

    ``dimension`` MUST be a key of :data:`_CATEGORY_DIMENSIONS`; the mapped
    column expression is substituted into the SQL template. The value is never
    taken from raw caller input — only from the trusted allowlist — so this is
    injection-safe despite the string substitution.
    """
    column = _CATEGORY_DIMENSIONS.get(dimension)
    if column is None:
        raise ValueError(
            f"invalid dimension {dimension!r}; expected one of {sorted(_CATEGORY_DIMENSIONS)}"
        )
    sql = text(_BY_CATEGORY_TEMPLATE.format(dimension_col=column))
    params: dict[str, Any] = {"since": since, "top_n": top_n}
    params.update(_filter_binds(model=model, role=role, operation=operation, status=status))
    rows = (await db.execute(sql, params)).mappings()
    return [dict(r) for r in rows]


async def by_model(
    db: AsyncSession,
    *,
    since: datetime,
    top_n: int,
    model: str | None = None,
    role: str | None = None,
    operation: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Per-model efficiency: spend, tokens, latency p50/p95, blended $/1k."""
    params: dict[str, Any] = {"since": since, "top_n": top_n}
    params.update(_filter_binds(model=model, role=role, operation=operation, status=status))
    rows = (await db.execute(_BY_MODEL_SQL, params)).mappings()
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
    "VALID_CATEGORY_DIMENSIONS",
    "by_category",
    "by_model",
    "by_organization",
    "by_pipeline",
    "by_user",
    "daily_user_spend",
    "recent",
    "summary",
]
