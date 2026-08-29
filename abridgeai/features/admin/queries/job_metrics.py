"""Canonical job-outcome and queue-state queries (PRD ADM-004).

The dashboard, the Operations module and the processing page all used to count
jobs their own way -- different populations, different time keys, different
denominators -- so the same platform reported three failure rates. These two
queries are the single definition; every surface reads them instead of rolling
its own aggregate. See the header comments in ``sql/jobs/*.sql`` for the
contract each one implements.
"""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

_SQL_DIR = resources.files("abridgeai.features.admin.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_TERMINAL_METRICS_SQL = _load("jobs/terminal_metrics.sql")
_QUEUE_STATE_SQL = _load("jobs/queue_state.sql")


async def terminal_metrics(
    db: AsyncSession, *, now: datetime, window_days: int
) -> dict[str, Any]:
    """Terminal job counts for the current window and the one before it."""
    row = (
        await db.execute(
            _TERMINAL_METRICS_SQL, {"now": now, "window_days": window_days}
        )
    ).mappings().one()
    return dict(row)


async def queue_state(db: AsyncSession, *, now: datetime) -> dict[str, Any]:
    """Point-in-time queue depth plus the age of the oldest in-flight job."""
    row = (await db.execute(_QUEUE_STATE_SQL, {"now": now})).mappings().one()
    return dict(row)


__all__ = ["queue_state", "terminal_metrics"]
