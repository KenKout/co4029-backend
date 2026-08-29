"""Tenant operations queries (PRD ADM-042)."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause

_SQL_DIR = resources.files("abridgeai.features.admin.queries.sql")


def _load(name: str) -> TextClause:
    return text(_SQL_DIR.joinpath(name).read_text(encoding="utf-8"))


_OPERATIONS_SUMMARY_SQL = _load("tenants/operations_summary.sql")


async def operations_summary(
    db: AsyncSession,
    *,
    organization_id: UUID,
    now: datetime,
    window_days: int,
) -> dict[str, Any]:
    """People, inventory, storage, jobs and config overrides for one tenant."""
    row = (
        await db.execute(
            _OPERATIONS_SUMMARY_SQL,
            {
                "organization_id": organization_id,
                "now": now,
                "window_days": window_days,
            },
        )
    ).mappings().one()
    return dict(row)


__all__ = ["operations_summary"]
