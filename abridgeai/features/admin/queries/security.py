"""Security & access queries for the operator dashboard (PRD ADM-020)."""

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


_SUMMARY_SQL = _load("security/summary.sql")


async def summary(
    db: AsyncSession,
    *,
    now: datetime,
    since: datetime,
    organization_id: UUID | None,
) -> dict[str, Any]:
    row = (
        await db.execute(
            _SUMMARY_SQL,
            {"now": now, "since": since, "organization_id": organization_id},
        )
    ).mappings().one()
    return dict(row)


__all__ = ["summary"]
