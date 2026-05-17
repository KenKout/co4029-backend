from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from importlib import resources
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_AT_RISK_SQL = text(
    resources.files("abridgeai.features.progress.queries.sql")
    .joinpath("at_risk_students.sql")
    .read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class AtRiskRow:
    user_id: UUID
    last_engagement_at: datetime | None
    completion_percent: Decimal
    days_since_last_engagement: float | None


async def list_at_risk_rows(db: AsyncSession, course_id: UUID) -> list[AtRiskRow]:
    rows = await db.execute(_AT_RISK_SQL, {"course_id": course_id})
    return [
        AtRiskRow(
            user_id=row.user_id,
            last_engagement_at=row.last_engagement_at,
            completion_percent=row.completion_percent,
            days_since_last_engagement=(
                float(row.days_since_last_engagement)
                if row.days_since_last_engagement is not None
                else None
            ),
        )
        for row in rows.all()
    ]


__all__ = ["AtRiskRow", "list_at_risk_rows"]
