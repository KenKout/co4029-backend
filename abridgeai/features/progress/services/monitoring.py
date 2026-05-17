from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.features.progress.queries.analytics import list_at_risk_rows
from abridgeai.features.progress.queries.authoring import (
    list_course_roster_progress,
)
from abridgeai.features.progress.schemas.authoring import (
    AtRiskListRead,
    AtRiskReason,
    AtRiskStudent,
    RosterProgressRead,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


_LOW_COMPLETION_THRESHOLD = 30
_INACTIVITY_DAYS = 7.0


async def get_roster_progress(db: AsyncSession, course_id: UUID) -> RosterProgressRead:
    students = await list_course_roster_progress(db, course_id)
    return RosterProgressRead(course_id=course_id, students=students)


async def get_at_risk_students(db: AsyncSession, course_id: UUID) -> AtRiskListRead:
    rows = await list_at_risk_rows(db, course_id)
    students: list[AtRiskStudent] = []
    for row in rows:
        reasons: list[AtRiskReason] = []
        days = row.days_since_last_engagement
        if row.last_engagement_at is None:
            reasons.append(
                AtRiskReason(
                    code="no_engagement",
                    detail="No material engagement events recorded.",
                )
            )
        elif days is not None and days >= _INACTIVITY_DAYS:
            reasons.append(
                AtRiskReason(
                    code="inactive_7d",
                    detail=(
                        f"No engagement for {int(days)} days "
                        "(threshold: 7)."
                    ),
                )
            )
        if int(row.completion_percent) < _LOW_COMPLETION_THRESHOLD:
            reasons.append(
                AtRiskReason(
                    code="low_completion",
                    detail=(
                        f"Average lesson completion {int(row.completion_percent)}% "
                        "below 30% theory/practice gap threshold."
                    ),
                )
            )
        if not reasons:
            continue
        students.append(
            AtRiskStudent(
                user_id=row.user_id,
                completion_percent=row.completion_percent,
                days_since_last_engagement=(
                    int(row.days_since_last_engagement)
                    if row.days_since_last_engagement is not None
                    else None
                ),
                reasons=reasons,
            )
        )
    return AtRiskListRead(course_id=course_id, students=students)


__all__ = ["get_at_risk_students", "get_roster_progress"]
