from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.runtime_settings import resolve_settings
from abridgeai.features.progress.queries.analytics import (
    AtRiskRow,
    list_at_risk_rows_for_courses,
)
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
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class AtRiskThresholds:
    """The administrator-tunable bar for "at risk", resolved once.

    Held together in one object because the SQL filter and the
    human-readable reason text must be generated from the SAME numbers. The
    previous code kept the thresholds in two places -- literals in the SQL
    and module constants in Python -- so a teacher could in principle read
    "threshold: 7 days" on a row selected by a different rule.
    """

    inactivity_days: int
    low_completion_percent: int
    grace_period_days: int


async def resolve_at_risk_thresholds(
    db: AsyncSession, organization_id: UUID | None = None
) -> AtRiskThresholds:
    settings = await resolve_settings(db, organization_id)
    return AtRiskThresholds(
        inactivity_days=int(settings["progress.at_risk_inactivity_days"]),
        low_completion_percent=int(settings["progress.at_risk_low_completion_percent"]),
        grace_period_days=int(settings["progress.at_risk_grace_period_days"]),
    )


def classify_at_risk_reasons(
    row: AtRiskRow, thresholds: AtRiskThresholds
) -> list[AtRiskReason]:
    """Why this row is at risk, in severity order.

    Every ``detail`` states the threshold that fired alongside the observed
    value, so a teacher can tell "inactive 12 days (threshold 7)" from
    "inactive 8 days (threshold 7)" without opening the settings page.

    Returns an empty list when nothing fires. The SQL already filters to
    rows that trip at least one rule, so an empty result means the Python
    and the SQL disagree -- callers treat it as "not at risk" rather than
    inventing a reason.
    """
    reasons: list[AtRiskReason] = []
    days = row.days_since_last_engagement
    if row.last_engagement_at is None:
        reasons.append(
            AtRiskReason(
                code="no_engagement",
                detail="No material engagement events recorded.",
            )
        )
    elif days is not None and days >= thresholds.inactivity_days:
        reasons.append(
            AtRiskReason(
                code="inactive",
                detail=(
                    f"No engagement for {int(days)} days "
                    f"(threshold: {thresholds.inactivity_days})."
                ),
            )
        )
    if int(row.completion_percent) < thresholds.low_completion_percent:
        reasons.append(
            AtRiskReason(
                code="low_completion",
                detail=(
                    f"Average lesson completion {int(row.completion_percent)}% "
                    f"below the {thresholds.low_completion_percent}% threshold."
                ),
            )
        )
    return reasons


def _to_student(row: AtRiskRow, reasons: list[AtRiskReason]) -> AtRiskStudent:
    return AtRiskStudent(
        user_id=row.user_id,
        completion_percent=row.completion_percent,
        days_since_last_engagement=(
            int(row.days_since_last_engagement)
            if row.days_since_last_engagement is not None
            else None
        ),
        reasons=reasons,
    )


async def get_roster_progress(db: AsyncSession, course_id: UUID) -> RosterProgressRead:
    students = await list_course_roster_progress(db, course_id)
    return RosterProgressRead(course_id=course_id, students=students)


async def get_at_risk_students(db: AsyncSession, course_id: UUID) -> AtRiskListRead:
    thresholds = await resolve_at_risk_thresholds(db)
    rows = await list_at_risk_rows_for_courses(
        db,
        [course_id],
        inactivity_days=thresholds.inactivity_days,
        low_completion_percent=thresholds.low_completion_percent,
        grace_period_days=thresholds.grace_period_days,
    )
    students = [
        _to_student(row, reasons)
        for row in rows
        if (reasons := classify_at_risk_reasons(row, thresholds))
    ]
    return AtRiskListRead(course_id=course_id, students=students)


async def count_students_needing_attention(
    db: AsyncSession, course_ids: Sequence[UUID]
) -> int:
    """DISTINCT students with at least one active risk signal across courses.

    Distinct *students*, not signals and not (student, course) pairs: a
    learner struggling in three of a teacher's courses is one person who
    needs attention, and counting them three times inflates the number the
    dashboard asks a teacher to act on.
    """
    thresholds = await resolve_at_risk_thresholds(db)
    rows = await list_at_risk_rows_for_courses(
        db,
        course_ids,
        inactivity_days=thresholds.inactivity_days,
        low_completion_percent=thresholds.low_completion_percent,
        grace_period_days=thresholds.grace_period_days,
    )
    return len({row.user_id for row in rows if classify_at_risk_reasons(row, thresholds)})


__all__ = [
    "AtRiskThresholds",
    "classify_at_risk_reasons",
    "count_students_needing_attention",
    "get_at_risk_students",
    "get_roster_progress",
    "resolve_at_risk_thresholds",
]
