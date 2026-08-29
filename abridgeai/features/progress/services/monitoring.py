from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
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

    Reasons are ordered by how actionable they are, not by when they fired:
    assessment performance first (a failing grade is the most specific
    signal a teacher can act on), then incomplete work, then inactivity.
    A student can be 100% through the lessons and still at risk because
    their attempts are failing -- and the first line must say that, not
    that they have been quiet for three weeks.

    Returns an empty list when nothing fires. The SQL already filters to
    rows that trip at least one rule, so an empty result means the Python
    and the SQL disagree -- callers treat it as "not at risk" rather than
    inventing a reason.
    """
    reasons: list[AtRiskReason] = []

    # --- 1. Assessment performance: failing or ungraded attempts. The most
    # specific, most actionable signal, so it leads the list.
    failed_quiz = row.failed_quiz_attempts
    failed_iv = row.failed_interview_sessions
    if failed_quiz > 0 or failed_iv > 0:
        parts = []
        if failed_quiz:
            parts.append(
                f"{failed_quiz} quiz attempt{'s' if failed_quiz != 1 else ''} failed"
            )
        if failed_iv:
            parts.append(
                f"{failed_iv} interview{'s' if failed_iv != 1 else ''} failed"
            )
        reasons.append(
            AtRiskReason(
                code="failed_assessments",
                detail=", ".join(parts) + ".",
            )
        )

    ungraded_quiz = row.ungraded_quiz_attempts
    pending_iv = row.pending_interview_sessions
    if ungraded_quiz > 0 or pending_iv > 0:
        parts = []
        if ungraded_quiz:
            parts.append(
                f"{ungraded_quiz} quiz attempt{'s' if ungraded_quiz != 1 else ''} not graded"
            )
        if pending_iv:
            parts.append(
                f"{pending_iv} interview{'s' if pending_iv != 1 else ''} awaiting grading"
            )
        reasons.append(
            AtRiskReason(
                code="ungraded_assessments",
                detail=", ".join(parts) + ".",
            )
        )

    # --- 2. Incomplete critical work: lesson progress below the bar.
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

    # --- 3. Inactivity, last: the least specific signal, and the one most
    # likely to be a symptom of 1 or 2 rather than an independent cause.
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


def severity_for(reasons: list[AtRiskReason]) -> str:
    """Coarse severity band for a set of reasons (FR-021).

    "high" when the student is absent -- no engagement at all, or silent
    past the inactivity threshold. "medium" when they are present but
    behind. The distinction matters because the two need different
    interventions: an absent student needs contacting, a behind one needs
    help with the material.
    """
    codes = {r.code for r in reasons}
    if codes & {"no_engagement", "inactive"}:
        return "high"
    return "medium"


@dataclass(frozen=True)
class StudentAtRisk:
    """One (student, course) risk row, scored and ready to render."""

    user_id: UUID
    course_id: UUID
    completion_percent: Decimal
    last_engagement_at: datetime | None
    days_since_last_engagement: int | None
    primary_reason: str
    signal_count: int
    severity: str


async def list_students_needing_attention(
    db: AsyncSession, course_ids: Sequence[UUID]
) -> list[StudentAtRisk]:
    """Scored risk rows across ``course_ids``, worst first.

    One row per (student, course): a student at risk in two of a teacher's
    courses appears twice here, because the follow-up is per course. The
    headline COUNT deliberately deduplicates -- see
    :func:`count_students_needing_attention` -- so the tile and this list
    answer different questions and are expected to differ.

    Ordered high severity first, then by how long the student has been
    silent, so the top of the list is the person who has been gone longest.
    """
    thresholds = await resolve_at_risk_thresholds(db)
    rows = await list_at_risk_rows_for_courses(
        db,
        course_ids,
        inactivity_days=thresholds.inactivity_days,
        low_completion_percent=thresholds.low_completion_percent,
        grace_period_days=thresholds.grace_period_days,
    )
    scored: list[StudentAtRisk] = []
    for row in rows:
        reasons = classify_at_risk_reasons(row, thresholds)
        if not reasons:
            continue
        scored.append(
            StudentAtRisk(
                user_id=row.user_id,
                course_id=row.course_id,
                completion_percent=row.completion_percent,
                last_engagement_at=row.last_engagement_at,
                days_since_last_engagement=(
                    int(row.days_since_last_engagement)
                    if row.days_since_last_engagement is not None
                    else None
                ),
                primary_reason=reasons[0].detail,
                signal_count=len(reasons),
                severity=severity_for(reasons),
            )
        )
    # A student who never engaged has no "days silent" to sort on but is the
    # most absent of all, so they sort above every finite gap.
    scored.sort(
        key=lambda s: (
            s.severity != "high",
            -(
                float("inf")
                if s.last_engagement_at is None
                else (s.days_since_last_engagement or 0)
            ),
        )
    )
    return scored


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
    "StudentAtRisk",
    "count_students_needing_attention",
    "list_students_needing_attention",
    "severity_for",
    "get_at_risk_students",
    "get_roster_progress",
    "resolve_at_risk_thresholds",
]
