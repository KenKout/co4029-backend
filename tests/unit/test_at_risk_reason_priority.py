"""At-risk reason priority: assessments before work before inactivity.

The dashboard and student pages render ``reasons[0].detail`` as the
headline reason. A student can be 100% through the lessons and still at
risk because their attempts are failing; the first line must say that,
not that they have been quiet.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from abridgeai.features.progress.queries.analytics import AtRiskRow
from abridgeai.features.progress.services.monitoring import (
    AtRiskThresholds,
    classify_at_risk_reasons,
)

THRESHOLDS = AtRiskThresholds(
    inactivity_days=7,
    low_completion_percent=30,
    grace_period_days=14,
)


def _row(
    *,
    completion_percent: int = 100,
    days_since_last_engagement: float | None = None,
    failed_quiz_attempts: int = 0,
    total_quiz_attempts: int = 0,
    ungraded_quiz_attempts: int = 0,
    failed_interview_sessions: int = 0,
    total_interview_sessions: int = 0,
    pending_interview_sessions: int = 0,
    last_engagement_at: datetime | None = None,
) -> AtRiskRow:
    if last_engagement_at is None and days_since_last_engagement is not None:
        last_engagement_at = datetime.now(tz=UTC) - timedelta(
            days=days_since_last_engagement
        )
    return AtRiskRow(
        course_id=uuid4(),
        user_id=uuid4(),
        enrolled_at=datetime.now(tz=UTC) - timedelta(days=60),
        last_engagement_at=last_engagement_at,
        completion_percent=Decimal(completion_percent),
        failed_quiz_attempts=failed_quiz_attempts,
        total_quiz_attempts=total_quiz_attempts,
        ungraded_quiz_attempts=ungraded_quiz_attempts,
        failed_interview_sessions=failed_interview_sessions,
        total_interview_sessions=total_interview_sessions,
        pending_interview_sessions=pending_interview_sessions,
        days_since_last_engagement=days_since_last_engagement,
        days_since_enrolled=60.0,
    )


def test_failed_assessments_outrank_inactivity() -> None:
    """100% complete + silent + failing attempts: failures are the headline."""
    reasons = classify_at_risk_reasons(
        _row(
            completion_percent=100,
            days_since_last_engagement=22,
            failed_quiz_attempts=3,
            total_quiz_attempts=5,
        ),
        THRESHOLDS,
    )
    codes = [r.code for r in reasons]
    assert codes == ["failed_assessments", "inactive"]
    assert reasons[0].detail == "3 of 5 quiz attempts failed."


def test_ungraded_interviews_outrank_low_completion() -> None:
    reasons = classify_at_risk_reasons(
        _row(
            completion_percent=20,
            days_since_last_engagement=9,
            pending_interview_sessions=1,
        ),
        THRESHOLDS,
    )
    codes = [r.code for r in reasons]
    assert codes == ["ungraded_assessments", "low_completion", "inactive"]
    assert reasons[0].detail == "1 interview awaiting grading."


def test_full_priority_order() -> None:
    reasons = classify_at_risk_reasons(
        _row(
            completion_percent=15,
            days_since_last_engagement=12,
            failed_quiz_attempts=1,
            total_quiz_attempts=4,
            failed_interview_sessions=2,
            total_interview_sessions=10,
            ungraded_quiz_attempts=1,
            pending_interview_sessions=1,
        ),
        THRESHOLDS,
    )
    codes = [r.code for r in reasons]
    assert codes == [
        "failed_assessments",
        "ungraded_assessments",
        "low_completion",
        "inactive",
    ]
    assert reasons[0].detail == "1 of 4 quiz attempts failed, 2 of 10 interviews failed."


def test_combined_counts_read_naturally() -> None:
    reasons = classify_at_risk_reasons(
        _row(
            failed_quiz_attempts=1,
            total_quiz_attempts=3,
            failed_interview_sessions=1,
            total_interview_sessions=2,
            days_since_last_engagement=3,
        ),
        THRESHOLDS,
    )
    assert reasons[0].code == "failed_assessments"
    assert reasons[0].detail == "1 of 3 quiz attempts failed, 1 of 2 interviews failed."
    assert len(reasons) == 1


def test_absent_student_with_clean_assessments_keeps_original_reasons() -> None:
    reasons = classify_at_risk_reasons(
        _row(completion_percent=10, last_engagement_at=None),
        THRESHOLDS,
    )
    codes = [r.code for r in reasons]
    assert codes == ["low_completion", "no_engagement"]
