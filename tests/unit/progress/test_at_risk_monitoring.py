"""Unit tests for the at-risk risk engine.

Covers the two pieces that are pure Python and therefore provable without a
database: how a row becomes a set of reasons, and how rows across many
courses collapse into the dashboard's headline count. The SQL filter (grace
period, thresholds) is exercised in ``tests/integration/test_progress.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from abridgeai.features.progress.queries.analytics import AtRiskRow
from abridgeai.features.progress.services import monitoring
from abridgeai.features.progress.services.monitoring import (
    AtRiskThresholds,
    classify_at_risk_reasons,
)

DEFAULTS = AtRiskThresholds(
    inactivity_days=7, low_completion_percent=30, grace_period_days=14
)


def _row(
    *,
    user_id: UUID | None = None,
    course_id: UUID | None = None,
    days_inactive: float | None = 0.0,
    completion: int = 100,
    quiz_failed: int = 0,
    quiz_total: int = 0,
    quiz_ungraded: int = 0,
    interview_failed: int = 0,
    interview_total: int = 0,
    interview_pending: int = 0,
) -> AtRiskRow:
    now = datetime.now(tz=UTC)
    return AtRiskRow(
        course_id=course_id or uuid4(),
        user_id=user_id or uuid4(),
        enrolled_at=now - timedelta(days=90),
        last_engagement_at=(
            None if days_inactive is None else now - timedelta(days=days_inactive)
        ),
        completion_percent=Decimal(completion),
        failed_quiz_attempts=quiz_failed,
        total_quiz_attempts=quiz_total,
        ungraded_quiz_attempts=quiz_ungraded,
        failed_interview_sessions=interview_failed,
        total_interview_sessions=interview_total,
        pending_interview_sessions=interview_pending,
        days_since_last_engagement=days_inactive,
        days_since_enrolled=90.0,
    )


def test_no_engagement_reported_without_an_inactivity_number() -> None:
    """A student who never engaged has no "days inactive" to quote.

    Reporting "inactive 90 days" for someone with no engagement row would
    invent a number from the enrolment date rather than from behaviour.
    """
    reasons = classify_at_risk_reasons(_row(days_inactive=None), DEFAULTS)
    assert [r.code for r in reasons] == ["no_engagement"]
    assert "threshold" not in reasons[0].detail


def test_inactivity_detail_states_the_threshold_that_fired() -> None:
    """FR-026: the reason must name the bar, not just the observation."""
    reasons = classify_at_risk_reasons(_row(days_inactive=12.4), DEFAULTS)
    assert [r.code for r in reasons] == ["inactive"]
    assert "12 days" in reasons[0].detail
    assert "threshold: 7" in reasons[0].detail


def test_reason_text_follows_a_retuned_threshold() -> None:
    """The quoted threshold comes from settings, not from a literal.

    The old code hard-coded "threshold: 7" in the string while the SQL held
    its own `7`; retuning one left the other lying.
    """
    tuned = AtRiskThresholds(
        inactivity_days=30, low_completion_percent=50, grace_period_days=14
    )
    reasons = classify_at_risk_reasons(_row(days_inactive=40, completion=45), tuned)
    details = " ".join(r.detail for r in reasons)
    assert "threshold: 30" in details
    assert "50% threshold" in details
    assert "threshold: 7" not in details


def test_inactive_exactly_at_the_threshold_fires() -> None:
    """The threshold is inclusive — 7 days at a 7-day bar is at risk."""
    assert classify_at_risk_reasons(_row(days_inactive=7.0), DEFAULTS)


def test_completion_exactly_at_the_threshold_does_not_fire() -> None:
    """30% at a 30% bar is not "below 30%"."""
    reasons = classify_at_risk_reasons(_row(days_inactive=0.0, completion=30), DEFAULTS)
    assert reasons == []


def test_multiple_signals_are_ordered_engagement_first() -> None:
    """Signal order is fixed: assessment failures, completion, then inactivity.

    The engine deliberately lists inactivity LAST — it is the least specific
    signal and usually a symptom of the earlier ones — so the primary reason
    stays the most actionable (FR-022 renders reasons[0] as the primary).
    """
    reasons = classify_at_risk_reasons(_row(days_inactive=20, completion=5), DEFAULTS)
    assert [r.code for r in reasons] == ["low_completion", "inactive"]


def test_healthy_row_yields_no_reasons() -> None:
    assert classify_at_risk_reasons(_row(days_inactive=1, completion=80), DEFAULTS) == []


@pytest.mark.asyncio
async def test_count_is_distinct_students_not_signals(monkeypatch) -> None:
    """FR-020: one struggling person is one count, however many ways.

    This student is at risk in three courses and trips two rules in each.
    Counting rows would say 3 and counting signals would say 6; the number
    a teacher acts on is 1.
    """
    student = uuid4()
    rows = [
        _row(user_id=student, course_id=uuid4(), days_inactive=20, completion=5)
        for _ in range(3)
    ]

    async def fake_rows(_db, _ids, **_kw):
        return rows

    async def fake_thresholds(_db, _org=None):
        return DEFAULTS

    monkeypatch.setattr(monitoring, "list_at_risk_rows_for_courses", fake_rows)
    monkeypatch.setattr(monitoring, "resolve_at_risk_thresholds", fake_thresholds)

    assert await monitoring.count_students_needing_attention(None, [uuid4()]) == 1


@pytest.mark.asyncio
async def test_count_excludes_rows_that_score_no_reason(monkeypatch) -> None:
    """Python is the arbiter, not the SQL's WHERE clause.

    If the two ever disagree, a row the classifier cannot explain is left
    out rather than counted as an unexplained risk.
    """
    healthy, risky = uuid4(), uuid4()
    rows = [
        _row(user_id=healthy, days_inactive=1, completion=90),
        _row(user_id=risky, days_inactive=30, completion=5),
    ]

    async def fake_rows(_db, _ids, **_kw):
        return rows

    async def fake_thresholds(_db, _org=None):
        return DEFAULTS

    monkeypatch.setattr(monitoring, "list_at_risk_rows_for_courses", fake_rows)
    monkeypatch.setattr(monitoring, "resolve_at_risk_thresholds", fake_thresholds)

    assert await monitoring.count_students_needing_attention(None, [uuid4()]) == 1


@pytest.mark.asyncio
async def test_no_courses_short_circuits_without_a_query() -> None:
    """A teacher with no authorable courses must not hit the database."""
    from abridgeai.features.progress.queries.analytics import (
        list_at_risk_rows_for_courses,
    )

    # `db=None` would explode on any real execute; returning [] proves the
    # guard ran before the round trip.
    assert await list_at_risk_rows_for_courses(
        None, [], inactivity_days=7, low_completion_percent=30, grace_period_days=14
    ) == []
