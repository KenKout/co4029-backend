"""Boundary behavior for persisted quiz percentages."""

from decimal import Decimal

from abridgeai.features.quizzes.services.taking import _calculate_score_percent


def test_score_rounds_before_pass_threshold_comparison() -> None:
    score = _calculate_score_percent(Decimal("2"), 3)

    assert score == Decimal("66.67")
    assert score >= Decimal("66.67")
