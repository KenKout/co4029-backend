"""Unit tests for the pure gradebook grading-method reducer (Phase 9)."""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from abridgeai.features.quizzes.services.gradebook import (
    AttemptScore,
    _compute_final_grade,
)


def _score(n: int, pct: str, pts: str) -> AttemptScore:
    return AttemptScore(
        attempt_id=uuid.uuid4(),
        attempt_number=n,
        score_percent=Decimal(pct),
        score_points=Decimal(pts),
    )


def test_highest_picks_max_percent():
    a1, a2 = _score(1, "60.00", "6"), _score(2, "90.00", "9")
    out = _compute_final_grade([a1, a2], "highest")
    assert out.grade_percent == Decimal("90.00")
    assert out.based_on_attempt_id == a2.attempt_id
    assert out.attempts_counted == 2


def test_first_and_last_use_attempt_number():
    a1, a2 = _score(1, "60.00", "6"), _score(2, "90.00", "9")
    first = _compute_final_grade([a2, a1], "first")
    last = _compute_final_grade([a1, a2], "last")
    assert first.based_on_attempt_id == a1.attempt_id
    assert last.based_on_attempt_id == a2.attempt_id


def test_average_rounds_to_two_dp():
    out = _compute_final_grade([_score(1, "60.00", "6"), _score(2, "90.00", "9")], "average")
    assert out.grade_percent == Decimal("75.00")
    assert out.based_on_attempt_id is None


def test_empty_returns_none():
    assert _compute_final_grade([], "highest") is None


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        _compute_final_grade([_score(1, "60.00", "6")], "bogus")
