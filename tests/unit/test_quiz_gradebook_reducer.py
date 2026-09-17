"""Unit tests for the pure gradebook grading-method reducer (Phase 9).

The reducer picks a number; the number decides whether the student passed.
The tests below cover both halves, because the first without the second
only proves that ``max()`` works. The pass rule itself lives inline in
``recompute_final_grade`` as ``grade_percent >= quiz.passing_score_percent``
and has no seam of its own, so it is restated here as ``_passed``; if that
comparison ever changes, these tests keep passing while the product does
something else, which is worth knowing about.
"""

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


# The rule from ``recompute_final_grade``. Restated, not imported: it is an
# inline expression inside an async DB function, with no seam to call.
def _passed(grade, pass_mark: str) -> bool:
    return grade.grade_percent >= Decimal(pass_mark)


class TestTheMethodDecidesTheVerdict:
    """Identical attempts, four methods, different outcomes.

    This is the property a teacher actually chose the setting for. Without
    it the reducer could return any plausible attempt and every existing
    test would still pass.
    """

    def test_first_fails_where_the_others_pass(self) -> None:
        # Failed, then aced it.
        attempts = [_score(1, "30.00", "3"), _score(2, "95.00", "9.5")]
        verdicts = {
            m: _passed(_compute_final_grade(attempts, m), "50")
            for m in ("highest", "first", "last", "average")
        }
        assert verdicts == {
            "highest": True,
            "first": False,
            "last": True,
            "average": True,
        }

    def test_last_fails_where_the_others_pass(self) -> None:
        # Aced it, then failed — the mirror case, which catches a reducer
        # that confuses first with last.
        attempts = [_score(1, "95.00", "9.5"), _score(2, "30.00", "3")]
        verdicts = {
            m: _passed(_compute_final_grade(attempts, m), "50")
            for m in ("highest", "first", "last", "average")
        }
        assert verdicts == {
            "highest": True,
            "first": True,
            "last": False,
            "average": True,
        }

    def test_average_fails_where_highest_passes(self) -> None:
        # Two near misses either side of the mark: only the best attempt
        # clears it.
        attempts = [_score(1, "40.00", "4"), _score(2, "55.00", "5.5")]
        assert _passed(_compute_final_grade(attempts, "highest"), "50") is True
        assert _passed(_compute_final_grade(attempts, "average"), "50") is False


class TestThePassBoundary:
    def test_exactly_the_pass_mark_passes(self) -> None:
        """The comparison is ``>=``. A student on precisely the mark passed."""
        grade = _compute_final_grade([_score(1, "50.00", "5")], "highest")
        assert _passed(grade, "50") is True

    def test_one_hundredth_below_the_mark_fails(self) -> None:
        grade = _compute_final_grade([_score(1, "49.99", "4.999")], "highest")
        assert _passed(grade, "50") is False

    def test_average_rounds_up_across_the_pass_mark(self) -> None:
        """CURRENT BEHAVIOUR, pinned deliberately — and worth a second look.

        The true average of 49.99 and 50.00 is 49.995, which is below the
        mark. The reducer quantizes to two decimal places with ROUND_HALF_UP
        BEFORE the comparison, producing 50.00, so the student passes on an
        average that never reached the pass mark.

        The window is at most half a hundredth of a percentage point, and
        rounding a displayed grade is defensible — but the rounding decides a
        pass here, and nothing else in the codebase says so. If the intended
        rule is "compare the unrounded average", this test is the one that
        should change.
        """
        grade = _compute_final_grade(
            [_score(1, "49.99", "4.999"), _score(2, "50.00", "5")], "average"
        )
        assert grade.grade_percent == Decimal("50.00")
        assert _passed(grade, "50") is True


class TestAttributionOnTies:
    def test_highest_with_equal_scores_attributes_one_of_them(self) -> None:
        """Two attempts at the same top score: the grade must not be ambiguous.

        ``max`` keeps the first maximal element, so attribution follows the
        order the query returned — which is ``attempt_number``. Pinned so a
        change to the ordering surfaces here rather than in a gradebook a
        student disputes.
        """
        a1, a2 = _score(1, "80.00", "8"), _score(2, "80.00", "8")
        out = _compute_final_grade([a1, a2], "highest")
        assert out.grade_percent == Decimal("80.00")
        assert out.based_on_attempt_id == a1.attempt_id
        assert out.attempts_counted == 2
