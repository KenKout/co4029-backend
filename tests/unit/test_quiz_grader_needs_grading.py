"""Phase 4: unit tests for the manual-grade predicate."""

from __future__ import annotations

from decimal import Decimal

from abridgeai.features.quizzes.services.grader import GradeResult, needs_manual_grade

_WRONG = GradeResult(is_correct=False, points_awarded=Decimal("0"))
_RIGHT = GradeResult(is_correct=True, points_awarded=Decimal("1"))


def test_code_always_needs_grading():
    assert needs_manual_grade("code", _WRONG) is True
    assert needs_manual_grade("code", _RIGHT) is True  # code never auto-passes


def test_short_answer_needs_grading_only_when_wrong():
    assert needs_manual_grade("short_answer", _WRONG) is True
    assert needs_manual_grade("short_answer", _RIGHT) is False


def test_fill_blank_needs_grading_only_when_wrong():
    assert needs_manual_grade("fill_blank", _WRONG) is True
    assert needs_manual_grade("fill_blank", _RIGHT) is False


def test_mcq_and_tf_never_need_grading():
    assert needs_manual_grade("multiple_choice", _WRONG) is False
    assert needs_manual_grade("true_false", _WRONG) is False
