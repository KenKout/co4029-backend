"""Option-validation rules for single vs multi-answer MCQ (Phase 7).

Guards the bug where the teacher UI offered an "Allow multiple correct answers"
toggle but the option rules still demanded *exactly one* correct option — so a
multi-select question could never be saved.

Two validators had to agree:

* create path — ``_validate_question_options(question_type, options,
  single_answer=...)``
* update path — ``_update_question_options`` (previously hardcoded "exactly four
  options / exactly one correct")

These tests pin the create-path rules, which are pure and DB-free. The update
path shares the same rule set and is covered by the service-level checks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.services.authoring import _validate_question_options


def _opt(key: str, *, correct: bool) -> SimpleNamespace:
    return SimpleNamespace(option_key=key, option_text=f"Option {key}", is_correct=correct)


def test_single_answer_requires_exactly_one_correct() -> None:
    opts = [_opt("A", correct=True), _opt("B", correct=True), _opt("C", correct=False)]
    with pytest.raises(AppError, match="exactly one correct"):
        _validate_question_options("multiple_choice", opts, single_answer=True)


def test_single_answer_accepts_one_correct() -> None:
    opts = [_opt("A", correct=True), _opt("B", correct=False)]
    _validate_question_options("multiple_choice", opts, single_answer=True)


def test_multi_select_accepts_two_correct() -> None:
    """The exact case the UI toggle enables."""
    opts = [
        _opt("A", correct=True),
        _opt("B", correct=True),
        _opt("C", correct=False),
        _opt("D", correct=False),
    ]
    _validate_question_options("multiple_choice", opts, single_answer=False)


def test_multi_select_accepts_all_correct() -> None:
    opts = [_opt("A", correct=True), _opt("B", correct=True), _opt("C", correct=True)]
    _validate_question_options("multiple_choice", opts, single_answer=False)


def test_multi_select_still_requires_at_least_one_correct() -> None:
    opts = [_opt("A", correct=False), _opt("B", correct=False)]
    with pytest.raises(AppError, match="at least one correct"):
        _validate_question_options("multiple_choice", opts, single_answer=False)


def test_option_count_relaxed_to_two_through_ten() -> None:
    """The old rule was a hard "exactly four"."""
    two = [_opt("A", correct=True), _opt("B", correct=False)]
    _validate_question_options("multiple_choice", two, single_answer=True)

    ten = [_opt(chr(65 + i), correct=(i == 0)) for i in range(10)]
    _validate_question_options("multiple_choice", ten, single_answer=True)

    with pytest.raises(AppError, match="between 2 and 10"):
        _validate_question_options("multiple_choice", [_opt("A", correct=True)], single_answer=True)


def test_true_false_stays_strict() -> None:
    """true_false is always single-answer regardless of the flag."""
    both = [_opt("T", correct=True), _opt("F", correct=True)]
    with pytest.raises(AppError, match="exactly one correct"):
        _validate_question_options("true_false", both, single_answer=False)
