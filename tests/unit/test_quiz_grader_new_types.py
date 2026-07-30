"""Phase 7: unit tests for the new question-type grading functions.

These exercise the pure per-type graders directly with lightweight stand-in
question objects (no DB), covering numerical tolerance, matching, and ordering.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

from abridgeai.features.quizzes.services.grader import (
    _ONE,
    _ZERO,
    _grade_matching,
    _grade_numerical,
    _grade_ordering,
)


def test_numerical_within_tolerance_is_correct():
    q = SimpleNamespace(numeric_answer=Decimal("3.14"), numeric_tolerance=Decimal("0.01"))
    assert _grade_numerical(q, "3.15") == _ONE
    assert _grade_numerical(q, "3.13") == _ONE


def test_numerical_outside_tolerance_is_wrong():
    q = SimpleNamespace(numeric_answer=Decimal("3.14"), numeric_tolerance=Decimal("0.01"))
    assert _grade_numerical(q, "3.20") == _ZERO


def test_numerical_exact_when_zero_tolerance():
    q = SimpleNamespace(numeric_answer=Decimal("42"), numeric_tolerance=Decimal("0"))
    assert _grade_numerical(q, "42") == _ONE
    assert _grade_numerical(q, "42.1") == _ZERO


def test_numerical_bad_input_is_wrong():
    q = SimpleNamespace(numeric_answer=Decimal("1"), numeric_tolerance=Decimal("0"))
    assert _grade_numerical(q, "not-a-number") == _ZERO
    assert _grade_numerical(q, None) == _ZERO


def test_matching_all_pairs_correct():
    q = SimpleNamespace(
        match_pairs=[{"left": "1", "right": "one"}, {"left": "2", "right": "two"}]
    )
    assert _grade_matching(q, json.dumps({"1": "one", "2": "two"})) == _ONE


def test_matching_one_wrong_pair_fails():
    q = SimpleNamespace(
        match_pairs=[{"left": "1", "right": "one"}, {"left": "2", "right": "two"}]
    )
    assert _grade_matching(q, json.dumps({"1": "one", "2": "three"})) == _ZERO


def test_ordering_exact_sequence_correct():
    q = SimpleNamespace(ordering_sequence=["a", "b", "c"])
    assert _grade_ordering(q, json.dumps(["a", "b", "c"])) == _ONE


def test_ordering_wrong_order_fails():
    q = SimpleNamespace(ordering_sequence=["a", "b", "c"])
    assert _grade_ordering(q, json.dumps(["a", "c", "b"])) == _ZERO


def test_ordering_length_mismatch_fails():
    q = SimpleNamespace(ordering_sequence=["a", "b", "c"])
    assert _grade_ordering(q, json.dumps(["a", "b"])) == _ZERO
