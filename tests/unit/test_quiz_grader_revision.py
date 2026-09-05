"""Unit tests for the revision-snapshot-aware grader (Phase 1 regrade).

These exercise :func:`grade_answer_against_revision` — the pure function the
regrade engine uses to re-grade a stored answer against the CURRENT question
definition snapshot. No DB; snapshots are plain dicts.
"""

from __future__ import annotations

from decimal import Decimal

from abridgeai.features.quizzes.services.grader import grade_answer_against_revision


def test_mcq_regrade_against_snapshot_flips_when_key_changes() -> None:
    old = {
        "question_type": "multiple_choice",
        "options": [
            {"option_key": "A", "is_correct": True},
            {"option_key": "B", "is_correct": False},
        ],
    }
    new = {
        "question_type": "multiple_choice",
        "options": [
            {"option_key": "A", "is_correct": False},
            {"option_key": "B", "is_correct": True},
        ],
    }
    # Student picked A: correct under old key, wrong under the new key.
    assert (
        grade_answer_against_revision(old, selected_option_key="A", answer_text=None).is_correct
        is True
    )
    assert (
        grade_answer_against_revision(new, selected_option_key="A", answer_text=None).is_correct
        is False
    )


def test_mcq_no_selection_scores_zero() -> None:
    snap = {
        "question_type": "multiple_choice",
        "options": [{"option_key": "A", "is_correct": True}],
    }
    result = grade_answer_against_revision(snap, selected_option_key=None, answer_text=None)
    assert result.is_correct is False
    assert result.points_awarded == Decimal("0")


def test_true_false_grades_by_key() -> None:
    snap = {
        "question_type": "true_false",
        "options": [
            {"option_key": "T", "is_correct": False},
            {"option_key": "F", "is_correct": True},
        ],
    }
    assert grade_answer_against_revision(snap, selected_option_key="F", answer_text=None).is_correct
    assert not grade_answer_against_revision(
        snap, selected_option_key="T", answer_text=None
    ).is_correct


def test_short_answer_normalized_match() -> None:
    snap = {"question_type": "short_answer", "correct_answer": "Time-Variant"}
    # Case + hyphen normalized: "time variant" == "Time-Variant".
    assert grade_answer_against_revision(
        snap, selected_option_key=None, answer_text="time variant"
    ).is_correct
    assert not grade_answer_against_revision(
        snap, selected_option_key=None, answer_text="invariant"
    ).is_correct


def test_fill_blank_positional_list() -> None:
    snap = {"question_type": "fill_blank", "correct_answer": ["alpha", "beta"]}
    assert grade_answer_against_revision(
        snap, selected_option_key=None, answer_text='["alpha", "beta"]'
    ).is_correct
    # Wrong order fails.
    assert not grade_answer_against_revision(
        snap, selected_option_key=None, answer_text='["beta", "alpha"]'
    ).is_correct


def test_code_and_unknown_always_zero() -> None:
    for qtype in ("code", "something_new"):
        snap = {"question_type": qtype}
        assert not grade_answer_against_revision(
            snap, selected_option_key=None, answer_text="anything"
        ).is_correct


def test_numerical_regrade_uses_tolerance() -> None:
    snap = {
        "question_type": "numerical",
        "numeric_answer": Decimal("10"),
        "numeric_tolerance": Decimal("0.25"),
    }
    assert grade_answer_against_revision(
        snap, selected_option_key=None, answer_text="10.2"
    ).is_correct
    assert not grade_answer_against_revision(
        snap, selected_option_key=None, answer_text="10.3"
    ).is_correct


def test_matching_regrade_uses_current_pairs() -> None:
    snap = {
        "question_type": "matching",
        "match_pairs": [
            {"left": "France", "right": "Paris"},
            {"left": "Japan", "right": "Tokyo"},
        ],
    }
    assert grade_answer_against_revision(
        snap,
        selected_option_key=None,
        answer_text='{"France":"Paris","Japan":"Tokyo"}',
    ).is_correct


def test_ordering_regrade_uses_current_sequence() -> None:
    snap = {
        "question_type": "ordering",
        "ordering_sequence": ["first", "second", "third"],
    }
    assert grade_answer_against_revision(
        snap,
        selected_option_key=None,
        answer_text='["first","second","third"]',
    ).is_correct
    assert not grade_answer_against_revision(
        snap,
        selected_option_key=None,
        answer_text='["second","first","third"]',
    ).is_correct


def test_multi_select_regrade_compares_stable_option_keys() -> None:
    snap = {
        "question_type": "multiple_choice",
        "single_answer": False,
        "options": [
            {"option_key": "A", "is_correct": True},
            {"option_key": "B", "is_correct": False},
            {"option_key": "C", "is_correct": True},
        ],
    }
    assert grade_answer_against_revision(
        snap,
        selected_option_key=None,
        selected_option_keys=["A", "C"],
        answer_text=None,
    ).is_correct
