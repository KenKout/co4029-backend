"""Generation-pipeline support for Phase 3 formats + Phase 7 question types.

Covers the adoption work that taught the AI pipeline about the capabilities the
manual authoring path already had:

* **Stage 2 (Phase 3)** — ``prompt_format`` / ``hint_format`` /
  ``explanation_format`` flow through the parser and fail SAFE (an unknown or
  drifted value becomes ``plain``, never trusted as HTML).
* **Stage 3/4 (Phase 7)** — ``numerical``, ``matching``, ``ordering`` and
  multi-select ``multiple_choice`` parse, and malformed variants are DROPPED
  rather than persisted with a broken/ambiguous answer key.
* **Stage 5** — the validation-stage projections flatten the new answer shapes
  into readable text, so the reviewer LLM can judge groundedness instead of
  seeing an empty ``correct_answer`` and rejecting everything.

The request schema and the parser vocabularies are pinned as EQUAL, because a
type accepted at the HTTP boundary but rejected by the parser produces a
"successful" run that silently generates nothing (the ``code`` bug).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    QuizQuestionType,
    parse_generation_response,
    question_for_review,
)
from abridgeai.features.quizzes.ai.stages.validation.logic import _question_for_review
from abridgeai.features.quizzes.schemas.run import QuestionType


def _wrap(entry: dict[str, Any]) -> dict[str, Any]:
    base = {
        "question": "Q?",
        "explanation": "because",
        "hint": "think",
        "bloom_level": "understand",
        "difficulty": "medium",
    }
    return {"questions": [{**base, **entry}]}


def _parse_one(entry: dict[str, Any]):
    out = parse_generation_response(_wrap(entry))
    return out[0] if out else None


# ---------------------------------------------------------------------------
# Vocabulary lockstep — the ``code`` class of bug
# ---------------------------------------------------------------------------


def _literal_values(annotation: Any) -> set[str]:
    from typing import get_args

    return set(get_args(annotation))


def test_request_and_parser_vocabularies_are_equal() -> None:
    """A type accepted by the API but rejected by the parser silently drops."""
    assert _literal_values(QuestionType) == _literal_values(QuizQuestionType)


def test_code_is_not_generatable() -> None:
    assert "code" not in _literal_values(QuestionType)
    assert "code" not in _literal_values(QuizQuestionType)


def test_phase7_types_are_generatable() -> None:
    vocab = _literal_values(QuestionType)
    assert {"numerical", "matching", "ordering"} <= vocab


# ---------------------------------------------------------------------------
# Stage 2 — rich formats fail safe
# ---------------------------------------------------------------------------


def test_formats_pass_through() -> None:
    q = _parse_one(
        {
            "question_type": "short_answer",
            "correct_answer": "etl",
            "prompt_format": "markdown",
            "explanation_format": "html",
        }
    )
    assert q is not None
    assert q.prompt_format == "markdown"
    assert q.explanation_format == "html"


def test_unknown_format_falls_back_to_plain() -> None:
    """Fails safe: a drifted/hostile discriminator must not be trusted as HTML."""
    for bogus in ("javascript", "HTMLish", "", "rich-text"):
        q = _parse_one(
            {
                "question_type": "short_answer",
                "correct_answer": "etl",
                "prompt_format": bogus,
            }
        )
        assert q is not None
        assert q.prompt_format == "plain", bogus


def test_format_is_case_insensitive() -> None:
    q = _parse_one(
        {"question_type": "short_answer", "correct_answer": "x", "prompt_format": "HTML"}
    )
    assert q is not None
    assert q.prompt_format == "html"


# ---------------------------------------------------------------------------
# Stage 3 — numerical
# ---------------------------------------------------------------------------


def test_numerical_parses() -> None:
    q = _parse_one({"question_type": "numerical", "numeric_answer": 10.5, "numeric_tolerance": 0.5})
    assert q is not None
    assert q.numeric_answer == Decimal("10.5")
    assert q.numeric_tolerance == Decimal("0.5")
    assert q.options == []


def test_numerical_accepts_answer_via_correct_answer_alias() -> None:
    """Models often emit the value in ``correct_answer`` instead."""
    q = _parse_one({"question_type": "numerical", "correct_answer": "42"})
    assert q is not None
    assert q.numeric_answer == Decimal("42")


def test_numerical_zero_is_valid() -> None:
    q = _parse_one({"question_type": "numerical", "numeric_answer": 0})
    assert q is not None
    assert q.numeric_answer == Decimal("0")


def test_numerical_without_answer_is_dropped() -> None:
    """No expected value → permanently ungradeable, so reject at the boundary."""
    assert _parse_one({"question_type": "numerical"}) is None


def test_numerical_negative_tolerance_is_dropped() -> None:
    assert (
        _parse_one({"question_type": "numerical", "numeric_answer": 5, "numeric_tolerance": -1})
        is None
    )


# ---------------------------------------------------------------------------
# Stage 4 — matching
# ---------------------------------------------------------------------------


def test_matching_parses() -> None:
    q = _parse_one(
        {
            "question_type": "matching",
            "match_pairs": [
                {"left": "France", "right": "Paris"},
                {"left": "Japan", "right": "Tokyo"},
            ],
        }
    )
    assert q is not None
    assert q.match_pairs == [
        {"left": "France", "right": "Paris"},
        {"left": "Japan", "right": "Tokyo"},
    ]


def test_matching_accepts_key_aliases() -> None:
    """``term``/``definition`` and ``prompt``/``answer`` drift is tolerated."""
    q = _parse_one(
        {
            "question_type": "matching",
            "match_pairs": [
                {"term": "A", "definition": "1"},
                {"prompt": "B", "answer": "2"},
            ],
        }
    )
    assert q is not None
    assert q.match_pairs == [{"left": "A", "right": "1"}, {"left": "B", "right": "2"}]


def test_matching_single_pair_is_dropped() -> None:
    assert (
        _parse_one({"question_type": "matching", "match_pairs": [{"left": "A", "right": "1"}]})
        is None
    )


def test_matching_duplicate_right_is_dropped() -> None:
    """Two identical right values make two pairings equally defensible."""
    assert (
        _parse_one(
            {
                "question_type": "matching",
                "match_pairs": [
                    {"left": "A", "right": "same"},
                    {"left": "B", "right": "same"},
                ],
            }
        )
        is None
    )


def test_matching_duplicate_left_is_dropped() -> None:
    assert (
        _parse_one(
            {
                "question_type": "matching",
                "match_pairs": [
                    {"left": "dup", "right": "1"},
                    {"left": "dup", "right": "2"},
                ],
            }
        )
        is None
    )


def test_matching_parses_with_distractors() -> None:
    """Distractors are extra unpaired right-side choices; they parse onto the
    dedicated field and don't disturb the pairs."""
    q = _parse_one(
        {
            "question_type": "matching",
            "match_pairs": [
                {"left": "France", "right": "Paris"},
                {"left": "Japan", "right": "Tokyo"},
            ],
            "match_distractors": ["Berlin", "Madrid"],
        }
    )
    assert q is not None
    assert q.match_distractors == ["Berlin", "Madrid"]


def test_matching_distractor_alias_and_object_shape() -> None:
    """``distractors`` alias and object entries ({right/value/text}) are
    tolerated the way ``match_pairs`` aliases are."""
    q = _parse_one(
        {
            "question_type": "matching",
            "match_pairs": [
                {"left": "A", "right": "1"},
                {"left": "B", "right": "2"},
            ],
            "distractors": [{"right": "3"}, "4"],
        }
    )
    assert q is not None
    assert q.match_distractors == ["3", "4"]


def test_matching_distractor_colliding_with_answer_is_dropped() -> None:
    """A distractor equal to a correct right value is rejected — it would make
    that value both right and wrong."""
    assert (
        _parse_one(
            {
                "question_type": "matching",
                "match_pairs": [
                    {"left": "A", "right": "1"},
                    {"left": "B", "right": "2"},
                ],
                "match_distractors": ["2"],
            }
        )
        is None
    )


# ---------------------------------------------------------------------------
# Stage 4 — ordering
# ---------------------------------------------------------------------------


def test_ordering_parses() -> None:
    q = _parse_one({"question_type": "ordering", "ordering_sequence": ["one", "two", "three"]})
    assert q is not None
    assert q.ordering_sequence == ["one", "two", "three"]


def test_ordering_accepts_objects_with_positions() -> None:
    """Unordered objects carrying explicit positions are sorted into order."""
    q = _parse_one(
        {
            "question_type": "ordering",
            "ordering_sequence": [
                {"item": "c", "position": 3},
                {"item": "a", "position": 1},
                {"item": "b", "position": 2},
            ],
        }
    )
    assert q is not None
    assert q.ordering_sequence == ["a", "b", "c"]


def test_ordering_too_few_items_is_dropped() -> None:
    assert _parse_one({"question_type": "ordering", "ordering_sequence": ["a", "b"]}) is None


def test_ordering_duplicate_items_is_dropped() -> None:
    """Duplicates make the exact-sequence answer ambiguous."""
    assert _parse_one({"question_type": "ordering", "ordering_sequence": ["a", "b", "a"]}) is None


# ---------------------------------------------------------------------------
# Stage 3 — multi-select MCQ
# ---------------------------------------------------------------------------


def _mcq(correct: Any, **extra: Any) -> dict[str, Any]:
    return {
        "question_type": "multiple_choice",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct_answer": correct,
        **extra,
    }


def test_multi_select_parses_from_letter_array() -> None:
    q = _parse_one(_mcq(["A", "C"], single_answer=False))
    assert q is not None
    assert q.single_answer is False
    assert sorted(o.option_key for o in q.options if o.is_correct) == ["A", "C"]


def test_multi_select_inferred_from_multiple_correct_letters() -> None:
    """A model that lists two letters without the flag still works."""
    q = _parse_one(_mcq(["A", "B"]))
    assert q is not None
    assert q.single_answer is False


def test_multi_select_accepts_comma_separated_letters() -> None:
    q = _parse_one(_mcq("A, C", single_answer=False))
    assert q is not None
    assert sorted(o.option_key for o in q.options if o.is_correct) == ["A", "C"]


def test_single_answer_mcq_still_requires_exactly_one() -> None:
    q = _parse_one(_mcq("B"))
    assert q is not None
    assert q.single_answer is True
    assert sum(1 for o in q.options if o.is_correct) == 1


def test_multi_select_with_one_correct_is_dropped() -> None:
    """Checkboxes with a single answer misleads learners."""
    assert _parse_one(_mcq(["A"], single_answer=False)) is None


def test_multi_select_with_all_correct_is_dropped() -> None:
    assert _parse_one(_mcq(["A", "B", "C", "D"], single_answer=False)) is None


# ---------------------------------------------------------------------------
# Stage 5 — validation-stage projections
# ---------------------------------------------------------------------------


def test_review_projection_renders_numerical_answer() -> None:
    q = _parse_one({"question_type": "numerical", "numeric_answer": 3, "numeric_tolerance": 0})
    assert q is not None
    assert question_for_review(q)["correct_answer"] == "3 (tolerance 0)"


def test_review_projection_renders_matching_pairs() -> None:
    q = _parse_one(
        {
            "question_type": "matching",
            "match_pairs": [
                {"left": "Extract", "right": "Reads"},
                {"left": "Load", "right": "Writes"},
            ],
        }
    )
    assert q is not None
    assert question_for_review(q)["correct_answer"] == ("Extract -> Reads; Load -> Writes")


def test_review_projection_renders_ordering_sequence() -> None:
    q = _parse_one({"question_type": "ordering", "ordering_sequence": ["E", "T", "L"]})
    assert q is not None
    assert question_for_review(q)["correct_answer"] == "1. E; 2. T; 3. L"


def test_review_projection_lists_all_multi_select_letters() -> None:
    q = _parse_one(_mcq(["A", "C"], single_answer=False))
    assert q is not None
    assert question_for_review(q)["correct_answer"] == "A, C"


def test_validation_stage_projection_matches_parser_projection() -> None:
    """Both projections must agree, or the reviewer sees a different question
    depending on which code path built its input."""
    cases: list[dict[str, Any]] = [
        {"question_type": "numerical", "numeric_answer": 3, "numeric_tolerance": 0},
        {
            "question_type": "matching",
            "match_pairs": [
                {"left": "Extract", "right": "Reads"},
                {"left": "Load", "right": "Writes"},
            ],
        },
        {"question_type": "ordering", "ordering_sequence": ["E", "T", "L"]},
    ]
    for case in cases:
        q = _parse_one(case)
        assert q is not None, case
        via_parser = question_for_review(q)
        via_validation = _question_for_review(q.model_dump())
        assert via_parser["correct_answer"] == via_validation["correct_answer"], case
