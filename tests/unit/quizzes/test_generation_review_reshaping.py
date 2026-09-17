"""Reshaping a generated question for the validation stage.

The validator judges groundedness from a compact text view: prompt, options
and a single ``correct_answer`` string. Every question type stores its answer
somewhere different — option rows, a dedicated numeric column, a pair list, a
sequence, the raw payload — so this module is where they are flattened into
one shape.

What makes it worth pinning is the failure mode. A type whose answer does not
reach the view arrives at the validator with an empty ``correct_answer``, and
the validator reads that as an answer the source does not support: the
question is rejected as ungrounded, and the cause looks like a content problem
rather than a reshaping one.

Pure functions over plain dicts — the module imports only ``string``.
"""

from __future__ import annotations

from typing import Any

import pytest

from abridgeai.features.quizzes.ai.stages.generation.review import (
    _render_answer_for_review,
    normalize_question_text,
    question_for_review,
)


def _options(*triples: tuple[str, str, bool]) -> list[dict[str, Any]]:
    return [
        {"option_key": key, "option_text": text, "is_correct": correct}
        for key, text, correct in triples
    ]


class TestEveryTypeSurfacesItsAnswer:
    """The matrix that keeps a whole question type from being rejected."""

    def test_multiple_choice_surfaces_the_correct_letter(self) -> None:
        out = question_for_review(
            {
                "question_type": "multiple_choice",
                "options": _options(("A", "ay", False), ("B", "bee", True)),
            }
        )
        assert out["correct_answer"] == "B"
        assert out["options"] == {"A": "ay", "B": "bee"}

    def test_multi_select_surfaces_every_correct_letter(self) -> None:
        """One letter would understate the answer the validator must judge."""
        out = question_for_review(
            {
                "question_type": "multiple_choice",
                "single_answer": False,
                "options": _options(("A", "ay", True), ("B", "bee", False), ("C", "cee", True)),
            }
        )
        assert out["correct_answer"] == "A, C"

    @pytest.mark.parametrize(
        ("correct_key", "expected"),
        [("T", "True"), ("F", "False")],
    )
    def test_true_false_is_rendered_as_words_not_keys(
        self, correct_key: str, expected: str
    ) -> None:
        """The validator's prompt expects ``True``/``False``, not ``T``/``F``."""
        out = question_for_review(
            {
                "question_type": "true_false",
                "options": _options(
                    ("T", "True", correct_key == "T"),
                    ("F", "False", correct_key == "F"),
                ),
            }
        )
        assert out["correct_answer"] == expected

    def test_short_answer_reads_the_raw_payload(self) -> None:
        out = question_for_review(
            {
                "question_type": "short_answer",
                "original_generated_payload": {"correct_answer": "photosynthesis"},
            }
        )
        assert out["correct_answer"] == "photosynthesis"

    def test_fill_blank_carries_the_whole_blank_list(self) -> None:
        out = question_for_review(
            {
                "question_type": "fill_blank",
                "original_generated_payload": {"correct_answer": ["a", "b"]},
            }
        )
        assert out["correct_answer"] == ["a", "b"]

    def test_numerical_matching_and_ordering_are_flattened(self) -> None:
        """These carry their answer on dedicated fields rather than options.

        Without the flattening they would reach the validator with nothing in
        ``correct_answer`` and every question of the type would be refused.
        """
        numerical = question_for_review(
            {"question_type": "numerical", "numeric_answer": 3, "numeric_tolerance": 0.5}
        )
        assert numerical["correct_answer"] == "3 (tolerance 0.5)"

        matching = question_for_review(
            {
                "question_type": "matching",
                "match_pairs": [
                    {"left": "Extract", "right": "Reads"},
                    {"left": "Load", "right": "Writes"},
                ],
            }
        )
        assert matching["correct_answer"] == "Extract -> Reads; Load -> Writes"

        ordering = question_for_review(
            {"question_type": "ordering", "ordering_sequence": ["Extract", "Transform", "Load"]}
        )
        assert ordering["correct_answer"] == "1. Extract; 2. Transform; 3. Load"


class TestAnswerRendering:
    @pytest.mark.parametrize(
        ("qtype", "data", "expected"),
        [
            ("numerical", {"numeric_answer": 3}, "3"),
            ("numerical", {"numeric_answer": 3, "numeric_tolerance": 0}, "3 (tolerance 0)"),
            ("numerical", {}, ""),
            ("matching", {"match_pairs": [{"left": "L", "right": "R"}]}, "L -> R"),
            ("matching", {"match_pairs": "not a list"}, ""),
            ("matching", {}, ""),
            ("ordering", {"ordering_sequence": ["a", "b"]}, "1. a; 2. b"),
            ("ordering", {"ordering_sequence": "not a list"}, ""),
            ("ordering", {}, ""),
            ("short_answer", {"correct_answer": "x"}, ""),
        ],
    )
    def test_shapes_render_or_fall_back_to_empty(
        self, qtype: str, data: dict[str, Any], expected: str
    ) -> None:
        """A malformed field yields empty rather than raising.

        The stage runs over model output, so a wrong type in one field must
        cost that question its answer text, not the whole batch.
        """
        assert _render_answer_for_review(qtype, data) == expected

    def test_a_zero_tolerance_is_still_reported(self) -> None:
        """Zero is a real tolerance and must not be dropped as falsey."""
        assert _render_answer_for_review(
            "numerical", {"numeric_answer": 3, "numeric_tolerance": 0}
        ) == "3 (tolerance 0)"

    def test_non_dict_pairs_are_skipped_not_fatal(self) -> None:
        rendered = _render_answer_for_review(
            "matching", {"match_pairs": [{"left": "L", "right": "R"}, "junk"]}
        )
        assert rendered == "L -> R"


class TestInputShapes:
    def test_a_question_with_no_options_survives(self) -> None:
        out = question_for_review({"question_type": "multiple_choice"})
        assert out["options"] == {}
        assert out["correct_answer"] is None

    def test_an_object_without_model_dump_is_read_attribute_by_attribute(self) -> None:
        """The duck-typed fallback exists for candidate objects in the pipeline."""

        class Candidate:
            prompt_text = "P"
            question_type = "short_answer"
            options = None
            explanation = "E"
            bloom_level = "apply"
            difficulty = "medium"
            source_refs = None
            original_generated_payload = {"correct_answer": "duck"}

        out = question_for_review(Candidate())
        assert out["correct_answer"] == "duck"
        assert out["prompt_text"] == "P"
        assert out["bloom_level"] == "apply"

    def test_the_view_carries_what_the_validator_reads(self) -> None:
        out = question_for_review(
            {
                "prompt_text": "P",
                "question_type": "multiple_choice",
                "explanation": "E",
                "bloom_level": "apply",
                "difficulty": "hard",
                "options": _options(("A", "ay", True)),
            }
        )
        assert set(out) == {
            "prompt_text",
            "question_type",
            "options",
            "correct_answer",
            "explanation",
            "bloom_level",
            "difficulty",
        }


class TestDedupKeys:
    """Two questions differing only in punctuation or spacing are one question."""

    @pytest.mark.parametrize(
        "text",
        ["What is ETL?", "what   is etl", "WHAT IS ETL!!!", "  What is ETL  "],
    )
    def test_surface_differences_collapse_to_one_key(self, text: str) -> None:
        assert normalize_question_text(text) == "what is etl"

    def test_internal_whitespace_of_any_kind_collapses(self) -> None:
        assert normalize_question_text("  spaced\tout\n ") == "spaced out"

    def test_genuinely_different_questions_keep_different_keys(self) -> None:
        assert normalize_question_text("What is ETL?") != normalize_question_text("What is ELT?")
