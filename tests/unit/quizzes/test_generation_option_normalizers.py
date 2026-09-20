"""Option normalisers for the quiz generation stage.

These turn untrusted LLM JSON into option rows and their ``is_correct``
flags — the answer key. A normaliser that drops, mis-marks or reorders an
entry produces a question that is wrongly keyed or unanswerable, and the
learner meets it as an ordinary quiz item.

The shape validators run afterwards and reject the malformed cases these
tests describe, so a defect here costs a generated question rather than a
student's mark. That backstop is a reason to keep the rules honest, not a
reason to leave them unpinned: it only catches what it knows to look for.

Pure functions over plain dicts — no database, no gateway, no fixtures.
"""

from __future__ import annotations

import random

import pytest

from abridgeai.features.quizzes.ai.stages.generation.option_normalizers import (
    _coerce_correct_keys,
    _coerce_fill_blank_option,
    _normalize_fill_blank_options,
    coerce_fill_blank_answer,
    normalize_options,
    randomize_mcq_options,
)


def _texts(rows: list[dict[str, object]]) -> list[object]:
    return [row["option_text"] for row in rows]


def _correct(rows: list[dict[str, object]]) -> list[object]:
    return [row["option_text"] for row in rows if row["is_correct"]]


class TestTheWordBankAlwaysContainsTheAnswer:
    """``fill_blank`` hands the learner a bank to drag from.

    A bank without the answer in it cannot be completed, however well the
    rest of the question reads.
    """

    def test_an_answer_the_model_omitted_is_added_to_the_bank(self) -> None:
        rows = _normalize_fill_blank_options(["cat", "bird"], ["dog"])
        assert "dog" in _texts(rows)
        assert _correct(rows) == ["dog"]

    def test_truncation_keeps_the_answer(self) -> None:
        """A bank past the cap must not lose the entry that makes it solvable.

        Slicing the bank blind drops a correct entry sitting beyond the cap:
        the prepend above only covers an answer the model omitted, not one it
        supplied too late in a long list.
        """
        rows = _normalize_fill_blank_options([f"w{i}" for i in range(120)], ["w119"])
        assert len(rows) == 99
        assert _correct(rows) == ["w119"]

    def test_truncation_keeps_several_answers(self) -> None:
        rows = _normalize_fill_blank_options([f"w{i}" for i in range(150)], ["w3", "w100", "w149"])
        assert len(rows) == 99
        assert sorted(_correct(rows)) == ["w100", "w149", "w3"]

    def test_truncation_does_not_hoist_the_answer_to_the_front(self) -> None:
        """Position must not give the exercise away.

        Keeping correct entries by moving them to the head of the bank would
        satisfy the rule above and hand the learner the answer for free.
        """
        rows = _normalize_fill_blank_options([f"w{i}" for i in range(120)], ["w119"])
        answer_position = next(row["position"] for row in rows if row["is_correct"])
        assert answer_position != 1

    def test_bank_order_is_otherwise_preserved(self) -> None:
        rows = _normalize_fill_blank_options(["cat", "dog", "bird"], ["dog"])
        assert _texts(rows) == ["cat", "dog", "bird"]

    def test_duplicate_entries_are_folded_case_insensitively(self) -> None:
        """A bank listing the same word twice offers a phantom choice."""
        rows = _normalize_fill_blank_options(["dog", "DOG", "cat"], ["dog"])
        assert _texts(rows) == ["dog", "cat"]

    def test_the_answer_matches_regardless_of_case(self) -> None:
        rows = _normalize_fill_blank_options(["Dog", "cat"], ["dog"])
        assert _correct(rows) == ["Dog"], "the bank's own casing is preserved"

    def test_keys_are_canonical_and_fit_the_column(self) -> None:
        rows = _normalize_fill_blank_options(["a", "b"], ["a"])
        assert [row["option_key"] for row in rows] == ["O01", "O02"]
        assert all(len(str(row["option_key"])) <= 5 for row in rows)

    def test_nothing_parseable_yields_no_options(self) -> None:
        assert _normalize_fill_blank_options(None, None) == []


class TestAnswerCoercion:
    """The model is inconsistent about how it spells a list of blanks."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (["a", "b"], ["a", "b"]),
            ("a; b", ["a", "b"]),
            ("a, b", ["a", "b"]),
            ("single", ["single"]),
            ("  padded  ", ["padded"]),
            ("", []),
            (None, []),
            ([" a ", "", "b"], ["a", "b"]),
        ],
    )
    def test_blank_lists_are_accepted_in_every_spelling(
        self, raw: object, expected: list[str]
    ) -> None:
        assert coerce_fill_blank_answer(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("word", "word"),
            ({"option_text": "word"}, "word"),
            ({"text": "word"}, "word"),
            ({"value": "word"}, "word"),
            ({"unexpected": "word"}, ""),
            (42, ""),
        ],
    )
    def test_bank_entries_are_accepted_bare_or_wrapped(
        self, raw: object, expected: str
    ) -> None:
        assert _coerce_fill_blank_option(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A", {"A"}),
            ("a, c", {"A", "C"}),
            (["A", "b"], {"A", "B"}),
            (None, set()),
        ],
    )
    def test_correct_keys_are_upper_cased_and_split(
        self, raw: object, expected: set[str]
    ) -> None:
        assert _coerce_correct_keys(raw) == expected


class TestMultipleChoice:
    def test_mcq_choices_are_rekeyed_after_a_shuffle(self) -> None:
        rows = normalize_options(
            {"A": "correct", "B": "wrong one", "C": "wrong two", "D": "wrong three"},
            "A",
            "multiple_choice",
        )
        shuffled, correct_keys = randomize_mcq_options(rows, rng=random.Random(0))  # noqa: S311

        assert [row["option_key"] for row in shuffled] == ["A", "B", "C", "D"]
        assert len(correct_keys) == 1
        correct = next(row for row in shuffled if row["is_correct"])
        assert correct["option_text"] == "correct"
        assert correct["option_key"] == correct_keys[0]

    def test_the_dict_form_marks_correctness_from_the_answer_key(self) -> None:
        rows = normalize_options({"A": "ay", "B": "bee"}, "A", "multiple_choice")
        assert _correct(rows) == ["ay"]
        assert [row["option_key"] for row in rows] == ["A", "B"]

    def test_positions_follow_the_canonical_letter_order(self) -> None:
        """A gap in the letters must not renumber the ones that remain."""
        rows = normalize_options({"A": "ay", "B": "bee", "D": "dee"}, "A", "multiple_choice")
        assert [(row["option_key"], row["position"]) for row in rows] == [
            ("A", 1),
            ("B", 2),
            ("D", 4),
        ]

    def test_keys_are_upper_cased_and_ordered(self) -> None:
        rows = normalize_options({"c": "cee", "a": "ay"}, "A", "multiple_choice")
        assert [row["option_key"] for row in rows] == ["A", "C"]

    def test_the_list_form_takes_correctness_from_each_item(self) -> None:
        """The two input shapes read correctness from different places.

        A dict of letters is scored against ``correct_answer``; a list of
        option objects is taken at its word, each item's own ``is_correct``
        deciding. An item list carrying no flag therefore yields no correct
        option, and the multiple-choice validator refuses the question.
        """
        rows = normalize_options(
            [{"option_key": "A", "option_text": "ay"}, {"key": "B", "text": "bee"}],
            "A",
            "multiple_choice",
        )
        assert _texts(rows) == ["ay", "bee"]
        assert _correct(rows) == []

        flagged = normalize_options(
            [{"option_key": "A", "option_text": "ay", "is_correct": True}],
            None,
            "multiple_choice",
        )
        assert _correct(flagged) == ["ay"]

    def test_bare_strings_are_discarded(self) -> None:
        """A list of plain strings carries no key, so there is nothing to score."""
        assert normalize_options(["ay", "bee"], "A", "multiple_choice") == []


class TestTrueFalse:
    """The pair is synthesised, never taken from the model's option array."""

    @pytest.mark.parametrize(
        ("correct", "expected"),
        [(True, "True"), ("true", "True"), (False, "False"), ("false", "False")],
    )
    def test_the_pair_is_built_from_the_answer(self, correct: object, expected: str) -> None:
        rows = normalize_options(None, correct, "true_false")
        assert _texts(rows) == ["True", "False"]
        assert _correct(rows) == [expected]

    def test_an_unreadable_answer_marks_neither(self) -> None:
        """Guessing here would invent an answer the model never gave.

        Both options come back unmarked and the true/false validator refuses
        the question, which is the correct outcome: a coin flip would be
        recorded as the teacher's answer key.
        """
        rows = normalize_options(None, "maybe", "true_false")
        assert _texts(rows) == ["True", "False"]
        assert _correct(rows) == []


class TestTypesThatCarryNoOptions:
    @pytest.mark.parametrize(
        "question_type", ["short_answer", "numerical", "matching", "ordering", "unknown"]
    )
    def test_no_option_rows_are_produced(self, question_type: str) -> None:
        """These answer on the question's own columns.

        Returning nothing also discards options from a model that wrongly
        emitted them, rather than persisting rows the type has no use for.
        """
        assert normalize_options(["x"], "x", question_type) == []
