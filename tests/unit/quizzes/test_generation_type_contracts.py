"""Pin the generation prompt's per-type contracts to the parser's behaviour.

The system prompt (``ai/stages/generation/prompts/system.j2``) states an explicit
contract per ``question_type``. If the parser and the prompt disagree, the LLM
follows the prompt and the parser silently ``continue``s past the result — a
"successful" run that generates fewer questions than requested, with no error.

These tests assert:

1. The request schema's ``QuestionType`` and the parser's ``QuizQuestionType``
   expose the SAME vocabulary (the ``code`` mismatch regression).
2. Each type's documented happy-path shape is accepted.
3. Each violation the prompt forbids is either rejected outright or safely
   normalised — never accepted with a wrong answer key.
"""

from __future__ import annotations

from typing import Any, get_args

from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    QuizQuestionType,
    parse_generation_response,
)
from abridgeai.features.quizzes.schemas.run import QuestionType


def _parse_one(entry: dict[str, Any]) -> Any:
    out = parse_generation_response({"questions": [entry]})
    return out[0] if out else None


# ---------------------------------------------------------------------------
# 1. vocabulary parity — the ``code`` regression
# ---------------------------------------------------------------------------


def test_request_and_parser_vocabularies_match() -> None:
    """A type accepted at the HTTP boundary must be producible by the parser.

    ``code`` used to be in the request Literal but not the parser's, so asking
    for it yielded a run that silently produced nothing for those slots.
    """
    assert set(get_args(QuestionType)) == set(get_args(QuizQuestionType))


def test_code_is_not_requestable_for_generation() -> None:
    assert "code" not in get_args(QuestionType)
    assert "code" not in get_args(QuizQuestionType)


# ---------------------------------------------------------------------------
# 2. documented happy paths are accepted
# ---------------------------------------------------------------------------


def test_multiple_choice_contract_accepted() -> None:
    q = _parse_one(
        {
            "question_type": "multiple_choice",
            "question": "Which property means a warehouse keeps history?",
            "options": {"A": "time-variant", "B": "volatile", "C": "normalized", "D": "atomic"},
            "correct_answer": "A",
            "explanation": "It stores historical snapshots.",
        }
    )
    assert q is not None
    assert len(q.options) == 4
    assert sum(1 for o in q.options if o.is_correct) == 1
    # correct_answer is the LETTER, so option A must be the correct one
    correct = next(o for o in q.options if o.is_correct)
    assert correct.option_key == "A"


def test_true_false_contract_accepted_without_options() -> None:
    """Prompt says: emit NO options; the system synthesizes the T/F pair."""
    q = _parse_one(
        {
            "question_type": "true_false",
            "question": "A data warehouse is subject-oriented.",
            "correct_answer": "True",
            "explanation": "Subject orientation is one of Inmon's four properties.",
        }
    )
    assert q is not None
    assert {o.option_key for o in q.options} == {"T", "F"}
    assert next(o for o in q.options if o.is_correct).option_key == "T"


def test_short_answer_contract_accepted() -> None:
    q = _parse_one(
        {
            "question_type": "short_answer",
            "question": "What property describes storing historical snapshots?",
            "correct_answer": "time-variant",
            "explanation": "History is retained rather than overwritten.",
        }
    )
    assert q is not None
    assert q.options == []
    assert q.original_generated_payload["correct_answer"] == "time-variant"


def test_fill_blank_contract_accepted() -> None:
    q = _parse_one(
        {
            "question_type": "fill_blank",
            "question": "A warehouse is ___ and ___.",
            "correct_answer": ["integrated", "non-volatile"],
            "options": ["integrated", "non-volatile", "transactional", "normalized"],
            "explanation": "Two of Inmon's properties.",
        }
    )
    assert q is not None
    bank = {o.option_text for o in q.options}
    # every correct answer present verbatim
    assert {"integrated", "non-volatile"} <= bank
    # at least one distractor beyond the correct answers
    assert len(q.options) > 2


# ---------------------------------------------------------------------------
# 3. forbidden shapes are rejected (or safely normalised)
# ---------------------------------------------------------------------------


def test_multiple_choice_rejects_wrong_option_count() -> None:
    assert (
        _parse_one(
            {
                "question_type": "multiple_choice",
                "question": "q",
                "options": {"A": "a", "B": "b", "C": "c"},
                "correct_answer": "A",
                "explanation": "e",
            }
        )
        is None
    )


def test_multiple_choice_rejects_answer_given_as_text() -> None:
    """Prompt is explicit that ``correct_answer`` is the LETTER.

    Text instead of a letter marks zero options correct, which the shape
    validator rejects — so the question is dropped rather than persisted with
    no correct answer.
    """
    assert (
        _parse_one(
            {
                "question_type": "multiple_choice",
                "question": "q",
                "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                "correct_answer": "alpha",
                "explanation": "e",
            }
        )
        is None
    )


def test_short_answer_rejects_empty_answer() -> None:
    assert (
        _parse_one(
            {
                "question_type": "short_answer",
                "question": "q",
                "correct_answer": "",
                "explanation": "e",
            }
        )
        is None
    )


def test_fill_blank_rejects_bank_without_distractors() -> None:
    assert (
        _parse_one(
            {
                "question_type": "fill_blank",
                "question": "x is ___",
                "correct_answer": ["alpha"],
                "options": ["alpha"],
                "explanation": "e",
            }
        )
        is None
    )


def test_fill_blank_self_heals_bank_missing_a_correct_answer() -> None:
    """The normalizer prepends a missing correct answer rather than dropping.

    Documented deliberately: the learner must be able to reach the right answer,
    so a bank that omits it is repaired, not rejected (the distractor rule still
    applies afterwards).
    """
    q = _parse_one(
        {
            "question_type": "fill_blank",
            "question": "x is ___",
            "correct_answer": ["alpha"],
            "options": ["beta", "gamma"],
            "explanation": "e",
        }
    )
    assert q is not None
    texts = [o.option_text for o in q.options]
    assert "alpha" in texts
    assert next(o for o in q.options if o.option_text == "alpha").is_correct is True


def test_true_false_discards_llm_options_and_synthesizes_pair() -> None:
    """Prompt forbids options for true_false; stray ones must not corrupt it."""
    q = _parse_one(
        {
            "question_type": "true_false",
            "question": "q",
            "options": {"A": "x", "B": "y"},
            "correct_answer": "False",
            "explanation": "e",
        }
    )
    assert q is not None
    assert {o.option_key for o in q.options} == {"T", "F"}
    assert next(o for o in q.options if o.is_correct).option_key == "F"
