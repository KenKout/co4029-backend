"""All question-type vocabularies must stay in lockstep.

This guards the bug where a teacher selected ``ordering`` + ``matching`` on the
generate page and received 10 ``multiple_choice`` questions instead.

Root cause: the IDEATION parser kept its own ``_VALID_TYPES`` frozenset that had
never been extended past the original four types. Its ``question_type``
validator silently rewrites anything outside that set to ``multiple_choice``:

    return canonical if canonical in _VALID_TYPES else "multiple_choice"

So the LLM correctly returned ``matching``/``ordering`` templates, the parser
rewrote every one to ``multiple_choice``, and the generation stage dutifully
produced MCQs. Nothing errored — the request "succeeded" with the wrong types.

The pipeline has several independent copies of this vocabulary. Any one of them
falling behind reintroduces a silent-rewrite or silent-drop bug, so they are
pinned equal here rather than trusted to be updated together.
"""

from __future__ import annotations

from typing import get_args

from abridgeai.features.quizzes.ai.stages.generation.coercions import (
    _VALID_TYPES as GENERATION_VALID_TYPES,
)
from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    QuizQuestionType,
)
from abridgeai.features.quizzes.ai.stages.ideation.parsers import (
    _VALID_TYPES as IDEATION_VALID_TYPES,
)
from abridgeai.features.quizzes.schemas.run import QuestionType

EXPECTED_GENERATABLE = {
    "multiple_choice",
    "true_false",
    "short_answer",
    "fill_blank",
    "numerical",
    "matching",
    "ordering",
}


def test_request_schema_vocabulary() -> None:
    assert set(get_args(QuestionType)) == EXPECTED_GENERATABLE


def test_generation_parser_vocabulary() -> None:
    assert set(get_args(QuizQuestionType)) == EXPECTED_GENERATABLE
    assert GENERATION_VALID_TYPES == EXPECTED_GENERATABLE


def test_ideation_parser_vocabulary_matches_generation() -> None:
    """The regression: ideation lagged behind and silently rewrote to MCQ."""
    assert IDEATION_VALID_TYPES == EXPECTED_GENERATABLE


def test_all_three_vocabularies_are_identical() -> None:
    assert (
        set(get_args(QuestionType))
        == set(get_args(QuizQuestionType))
        == GENERATION_VALID_TYPES
        == IDEATION_VALID_TYPES
    )


def test_ideation_preserves_new_types_end_to_end() -> None:
    """Parse a realistic ideation response and assert types survive."""
    from abridgeai.features.quizzes.ai.stages.ideation.parsers import (
        parse_ideation_response,
    )

    payload = {
        "templates": [
            {
                "position": index,
                "section_id": "sec_a",
                "topic": f"topic {index}",
                "question_type": qtype,
                "bloom_level": "understand",
                "difficulty": "medium",
                "source_chunk_ids": ["11111111-1111-1111-1111-111111111111"],
                "rationale": "why",
            }
            for index, qtype in enumerate(sorted(EXPECTED_GENERATABLE), start=1)
        ]
    }
    templates = parse_ideation_response(payload)
    assert {t.question_type for t in templates} == EXPECTED_GENERATABLE, (
        "ideation must preserve every generatable type; a missing entry in "
        "_VALID_TYPES silently rewrites it to multiple_choice"
    )


def test_ideation_still_rejects_genuinely_unknown_types() -> None:
    """The fallback itself is correct — only the allow-list was stale."""
    from abridgeai.features.quizzes.ai.stages.ideation.parsers import (
        parse_ideation_response,
    )

    payload = {
        "templates": [
            {
                "position": 1,
                "section_id": "sec_a",
                "topic": "t",
                "question_type": "essay_with_rubric",
                "bloom_level": "understand",
                "difficulty": "medium",
                "source_chunk_ids": [],
                "rationale": "r",
            }
        ]
    }
    templates = parse_ideation_response(payload)
    assert templates[0].question_type == "multiple_choice"


def test_orm_check_constraint_covers_every_generatable_type() -> None:
    """A generatable type rejected by the DB CHECK would fail at flush time."""
    from abridgeai.features.quizzes.models import QuizQuestion

    constraint = next(
        c
        for c in QuizQuestion.__table__.constraints
        if getattr(c, "name", None) == "ck_quiz_questions_question_type"
    )
    sqltext = str(constraint.sqltext)
    for qtype in EXPECTED_GENERATABLE:
        assert f"'{qtype}'" in sqltext, f"{qtype} missing from the CHECK constraint"
