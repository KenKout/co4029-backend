"""No-leak projection tests for Phase 7 matching/ordering (public schema).

SECURITY-CRITICAL: the raw answer keys (``match_pairs`` = [{left,right}] and
``ordering_sequence`` = correct order) MUST NEVER reach a learner. The public
``QuizQuestionPublic`` schema derives only safe projections:

  * ``match_prompts``  — the left column (safe to show).
  * ``match_choices``  — the right values, shuffled (pairing not implied).
  * ``ordering_items`` — items shuffled (correct sequence never revealed).

These tests assert the raw keys don't serialize AND the derived lists are
correct + not in answer order.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from abridgeai.features.quizzes.schemas.public import QuizQuestionPublic


def _base(**over):
    """A minimal ORM-like question stand-in for model_validate."""
    row = {
        "id": uuid.uuid4(),
        "quiz_id": uuid.uuid4(),
        "position": 1,
        "question_type": "matching",
        "prompt_text": "Match the pairs",
        "hint_text": None,
        "prompt_format": "plain",
        "hint_format": "plain",
        "single_answer": True,
        "options": [],
        "learning_outcome_id": None,
        "outcome_position": None,
        "match_pairs": None,
        "match_distractors": None,
        "ordering_sequence": None,
    }
    row.update(over)
    return SimpleNamespace(**row)


def test_matching_derives_prompts_and_shuffled_choices_no_leak():
    pairs = [
        {"left": "France", "right": "Paris"},
        {"left": "Japan", "right": "Tokyo"},
        {"left": "Egypt", "right": "Cairo"},
    ]
    pub = QuizQuestionPublic.model_validate(_base(question_type="matching", match_pairs=pairs))
    dumped = pub.model_dump()

    # Raw answer key must NOT serialize.
    assert "match_pairs" not in dumped

    # Left column preserved in order.
    assert pub.match_prompts == ["France", "Japan", "Egypt"]

    # Right values present as a set, but not necessarily in pair order.
    assert sorted(pub.match_choices) == ["Cairo", "Paris", "Tokyo"]

    # The dumped JSON string must not contain the pairing structure.
    import json

    blob = json.dumps(dumped, default=str)
    assert "right" not in blob
    assert "left" not in blob


def test_matching_choices_not_positionally_aligned_with_prompts():
    """Anti-alignment guard: on a multi-item list the shuffled choices must not
    sit in the same positions as their correct pairing (which would imply the
    answer by index)."""
    pairs = [
        {"left": "A", "right": "1"},
        {"left": "B", "right": "2"},
        {"left": "C", "right": "3"},
        {"left": "D", "right": "4"},
        {"left": "E", "right": "5"},
    ]
    pub = QuizQuestionPublic.model_validate(_base(question_type="matching", match_pairs=pairs))
    answer_aligned = ["1", "2", "3", "4", "5"]
    assert pub.match_choices != answer_aligned
    assert sorted(pub.match_choices) == answer_aligned


def test_ordering_derives_shuffled_items_not_in_answer_order():
    sequence = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    pub = QuizQuestionPublic.model_validate(
        _base(question_type="ordering", ordering_sequence=sequence)
    )
    dumped = pub.model_dump()

    assert "ordering_sequence" not in dumped
    # Same items, shuffled — the answer order must not be reproduced.
    assert sorted(pub.ordering_items) == sorted(sequence)
    assert pub.ordering_items != sequence


def test_plain_mcq_has_empty_derived_lists():
    pub = QuizQuestionPublic.model_validate(
        _base(question_type="multiple_choice", match_pairs=None, ordering_sequence=None)
    )
    assert pub.match_prompts == []
    assert pub.match_choices == []
    assert pub.ordering_items == []


def test_shuffle_is_stable_across_validations():
    """Same question id → same derived order (won't reshuffle mid-attempt)."""
    qid = uuid.uuid4()
    seq = ["a", "b", "c", "d", "e", "f"]
    a = QuizQuestionPublic.model_validate(
        _base(id=qid, question_type="ordering", ordering_sequence=seq)
    )
    b = QuizQuestionPublic.model_validate(
        _base(id=qid, question_type="ordering", ordering_sequence=seq)
    )
    assert a.ordering_items == b.ordering_items


def test_matching_distractors_join_choice_pool_without_becoming_prompts():
    """Distractors enlarge the shuffled choice pool but never appear as a
    prompt and never serialize raw — they're just extra wrong choices."""
    pairs = [
        {"left": "France", "right": "Paris"},
        {"left": "Japan", "right": "Tokyo"},
    ]
    pub = QuizQuestionPublic.model_validate(
        _base(
            question_type="matching",
            match_pairs=pairs,
            match_distractors=["Berlin", "Madrid"],
        )
    )
    dumped = pub.model_dump()

    # Raw answer key AND raw distractor list must NOT serialize.
    assert "match_pairs" not in dumped
    assert "match_distractors" not in dumped

    # Prompts are the left column only — distractors never leak in as prompts.
    assert pub.match_prompts == ["France", "Japan"]

    # The choice pool is the correct rights PLUS the distractors, shuffled.
    assert sorted(pub.match_choices) == ["Berlin", "Madrid", "Paris", "Tokyo"]


def test_matching_distractor_colliding_with_answer_is_dropped():
    """A distractor equal to a correct right value would make that value both
    right and wrong; the projection drops the duplicate (case-insensitive)."""
    pairs = [{"left": "France", "right": "Paris"}]
    pub = QuizQuestionPublic.model_validate(
        _base(
            question_type="matching",
            match_pairs=pairs,
            match_distractors=["paris", "London"],
        )
    )
    # "paris" collides with the answer "Paris" → dropped; only "London" survives.
    assert sorted(pub.match_choices) == ["London", "Paris"]


def test_matching_no_distractors_is_classic_one_to_one():
    """Empty / absent distractors → choice pool is exactly the answer set."""
    pairs = [
        {"left": "A", "right": "1"},
        {"left": "B", "right": "2"},
    ]
    pub = QuizQuestionPublic.model_validate(
        _base(question_type="matching", match_pairs=pairs, match_distractors=[])
    )
    assert sorted(pub.match_choices) == ["1", "2"]


def test_fill_blank_derives_shuffled_bank_from_correct_answer_no_leak():
    """Teacher-created fill_blank (no option rows) exposes the answer words as
    a shuffled word bank; the raw ``correct_answer`` never serializes."""
    pub = QuizQuestionPublic.model_validate(
        _base(
            question_type="fill_blank",
            options=[],
            original_generated_payload={"correct_answer": ["123", "456"]},
        )
    )
    dumped = pub.model_dump()

    # The answer key must NOT serialize — only the shuffled word bank may.
    assert "original_generated_payload" not in dumped
    assert "correct_answer" not in dumped
    assert sorted(pub.fill_blank_choices) == ["123", "456"]


def test_fill_blank_uses_option_bank_with_distractors_when_present():
    """AI-generated fill_blank carries the full word bank as option rows; the
    projection surfaces every option text (correct + distractors), still with
    no is_correct leak."""
    options = [
        SimpleNamespace(id=uuid.uuid4(), option_key="O01", option_text="alpha", position=1),
        SimpleNamespace(id=uuid.uuid4(), option_key="O02", option_text="beta", position=2),
        SimpleNamespace(id=uuid.uuid4(), option_key="O03", option_text="gamma", position=3),
    ]
    pub = QuizQuestionPublic.model_validate(
        _base(
            question_type="fill_blank",
            options=options,
            original_generated_payload={"correct_answer": ["alpha", "beta"]},
        )
    )
    assert sorted(pub.fill_blank_choices) == ["alpha", "beta", "gamma"]
    for option in pub.options:
        assert "is_correct" not in option.model_dump()


def test_fill_blank_shuffle_is_stable_and_not_in_answer_order():
    """Same question id → same bank order; the bank is shuffled so the answer
    order is not implied by position."""
    qid = uuid.uuid4()
    answers = ["w1", "w2", "w3", "w4", "w5"]
    a = QuizQuestionPublic.model_validate(
        _base(
            id=qid,
            question_type="fill_blank",
            options=[],
            original_generated_payload={"correct_answer": answers},
        )
    )
    b = QuizQuestionPublic.model_validate(
        _base(
            id=qid,
            question_type="fill_blank",
            options=[],
            original_generated_payload={"correct_answer": answers},
        )
    )
    assert a.fill_blank_choices == b.fill_blank_choices
    assert sorted(a.fill_blank_choices) == sorted(answers)
    assert a.fill_blank_choices != answers
