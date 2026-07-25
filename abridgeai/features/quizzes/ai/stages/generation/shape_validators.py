"""Per-type shape validators for ``GeneratedQuestion``.

Each function validates the structural invariants of one
``question_type`` and raises ``ValueError`` on violation. The main
parser's ``_check_shape`` model_validator dispatches by type into one
of these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abridgeai.features.quizzes.ai.stages.generation.parsers import GeneratedQuestion


def validate_multiple_choice(question: GeneratedQuestion) -> None:
    """``multiple_choice`` — 4 options A-D; correct count depends on mode.

    Phase 7: when ``single_answer`` is False the question is multi-select and
    needs >= 2 correct options (a "multi-select" with one answer is just a
    single-answer question rendered with checkboxes, which misleads learners).
    Single-answer keeps the strict exactly-one rule.
    """
    if len(question.options) != 4:
        raise ValueError("multiple_choice requires exactly 4 options")
    if {opt.option_key for opt in question.options} != {"A", "B", "C", "D"}:
        raise ValueError("multiple_choice option keys must be A, B, C, D")
    n_correct = sum(1 for opt in question.options if opt.is_correct)
    if question.single_answer:
        if n_correct != 1:
            raise ValueError("multiple_choice requires exactly one correct option")
    elif n_correct < 2:
        raise ValueError(
            "multi-select multiple_choice requires at least two correct options"
        )
    elif n_correct == len(question.options):
        raise ValueError(
            "multi-select multiple_choice cannot mark every option correct"
        )


def validate_true_false(question: GeneratedQuestion) -> None:
    if len(question.options) != 2:
        raise ValueError("true_false requires exactly 2 options")
    if {opt.option_key for opt in question.options} != {"T", "F"}:
        raise ValueError("true_false option keys must be T, F")
    if sum(1 for opt in question.options if opt.is_correct) != 1:
        raise ValueError("true_false requires exactly one correct option")


def validate_fill_blank(question: GeneratedQuestion) -> None:
    blanks = question.original_generated_payload.get("correct_answer")
    if not isinstance(blanks, list) or not blanks:
        raise ValueError("fill_blank requires correct_answer as a non-empty list")
    if not all(isinstance(b, str) and b.strip() for b in blanks):
        raise ValueError("fill_blank correct_answer entries must be non-empty strings")
    # Word-bank invariants. Every correct answer MUST be present
    # verbatim (case-insensitive) so the learner can drag it into a
    # slot. The bank MUST also include at least one distractor — a
    # bank consisting only of correct answers gives the answer away.
    bank_texts = [opt.option_text for opt in question.options]
    if not bank_texts:
        raise ValueError("fill_blank requires a non-empty options word bank")
    bank_lower = {text.lower() for text in bank_texts}
    correct_lower = {b.lower() for b in blanks}
    missing = correct_lower - bank_lower
    if missing:
        raise ValueError(
            "fill_blank options word bank is missing correct answers: "
            f"{sorted(missing)}"
        )
    correct_count = sum(1 for opt in question.options if opt.is_correct)
    if correct_count != len(correct_lower):
        raise ValueError(
            "fill_blank options must mark every distinct correct answer "
            "as is_correct=True exactly once"
        )
    distractor_count = len(question.options) - correct_count
    if distractor_count < 1:
        raise ValueError(
            "fill_blank options word bank must include at least one "
            "distractor in addition to the correct answers"
        )


def validate_short_answer(question: GeneratedQuestion) -> None:
    answer = question.original_generated_payload.get("correct_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("short_answer requires correct_answer as a non-empty string")


# ---------------------------------------------------------------------------
# Phase 7 expanded types
# ---------------------------------------------------------------------------


def validate_numerical(question: GeneratedQuestion) -> None:
    """``numerical`` — expected value required, tolerance non-negative.

    The grader (``_grade_numerical``) accepts a submission when
    ``|submitted - numeric_answer| <= numeric_tolerance``, so a question with
    no ``numeric_answer`` can never be answered correctly. Reject it rather
    than persist a permanently-wrong question.
    """
    if question.options:
        raise ValueError("numerical questions do not support options")
    if question.numeric_answer is None:
        raise ValueError("numerical requires numeric_answer")
    tolerance = question.numeric_tolerance
    if tolerance is not None and tolerance < 0:
        raise ValueError("numerical numeric_tolerance must be >= 0")


def validate_matching(question: GeneratedQuestion) -> None:
    """``matching`` — >=2 unique left prompts, each with a non-empty right value.

    The student-facing projection shuffles the right column independently, so
    duplicate ``right`` values would make two different pairings both look
    correct while the grader accepts only the exact mapping. Reject duplicates
    on BOTH sides to keep the question unambiguously gradeable.
    """
    if question.options:
        raise ValueError("matching questions do not support options")
    pairs = question.match_pairs
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError("matching requires at least 2 pairs")
    if len(pairs) > 10:
        raise ValueError("matching supports at most 10 pairs")
    lefts: list[str] = []
    rights: list[str] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("matching pairs must be objects with left and right")
        left = str(pair.get("left", "")).strip()
        right = str(pair.get("right", "")).strip()
        if not left or not right:
            raise ValueError("matching pairs require non-empty left and right")
        lefts.append(left.lower())
        rights.append(right.lower())
    if len(set(lefts)) != len(lefts):
        raise ValueError("matching left prompts must be unique")
    if len(set(rights)) != len(rights):
        raise ValueError("matching right values must be unique (ambiguous otherwise)")


def validate_ordering(question: GeneratedQuestion) -> None:
    """``ordering`` — >=3 unique items in their correct order.

    The grader compares the student's sequence element-by-element against
    ``ordering_sequence``, so duplicate items make the correct answer
    ambiguous. Fewer than 3 items is a coin-flip, not an assessment.
    """
    if question.options:
        raise ValueError("ordering questions do not support options")
    items = question.ordering_sequence
    if not isinstance(items, list) or len(items) < 3:
        raise ValueError("ordering requires at least 3 items")
    if len(items) > 10:
        raise ValueError("ordering supports at most 10 items")
    cleaned = [str(item).strip() for item in items]
    if any(not item for item in cleaned):
        raise ValueError("ordering items must be non-empty")
    lowered = [item.lower() for item in cleaned]
    if len(set(lowered)) != len(lowered):
        raise ValueError("ordering items must be unique (ambiguous otherwise)")


__all__ = [
    "validate_fill_blank",
    "validate_matching",
    "validate_multiple_choice",
    "validate_numerical",
    "validate_ordering",
    "validate_short_answer",
    "validate_true_false",
]
