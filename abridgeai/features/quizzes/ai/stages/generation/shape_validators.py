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
    if len(question.options) != 4:
        raise ValueError("multiple_choice requires exactly 4 options")
    if {opt.option_key for opt in question.options} != {"A", "B", "C", "D"}:
        raise ValueError("multiple_choice option keys must be A, B, C, D")
    if sum(1 for opt in question.options if opt.is_correct) != 1:
        raise ValueError("multiple_choice requires exactly one correct option")


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


__all__ = [
    "validate_fill_blank",
    "validate_multiple_choice",
    "validate_short_answer",
    "validate_true_false",
]
