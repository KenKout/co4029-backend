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
