"""Parsers and Pydantic schemas for the quiz GENERATION stage (T5.6).

Ports ``normalize_quiz_questions`` (mappers/quiz.py) and
``_question_for_review`` (pipelines/quiz_generation.py:1259-1277).

**Type vocabulary** matches the DB CHECK constraint exactly:
``multiple_choice``, ``true_false``, ``short_answer``, ``fill_blank``.
The legacy ``"mcq"`` alias is normalised to ``"multiple_choice"`` at the
parser boundary so old prompts/payloads still parse, but every consumer
downstream of this module sees the DB vocabulary.

**Type shape rules** (enforced by ``GeneratedQuestion._check_shape``):

* ``multiple_choice`` — exactly 4 options A-D, exactly 1 marked correct.
* ``true_false`` — exactly 2 options keyed ``T``/``F``, exactly 1
  marked correct. The parser auto-generates these options from the
  LLM's ``correct_answer`` ("True"/"False") when the LLM omits them
  (the prompt requests them but small models drift).
* ``short_answer`` — no options. ``correct_answer`` is a free-text
  string carried in ``original_generated_payload`` for grading.
* ``fill_blank`` — no options. ``correct_answer`` is a list of strings
  (one per blank, in order) carried in ``original_generated_payload``.
  The grader matches the student's drag-drop slots positionally.

**Module layout** — this file grew past the 250-LOC god-file budget, so its
concerns now live in sibling modules and this file is the hub:

* ``coercions`` — the type/format ``Literal`` vocabulary + defensive
  raw-LLM-JSON coercion helpers.
* ``option_normalizers`` — option-list shaping for ``multiple_choice`` /
  ``true_false``.
* ``shape_validators`` — per-type shape enforcement invoked by
  ``GeneratedQuestion._check_shape``.
* ``prepare`` — turn one raw LLM entry into the validated-schema input dict.
* ``review`` — reshape a normalised question into the validation-stage input.

This module keeps the two Pydantic schemas + ``parse_generation_response`` (they
bind the pieces together) and re-exports the public names below, so existing
imports of ``...generation.parsers`` keep working unchanged.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from abridgeai.features.quizzes.ai.stages.generation.coercions import (
    BloomLevel,
    Difficulty,
    QuizQuestionType,
    RichFormat,
    _coerce_decimal,
    _coerce_match_pairs,
    _coerce_ordering_sequence,
    _normalize_format,
    _normalize_question_type,
)
from abridgeai.features.quizzes.ai.stages.generation.prepare import (
    _coerce_single_answer,
    _extract_question_list,
    _prepare_question,
)
from abridgeai.features.quizzes.ai.stages.generation.review import (
    _render_answer_for_review,
    normalize_question_text,
    question_for_review,
)
from abridgeai.features.quizzes.ai.stages.generation.shape_validators import (
    validate_fill_blank,
    validate_matching,
    validate_multiple_choice,
    validate_numerical,
    validate_ordering,
    validate_short_answer,
    validate_true_false,
)


class GeneratedQuestionOption(BaseModel):
    option_key: str
    option_text: str
    is_correct: bool
    position: int


class GeneratedQuestion(BaseModel):
    """Normalised LLM-generated question consumed by T5.7 / T5.8."""

    position: int = Field(ge=1)
    question_type: QuizQuestionType = "multiple_choice"
    prompt_text: str = Field(min_length=1)
    hint_text: str | None = None
    explanation: str = Field(min_length=1)
    difficulty: Difficulty = "medium"
    bloom_level: BloomLevel = "understand"
    # Phase 3 rich-content discriminators. Default ``plain`` so existing
    # prompts/behaviour are unchanged; the persistence stage sanitizes each
    # field according to its own format before writing.
    prompt_format: RichFormat = "plain"
    hint_format: RichFormat = "plain"
    explanation_format: RichFormat = "plain"
    expected_response_ms: int = Field(default=60000, ge=0)
    source_refs_json: list[str] = Field(default_factory=list)
    original_generated_payload: dict[str, Any] = Field(default_factory=dict)
    options: list[GeneratedQuestionOption] = Field(default_factory=list)

    # --- Phase 7 type-specific answer fields ------------------------------
    # Only meaningful for their own type; every other type leaves them unset.
    # These map 1:1 onto the ``quiz_questions`` columns the grader reads.
    single_answer: bool = True
    """``multiple_choice`` only. False → multi-select (>=1 correct option)."""

    numeric_answer: Decimal | None = None
    """``numerical`` only. The expected value."""

    numeric_tolerance: Decimal | None = None
    """``numerical`` only. Accepted absolute deviation (``>= 0``)."""

    match_pairs: list[dict[str, str]] | None = None
    """``matching`` only. ``[{"left": .., "right": ..}]`` — the answer key."""

    match_distractors: list[str] | None = None
    """``matching`` only. Extra right-side values with NO left partner — they
    enlarge the learner's shuffled choice pool but are never a correct answer."""

    ordering_sequence: list[str] | None = None
    """``ordering`` only. Items in their CORRECT order (shuffled for students)."""

    @model_validator(mode="after")
    def _check_shape(self) -> GeneratedQuestion:
        if self.question_type == "multiple_choice":
            validate_multiple_choice(self)
        elif self.question_type == "true_false":
            validate_true_false(self)
        elif self.question_type == "fill_blank":
            validate_fill_blank(self)
        elif self.question_type == "short_answer":
            validate_short_answer(self)
        elif self.question_type == "numerical":
            validate_numerical(self)
        elif self.question_type == "matching":
            validate_matching(self)
        elif self.question_type == "ordering":
            validate_ordering(self)
        return self


def parse_generation_response(payload: Any) -> list[GeneratedQuestion]:  # noqa: ANN401 -- raw LLM JSON
    """Validate raw LLM JSON into normalised questions; drop bad entries."""
    raw = _extract_question_list(payload)
    out: list[GeneratedQuestion] = []
    for index, entry in enumerate(raw, start=1):
        prepared = _prepare_question(entry, default_position=index)
        if prepared is None:
            continue
        try:
            out.append(GeneratedQuestion.model_validate(prepared))
        except (ValidationError, ValueError, TypeError):
            continue
    return out


__all__ = [
    # Schemas + entry point owned by this module.
    "GeneratedQuestion",
    "GeneratedQuestionOption",
    "parse_generation_response",
    # Re-exported from coercions (type vocabulary + coercers) so existing
    # ``...generation.parsers`` imports keep resolving unchanged.
    "QuizQuestionType",
    "_coerce_decimal",
    "_coerce_match_pairs",
    "_coerce_ordering_sequence",
    "_normalize_format",
    "_normalize_question_type",
    # Re-exported from prepare / review.
    "_coerce_single_answer",
    "_extract_question_list",
    "_prepare_question",
    "_render_answer_for_review",
    "normalize_question_text",
    "question_for_review",
]
