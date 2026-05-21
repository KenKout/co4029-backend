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

Option-list shaping for ``multiple_choice`` and ``true_false`` lives in
the sibling ``option_normalizers`` module.
"""

from __future__ import annotations

import string
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from abridgeai.features.quizzes.ai.stages.generation.option_normalizers import (
    coerce_fill_blank_answer,
    normalize_options,
)
from abridgeai.features.quizzes.ai.stages.generation.shape_validators import (
    validate_fill_blank,
    validate_multiple_choice,
    validate_short_answer,
    validate_true_false,
)

QuizQuestionType = Literal[
    "multiple_choice", "true_false", "short_answer", "fill_blank"
]
BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty = Literal["easy", "medium", "hard"]

# Legacy alias map. The pipeline used "mcq" historically; the DB CHECK
# always wanted "multiple_choice". Normalise at the parser boundary so
# every downstream consumer sees the DB vocabulary.
_LEGACY_TYPE_ALIASES: dict[str, str] = {
    "mcq": "multiple_choice",
    "fill_in_the_blank": "fill_blank",
    "true/false": "true_false",
    "tf": "true_false",
}

_VALID_TYPES = frozenset({"multiple_choice", "true_false", "short_answer", "fill_blank"})


def _normalize_question_type(raw: Any) -> str:  # noqa: ANN401 -- raw LLM JSON
    """Map legacy or LLM-drifted aliases onto DB vocabulary."""
    if not isinstance(raw, str):
        return "multiple_choice"
    cleaned = raw.strip().lower()
    return _LEGACY_TYPE_ALIASES.get(cleaned, cleaned)


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
    expected_response_ms: int = Field(default=60000, ge=0)
    source_refs_json: list[str] = Field(default_factory=list)
    original_generated_payload: dict[str, Any] = Field(default_factory=dict)
    options: list[GeneratedQuestionOption] = Field(default_factory=list)

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


def question_for_review(question: GeneratedQuestion | dict[str, Any]) -> dict[str, Any]:
    """Reshape one question into the validation-stage (T5.7) input dict.

    The validator needs to see the question_type so it can apply the
    right shape rules per type. For non-MCQ questions we surface the
    expected answer text/list so the validator can judge groundedness.
    """
    data = question.model_dump() if isinstance(question, GeneratedQuestion) else question
    options = data.get("options") or []
    options_dict: dict[str, str] = {}
    correct: str | None = None
    if isinstance(options, list):
        for opt in options:
            key = opt.get("option_key") if isinstance(opt, dict) else None
            if isinstance(key, str):
                options_dict[key] = opt.get("option_text", "")
                if opt.get("is_correct"):
                    correct = key
    payload = data.get("original_generated_payload") or {}
    if data.get("question_type") in {"short_answer", "fill_blank"}:
        correct_text = payload.get("correct_answer")
    else:
        correct_text = correct
    return {
        "prompt_text": data.get("prompt_text"),
        "question_type": data.get("question_type"),
        "options": options_dict,
        "correct_answer": correct_text,
        "explanation": data.get("explanation"),
        "bloom_level": data.get("bloom_level"),
        "difficulty": data.get("difficulty"),
    }


def normalize_question_text(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for dedup keys."""
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(translator).split())


def _extract_question_list(payload: Any) -> list[Any]:  # noqa: ANN401 -- raw LLM JSON
    if isinstance(payload, dict):
        questions = payload.get("questions")
        return questions if isinstance(questions, list) else []
    return payload if isinstance(payload, list) else []


def _prepare_question(entry: Any, *, default_position: int) -> dict[str, Any] | None:  # noqa: ANN401 -- raw LLM JSON
    if not isinstance(entry, dict):
        return None
    raw_question = entry.get("question") or entry.get("prompt_text")
    if not isinstance(raw_question, str) or not raw_question.strip():
        return None

    question_type = _normalize_question_type(entry.get("question_type"))
    if question_type not in _VALID_TYPES:
        return None
    correct_raw = entry.get("correct_answer") or entry.get("correct")
    options = normalize_options(entry.get("options"), correct_raw, question_type)

    canonical_payload = dict(entry)
    canonical_payload["question_type"] = question_type
    if question_type == "fill_blank":
        canonical_payload["correct_answer"] = coerce_fill_blank_answer(correct_raw)
    elif question_type == "short_answer":
        canonical_payload["correct_answer"] = (
            correct_raw.strip() if isinstance(correct_raw, str) else ""
        )

    source_refs = entry.get("source_refs") or entry.get("source_chunk_ids") or []
    if not isinstance(source_refs, list):
        source_refs = []

    return {
        "position": int(entry.get("position") or default_position),
        "question_type": question_type,
        "prompt_text": raw_question.strip(),
        "hint_text": entry.get("hint") or entry.get("hint_text"),
        "explanation": (entry.get("explanation") or "").strip() or "(no explanation)",
        "difficulty": entry.get("difficulty") or "medium",
        "bloom_level": entry.get("bloom_level") or "understand",
        "expected_response_ms": int(entry.get("expected_response_ms") or 60000),
        "source_refs_json": [str(ref) for ref in source_refs],
        "original_generated_payload": canonical_payload,
        "options": options,
    }


__all__ = [
    "GeneratedQuestion",
    "GeneratedQuestionOption",
    "QuizQuestionType",
    "normalize_question_text",
    "parse_generation_response",
    "question_for_review",
]
