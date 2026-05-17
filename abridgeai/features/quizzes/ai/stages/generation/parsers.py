"""Parsers and Pydantic schemas for the quiz GENERATION stage (T5.6).

Ports ``normalize_quiz_questions`` (mappers/quiz.py) and
``_question_for_review`` (pipelines/quiz_generation.py:1259-1277).
"""

from __future__ import annotations

import string
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

QuizQuestionType = Literal["mcq", "true_false", "short_answer", "fill_blank"]
BloomLevel = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty = Literal["easy", "medium", "hard"]


class GeneratedQuestionOption(BaseModel):
    option_key: str
    option_text: str
    is_correct: bool
    position: int


class GeneratedQuestion(BaseModel):
    """Normalised LLM-generated question consumed by T5.7 / T5.8."""

    position: int = Field(ge=1)
    question_type: QuizQuestionType = "mcq"
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
    def _check_mcq_shape(self) -> GeneratedQuestion:
        if self.question_type != "mcq":
            return self
        if len(self.options) != 4:
            raise ValueError("MCQ questions require exactly 4 options")
        if {opt.option_key for opt in self.options} != {"A", "B", "C", "D"}:
            raise ValueError("MCQ option keys must be exactly A, B, C, D")
        if sum(1 for opt in self.options if opt.is_correct) != 1:
            raise ValueError("MCQ questions require exactly one correct option")
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
    """Reshape one question into the validation-stage (T5.7) input dict."""
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
    return {
        "prompt_text": data.get("prompt_text"),
        "options": options_dict,
        "correct_answer": correct,
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

    question_type = entry.get("question_type") or "mcq"
    correct_raw = entry.get("correct_answer") or entry.get("correct")
    correct = correct_raw.strip().upper() if isinstance(correct_raw, str) else None
    options = _normalize_options(entry.get("options"), correct, question_type)

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
        "original_generated_payload": entry,
        "options": options,
    }


def _normalize_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: str | None,
    question_type: str,
) -> list[dict[str, Any]]:
    if question_type != "mcq":
        return []
    if isinstance(options_raw, dict):
        cleaned = {
            str(key).strip().upper(): str(value).strip()
            for key, value in options_raw.items()
            if isinstance(value, str)
        }
        return [
            {
                "option_key": key,
                "option_text": cleaned.get(key, ""),
                "is_correct": key == correct,
                "position": pos,
            }
            for pos, key in enumerate(["A", "B", "C", "D"], start=1)
            if key in cleaned
        ]
    if isinstance(options_raw, list):
        out: list[dict[str, Any]] = []
        for pos, item in enumerate(options_raw, start=1):
            if not isinstance(item, dict):
                continue
            key = item.get("option_key") or item.get("key")
            if not isinstance(key, str):
                continue
            out.append(
                {
                    "option_key": key.strip().upper(),
                    "option_text": str(item.get("option_text") or item.get("text") or "").strip(),
                    "is_correct": bool(item.get("is_correct")),
                    "position": int(item.get("position") or pos),
                }
            )
        return out
    return []


__all__ = [
    "GeneratedQuestion",
    "GeneratedQuestionOption",
    "normalize_question_text",
    "parse_generation_response",
    "question_for_review",
]
