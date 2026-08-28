"""Parsers for the interview GENERATION stage (T6.5).

Validates raw LLM JSON into :class:`InterviewQuestionDraft` instances.
Mirrors the shape used by the quiz generation parser
(``features/quizzes/ai/stages/generation/parsers.py``) but produces
plain ``@dataclass`` drafts instead of Pydantic models — interview
questions go through a separate VALIDATION stage (T6.6) before
persistence, so a runtime-cheap dataclass is sufficient here.

The parser drops malformed entries instead of raising; an empty list
signals "every question failed validation" to the caller.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from abridgeai.features.interviews.ai.stages.generation.draft import (
    _VALID_DIFFICULTIES,
    _VALID_TYPES,
    InterviewDifficulty,
    InterviewQuestionDraft,
    InterviewQuestionType,
)
from abridgeai.features.interviews.ai.stages.generation.grouping import (
    assign_variant_group_ids,
    coerce_logical_question_index,
)


def parse_generation_response(
    payload: Any,  # noqa: ANN401 -- raw LLM JSON
    *,
    max_questions: int | None = None,
    require_logical_question_index: bool = False,
) -> list[InterviewQuestionDraft]:  # noqa: ANN401 -- raw LLM JSON
    """Validate raw LLM JSON into drafts; drop bad entries silently.

    Parameters
    ----------
    payload
        Raw LLM JSON (dict or list).
    max_questions
        Cap the returned list at this length. LLMs sometimes overshoot
        the requested ``question_count``; we keep the first N valid
        entries so the teacher sees exactly the number they asked for.
    require_logical_question_index
        All-angle generation requires an integer logical-group ordinal on each
        row. Other strategies ignore it and leave durable group IDs unset.
    """
    raw = _extract_question_list(payload)
    out: list[InterviewQuestionDraft] = []
    for entry in raw:
        draft = _prepare_question(entry)
        if draft is not None:
            out.append(draft)
    grouped = assign_variant_group_ids(
        out,
        require_index=require_logical_question_index,
    )
    if max_questions is None:
        return grouped
    return grouped[:max_questions]


def _extract_question_list(payload: Any) -> list[Any]:  # noqa: ANN401 -- raw LLM JSON
    if isinstance(payload, dict):
        questions = payload.get("questions")
        return questions if isinstance(questions, list) else []
    return payload if isinstance(payload, list) else []


def _prepare_question(entry: Any) -> InterviewQuestionDraft | None:  # noqa: ANN401 -- raw LLM JSON
    if not isinstance(entry, dict):
        return None

    raw_text = entry.get("prompt_text") or entry.get("question") or entry.get("stem")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    raw_type = (entry.get("question_type") or "technical").strip().lower()
    if raw_type == "behavioural":  # accept BrEng spelling from the LLM
        raw_type = "behavioral"
    if raw_type not in _VALID_TYPES:
        return None

    raw_difficulty = (entry.get("difficulty") or "medium").strip().lower()
    if raw_difficulty not in _VALID_DIFFICULTIES:
        return None

    expected_depth = _coerce_depth(entry.get("expected_depth"))
    linked_outcome_id = _coerce_uuid(entry.get("linked_outcome_id"))
    logical_question_index = coerce_logical_question_index(
        entry.get("logical_question_index")
    )
    source_refs = _coerce_uuid_list(entry.get("source_refs"))
    rationale = entry.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""

    model_answer = entry.get("model_answer")
    if not isinstance(model_answer, str):
        model_answer = ""

    return InterviewQuestionDraft(
        question_type=cast("InterviewQuestionType", raw_type),
        prompt_text=raw_text.strip(),
        difficulty=cast("InterviewDifficulty", raw_difficulty),
        expected_depth=expected_depth,
        linked_outcome_id=linked_outcome_id,
        logical_question_index=logical_question_index,
        source_refs=source_refs,
        rationale=rationale.strip(),
        model_answer=model_answer.strip(),
    )


def _coerce_depth(value: Any) -> int:  # noqa: ANN401 -- raw LLM JSON
    """Clamp ``expected_depth`` into the documented 1-5 range."""
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return 3
    if depth < 1:
        return 1
    if depth > 5:
        return 5
    return depth


def _coerce_uuid(value: Any) -> UUID | None:  # noqa: ANN401 -- raw LLM JSON
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return UUID(value.strip())
    except (TypeError, ValueError):
        return None


def _coerce_uuid_list(value: Any) -> list[UUID]:  # noqa: ANN401 -- raw LLM JSON
    if not isinstance(value, list):
        return []
    out: list[UUID] = []
    for item in value:
        coerced = _coerce_uuid(item)
        if coerced is not None:
            out.append(coerced)
    return out


__all__ = [
    "InterviewDifficulty",
    "InterviewQuestionDraft",
    "InterviewQuestionType",
    "parse_generation_response",
]
