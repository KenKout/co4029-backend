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

import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, cast
from uuid import UUID

InterviewQuestionType = Literal["technical", "behavioral", "situational", "system_design"]
InterviewDifficulty = Literal["easy", "medium", "hard"]

_VALID_TYPES: frozenset[str] = frozenset(
    {"technical", "behavioral", "situational", "system_design"}
)
_VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})


@dataclass
class InterviewQuestionDraft:
    """Pre-validation draft of a single generated interview question.

    Attributes
    ----------
    question_type
        Exactly one of ``technical`` / ``behavioral`` / ``situational``.
    prompt_text
        The stem the learner will read. Stands alone (no slide refs).
    difficulty
        One of ``easy`` / ``medium`` / ``hard``. Mapped to the ORM-side
        ``junior`` / ``mid_level`` / ``senior`` set during persistence.
    expected_depth
        Integer 1-5 describing how long an adequate answer should be.
        1 = one-sentence; 5 = multi-paragraph reasoning.
    linked_outcome_id
        UUID of the :class:`InterviewOutcome` this question assesses.
        ``None`` is allowed when no outcomes were provided to the stage.
    source_refs
        UUIDs of the source chunks the question is grounded in.
    rationale
        Internal explanation of why this question was generated.
        Never shown to the learner.
    model_answer
        Teacher-facing reference/model answer for this question. Authoring
        aid only — never shown to the learner and never used to auto-grade
        (live sessions are scored against :class:`InterviewOutcome` rubric
        criteria, not string/semantic matching against this field).
    """

    question_type: InterviewQuestionType
    prompt_text: str
    difficulty: InterviewDifficulty
    expected_depth: int
    linked_outcome_id: UUID | None
    # Model-supplied ordinal used only to correlate the current all-angle
    # response. The durable UUID is assigned server-side after parsing.
    logical_question_index: int | None = None
    variant_group_id: UUID | None = None
    source_refs: list[UUID] = field(default_factory=list)
    rationale: str = ""
    model_answer: str = ""


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
        Cap the returned list at this length. In all-angle mode, only whole
        logical groups that fit are retained; other modes keep the first N
        valid entries.
    require_logical_question_index
        All-angle generation requires an integer logical-group ordinal on each
        row. Other strategies ignore it and leave durable group IDs unset.
    """
    raw = _extract_question_list(payload)
    drafts = [draft for entry in raw if (draft := _prepare_question(entry)) is not None]
    if require_logical_question_index:
        from abridgeai.features.interviews.ai.stages.generation.variant_groups import (  # noqa: PLC0415
            select_all_angle_groups,
        )

        return select_all_angle_groups(drafts, max_questions)
    return drafts[:max_questions]


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
    logical_question_index = _coerce_logical_question_index(entry.get("logical_question_index"))
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


def _coerce_logical_question_index(value: Any) -> int | None:  # noqa: ANN401 -- raw LLM JSON
    """Normalize finite non-negative integral group ordinals from LLM JSON."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if isfinite(value) and value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        cleaned = value.strip()
        return int(cleaned) if re.fullmatch(r"[0-9]+", cleaned) else None
    return None


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
