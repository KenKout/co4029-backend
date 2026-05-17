"""Quiz GENERATION stage (T5.6) — extracted from god file lines 758-816.

Public entry point: :func:`generate_questions`. Parsers / Pydantic
schemas live in :mod:`.parsers` so callers can normalise raw LLM JSON
without re-running the LLM call (used by replay / regression tests).
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.generation.logic import generate_questions
from abridgeai.features.quizzes.ai.stages.generation.parsers import (
    GeneratedQuestion,
    GeneratedQuestionOption,
    normalize_question_text,
    parse_generation_response,
    question_for_review,
)

__all__ = [
    "GeneratedQuestion",
    "GeneratedQuestionOption",
    "generate_questions",
    "normalize_question_text",
    "parse_generation_response",
    "question_for_review",
]
