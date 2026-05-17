"""Interview GENERATION stage (T6.5).

Public entry point: :func:`generate_interview_questions`. Parsers and
draft dataclasses live in :mod:`.parsers` so callers can normalise raw
LLM JSON without re-running the LLM call (replay / regression tests).
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.generation.logic import (
    InterviewRetrievalContext,
    generate_interview_questions,
)
from abridgeai.features.interviews.ai.stages.generation.parsers import (
    InterviewDifficulty,
    InterviewQuestionDraft,
    InterviewQuestionType,
    parse_generation_response,
)

__all__ = [
    "InterviewDifficulty",
    "InterviewQuestionDraft",
    "InterviewQuestionType",
    "InterviewRetrievalContext",
    "generate_interview_questions",
    "parse_generation_response",
]
