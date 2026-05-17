"""Interview validation stage (T6.6) — partition AI-generated questions
against five quality criteria.

Public surface:

* :func:`validate_interview_questions` — orchestrator entry point used
  by the interview generation pipeline (T6.10).
* :class:`Verdict` — typed verdict dataclass with parallel index.
* :class:`ValidationCriterion` — five-criterion enum.
* :func:`parse_leading_verdicts` — pure parser usable in tests.
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.validation.logic import (
    DEFAULT_TYPE_WEIGHTS,
    MAX_PROMPT_CHARS,
    MIN_PROMPT_CHARS,
    TYPE_WEIGHT_TOLERANCE,
    VALIDATION_STAGE_NAME,
    InterviewRetrievalContext,
    validate_interview_questions,
)
from abridgeai.features.interviews.ai.stages.validation.parsers import (
    LeadingVerdict,
    parse_leading_verdicts,
)
from abridgeai.features.interviews.ai.stages.validation.verdicts import (
    ValidationCriterion,
    Verdict,
)

__all__ = [
    "DEFAULT_TYPE_WEIGHTS",
    "InterviewRetrievalContext",
    "LeadingVerdict",
    "MAX_PROMPT_CHARS",
    "MIN_PROMPT_CHARS",
    "TYPE_WEIGHT_TOLERANCE",
    "VALIDATION_STAGE_NAME",
    "ValidationCriterion",
    "Verdict",
    "parse_leading_verdicts",
    "validate_interview_questions",
]
