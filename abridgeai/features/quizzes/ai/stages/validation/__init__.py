"""Quiz validation stage (T5.7) — extracted from god file lines 817-868.

The validator is a separate LLM call (``LLMRole.VALIDATION``) that
judges generation-stage output and emits a per-question verdict. The
public surface is:

* :func:`validate_questions` — orchestrator entry point.
* :func:`parse_validation_response` — pure parser usable in tests.
* :func:`apply_verdicts` — partition questions into accepted vs
  rejected with reasons.
* :class:`Verdict` — typed verdict dataclass.
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.validation.logic import (
    VALIDATION_STAGE_NAME,
    validate_questions,
)
from abridgeai.features.quizzes.ai.stages.validation.parsers import (
    Verdict,
    parse_validation_response,
)
from abridgeai.features.quizzes.ai.stages.validation.verdicts import apply_verdicts

__all__ = [
    "VALIDATION_STAGE_NAME",
    "Verdict",
    "apply_verdicts",
    "parse_validation_response",
    "validate_questions",
]
