"""Quiz dedup stage (T5.8).

Public surface re-exports :func:`discard_duplicates` and the
:class:`QuestionDrop` dataclass. The :data:`REASON_*` constants are
also exported so callers (teacher-review surface, audit logger) can
match drop reasons by symbol rather than string literal.
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.dedup.logic import (
    REASON_BATCH_DUPLICATE,
    REASON_EMPTY_PROMPT,
    REASON_EXISTING_MODULE_DUPLICATE,
    QuestionDrop,
    discard_duplicates,
)

__all__ = [
    "QuestionDrop",
    "REASON_BATCH_DUPLICATE",
    "REASON_EMPTY_PROMPT",
    "REASON_EXISTING_MODULE_DUPLICATE",
    "discard_duplicates",
]
