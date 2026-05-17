"""Quiz persistence stage public API (T5.9).

Two helpers — ``persist_questions`` (initial draft batch) and
``replace_question_in_place`` (regeneration). Both ``db.flush()`` only;
the caller manages the transaction.
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.persistence.logic import (
    persist_questions,
    replace_question_in_place,
)

__all__ = ["persist_questions", "replace_question_in_place"]
