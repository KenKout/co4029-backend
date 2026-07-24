"""Review-visibility matrix schema (Phase 2).

A per-quiz matrix controlling what a student may see AFTER submitting, keyed by
time-window (immediately_after / later_while_open / after_close). Every flag
defaults to ``True`` so a quiz with no explicit configuration preserves the
historical always-on review behaviour.

Stored as JSONB on ``Quiz.review_options``; resolved per attempt by
:mod:`abridgeai.features.quizzes.services.review_visibility`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

WINDOW_KEYS = ("immediately_after", "later_while_open", "after_close")
FLAG_KEYS = (
    "show_score",
    "show_correctness",
    "show_correct_answers",
    "show_explanation",
    "show_points",
)


class ReviewWindowFlags(BaseModel):
    """What a student may see during one review time-window."""

    model_config = ConfigDict(extra="forbid")

    show_score: bool = True
    show_correctness: bool = True
    show_correct_answers: bool = True
    show_explanation: bool = True
    show_points: bool = True


class ReviewOptions(BaseModel):
    """The full 3-window × 5-flag matrix. All-true default = today's behaviour."""

    model_config = ConfigDict(extra="forbid")

    immediately_after: ReviewWindowFlags = ReviewWindowFlags()
    later_while_open: ReviewWindowFlags = ReviewWindowFlags()
    after_close: ReviewWindowFlags = ReviewWindowFlags()


__all__ = [
    "FLAG_KEYS",
    "WINDOW_KEYS",
    "ReviewOptions",
    "ReviewWindowFlags",
]
