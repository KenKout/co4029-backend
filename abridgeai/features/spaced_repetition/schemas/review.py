"""Pydantic models for the SR flashcard review loop.

The review loop is how a student *resolves* a due card without re-taking the
whole source quiz: fetch the due queue (`ReviewCard` list), answer each card,
and submit for grading. Grading reuses the canonical quiz grader and fires the
same SM-2 ``record_card_review`` write, so a passed card is rescheduled out and
a failed one resets — identical to answering it inside a quiz attempt.

The question payload embedded in each card is the quizzes feature's
``QuizQuestionPublic`` (no-leak: no ``is_correct`` on options), fetched through
the quizzes public API. It is typed as ``Any`` here to avoid importing a
sibling feature's schema into SR's own schema module; the router validates it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ReviewCard(BaseModel):
    """One due card in the review queue: SR state + its question payload."""

    question_id: UUID
    quiz_id: UUID
    lesson_id: UUID
    #: Mirrors ``CardsDueItem.lesson_slug`` so the review screen can build
    #: slug URLs without a second lookup.
    lesson_slug: str
    lesson_title: str
    course_slug: str
    course_title: str
    due_at: datetime
    ef: float
    last_q: int | None = None
    #: The no-leak ``QuizQuestionPublic`` payload (prompt, options w/o
    #: is_correct, derived shuffles). Rendered by the same SPA component the
    #: quiz-taking surface uses.
    question: Any


class ReviewQueue(BaseModel):
    items: list[ReviewCard]
    total_due: int
    #: Admin-configured daily review cap; 0 means unlimited. The queue length
    #: is bounded to what remains of this cap today. Bounds the queue ONLY —
    #: unlock eligibility and retention scoring are never affected.
    daily_cap: int = 0
    #: Cards this student has already reviewed today (counts toward the cap).
    reviewed_today: int = 0
    #: Cards still allowed today = max(0, daily_cap - reviewed_today); when
    #: daily_cap is 0 this equals total_due (no cap). 0 with total_due > 0
    #: means the student hit today's cap and should come back tomorrow.
    daily_remaining: int = 0


class ReviewSubmitRequest(BaseModel):
    """A student's answer to one review card. Mirrors the quiz answer shape."""

    selected_option_id: UUID | None = None
    answer_text: str | None = None
    hint_used: bool = False
    #: Client-measured attention time in ms; feeds the SM-2 ρ (speed) grade.
    #: Omitted → neutral ρ=1.0 fallback server-side.
    t_actual_ms: int | None = None


class ReviewSubmitResult(BaseModel):
    """Post-grade feedback + the new SM-2 schedule for the card."""

    question_id: UUID
    correct: bool
    #: SM-2 grade 0-5 derived from correctness + hint + speed.
    q: int
    #: True when q >= 3 (card advances); False resets it to a 1-day interval.
    passing: bool
    #: New review timestamp the card was rescheduled to.
    due_at: datetime
    #: Days until the next review (post-jitter).
    interval_days: int
    #: How many cards remain due AFTER this submission (drives the counter).
    remaining_due: int
    # -- feedback (what the learner sees after answering) --
    correct_option_ids: list[UUID] = []
    correct_answer_text: str | None = None
    explanation: str | None = None


__all__ = [
    "ReviewCard",
    "ReviewQueue",
    "ReviewSubmitRequest",
    "ReviewSubmitResult",
]
