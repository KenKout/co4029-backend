"""Teacher-facing (authoring) DTOs for the interviews feature (T6.2).

Each Authoring schema inherits from its Public counterpart and widens /
adds fields that are only safe to expose to interview authors:

* :class:`InterviewQuestionAuthoring` re-introduces ``difficulty``,
  ``review_status``, ``ai_generated``, ``source_refs_json``,
  ``reviewed_by`` / ``reviewed_at``, ``position``,
  ``linked_outcome_id`` — all of which would leak hints / source
  material to a learner taking the interview.
* :class:`InterviewOutcomeAuthoring` re-introduces
  ``importance_weight`` — the rubric weight the student must NEVER
  see (gaming the verdict).
* :class:`InterviewConfigAuthoring` widens ``status`` to the full
  enum (``draft`` / ``published`` / ``archived``) and surfaces
  ``supplementary_instructions``, ``generation_run_id``,
  ``min_outcomes_to_pass``, plus the audit + soft-delete column set.

Generation-pipeline schemas live in this module too
(:class:`InterviewGenerationRequest`, :class:`InterviewGenerationRunPublic`)
because they are teacher-only — students never trigger generation.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585)
-----------------------------------------------------------------

* §A13 — baseline DDL is ground truth. ``InterviewConfig.status`` enum
  is ``{draft, published, archived}``. ``InterviewQuestion.difficulty``
  is the level string ``{junior, mid_level, senior}`` (the inherited
  "difficulty → integer 1-5" note in §39 targets quiz_questions only;
  interviews keep the level-string shape).
  ``InterviewQuestion.review_status`` is
  ``{pending, approved, edited, rejected}``.
  ``source_refs_json`` keeps the ``_json`` suffix verbatim from
  baseline (the parallel rename to ``source_refs`` was scoped to
  ``quiz_questions`` only via migration 0007 — interviews are out of
  scope for that rename).
* §6.2 — generation request body carries ``mode`` literal
  ``{topic, outcome-based, coverage}`` (interview-specific generation
  modes; differs from quiz's ``{full, coverage, manual, regenerate}``).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from abridgeai.features.interviews.schemas.public import (
    InterviewConfigPublic,
    InterviewOutcomePublic,
    InterviewQuestionPublic,
    OutcomeTypeLiteral,
    PersonaLiteral,
    QuestionTypeLiteral,
    SupportedModesLiteral,
)

DifficultyLiteral = Literal["junior", "mid_level", "senior"]
ReviewStatusLiteral = Literal["pending", "approved", "edited", "rejected"]
ConfigStatusLiteral = Literal["draft", "published", "archived"]
GenerationModeLiteral = Literal["topic", "outcome-based", "coverage"]
GenerationRunStatusLiteral = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]


# --------------------------------------------------------------------------- #
# Config — create / patch / authoring projection
# --------------------------------------------------------------------------- #


class InterviewConfigCreate(BaseModel):
    """Body for ``POST /teacher/interviews``."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=255)
    course_id: UUID
    module_id: UUID
    persona: PersonaLiteral | None = None
    supported_modes: SupportedModesLiteral = "hybrid"
    time_limit_minutes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    lock_quiz_ef_until_pass: bool = False
    supplementary_instructions: str | None = None


class InterviewConfigUpdate(BaseModel):
    """Body for ``PATCH /teacher/interviews/{id}`` — partial fields."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=255)
    status: ConfigStatusLiteral | None = None
    persona: PersonaLiteral | None = None
    supported_modes: SupportedModesLiteral | None = None
    time_limit_minutes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    min_outcomes_to_pass: int | None = Field(default=None, ge=1)
    lock_quiz_ef_until_pass: bool | None = None
    supplementary_instructions: str | None = None


class InterviewConfigAuthoring(InterviewConfigPublic):
    """Authoring projection of :class:`InterviewConfig`.

    Inherits the public schema and widens ``status`` to the full
    Literal (the ORM CHECK constraint allows all three values), plus
    surfaces the supplementary instructions, the generation-run
    linkage, the pass-threshold knob, and the audit + soft-delete
    column set.
    """

    status: ConfigStatusLiteral  # type: ignore[assignment]
    supplementary_instructions: str | None = None
    min_outcomes_to_pass: int | None = None
    generation_run_id: UUID | None = None
    draft_question_count: int | None = None
    total_importance_weight: int | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


# --------------------------------------------------------------------------- #
# Outcome — create / authoring projection
# --------------------------------------------------------------------------- #


class InterviewOutcomeCreate(BaseModel):
    """Body for ``POST /teacher/interviews/{id}/outcomes``."""

    model_config = ConfigDict(extra="forbid")

    position: int = Field(ge=1)
    outcome_text: str
    outcome_type: OutcomeTypeLiteral
    importance_weight: int = Field(default=1, ge=1, le=5)


class InterviewOutcomeAuthoring(InterviewOutcomePublic):
    """Authoring projection of :class:`InterviewOutcome`.

    Re-introduces ``importance_weight`` (intentionally hidden in
    :class:`InterviewOutcomePublic` to prevent rubric-weight leakage)
    and surfaces audit + soft-delete metadata.
    """

    interview_config_id: UUID
    importance_weight: int
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


# --------------------------------------------------------------------------- #
# Question — create / authoring projection
# --------------------------------------------------------------------------- #


class InterviewQuestionCreate(BaseModel):
    """Body for ``POST /teacher/interviews/{id}/questions``."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    question_type: QuestionTypeLiteral
    difficulty: DifficultyLiteral | None = None
    linked_outcome_id: UUID | None = None
    position: int | None = Field(default=None, ge=1)


class InterviewQuestionAuthoring(InterviewQuestionPublic):
    """Authoring projection of :class:`InterviewQuestion`.

    Inherits the public projection and widens with the full review +
    pipeline + audit column set.
    """

    interview_config_id: UUID
    linked_outcome_id: UUID | None = None
    position: int | None = None
    difficulty: DifficultyLiteral | None = None
    review_status: ReviewStatusLiteral
    ai_generated: bool
    source_refs_json: list[Any] = []
    reviewed_by: UUID | None = None
    reviewed_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


# --------------------------------------------------------------------------- #
# Generation pipeline — request + run-status projections
# --------------------------------------------------------------------------- #


class InterviewGenerationRequest(BaseModel):
    """Body for ``POST /teacher/interviews/{id}/generate``.

    The mode discriminator selects the generation strategy:

    * ``topic`` — generate questions covering listed focus topics.
    * ``outcome-based`` — generate one (or more) per rubric outcome.
    * ``coverage`` — fill gaps in lesson coverage from ``source_lesson_ids``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: GenerationModeLiteral
    course_id: UUID
    module_id: UUID
    source_lesson_ids: list[UUID] = []
    question_count: int = Field(default=5, ge=1, le=50)
    focus_topics: list[str] = []
    avoid_topics: list[str] = []
    persona: PersonaLiteral | None = None
    supplementary_instructions: str | None = None


class InterviewGenerationRunPublic(BaseModel):
    """Status-poll projection of an interview generation run.

    The underlying ``GenerationRun`` row is shared with the quizzes
    feature (T5.1 stop-gap port until T5.x moves it to
    ``features/ai/models.py``).
    """

    model_config = ConfigDict(from_attributes=True)

    run_id: UUID
    status: GenerationRunStatusLiteral
    config_json: dict[str, Any] = {}
    started_at: datetime
    finished_at: datetime | None = None
    failure_message: str | None = None


# --------------------------------------------------------------------------- #
# Composed authoring view
# --------------------------------------------------------------------------- #


class InterviewForAuthoringPublic(BaseModel):
    """Composed authoring tree — config + every outcome + every question.

    Counterpart to
    :class:`~abridgeai.features.interviews.schemas.public.InterviewForTakingPublic`
    used by the teacher review screens. Carries every question
    regardless of ``review_status`` and exposes the full authoring
    projection (with ``difficulty`` and ``importance_weight``).
    """

    model_config = ConfigDict(from_attributes=True)

    config: InterviewConfigAuthoring
    outcomes: list[InterviewOutcomeAuthoring] = []
    questions: list[InterviewQuestionAuthoring] = []


# Suppress unused-import warning — Decimal is exported in case downstream
# generation-config payloads carry numeric thresholds. Keep available.
_DECIMAL_AVAILABLE = Decimal


__all__ = [
    "ConfigStatusLiteral",
    "DifficultyLiteral",
    "GenerationModeLiteral",
    "GenerationRunStatusLiteral",
    "InterviewConfigAuthoring",
    "InterviewConfigCreate",
    "InterviewConfigUpdate",
    "InterviewForAuthoringPublic",
    "InterviewGenerationRequest",
    "InterviewGenerationRunPublic",
    "InterviewOutcomeAuthoring",
    "InterviewOutcomeCreate",
    "InterviewQuestionAuthoring",
    "InterviewQuestionCreate",
    "ReviewStatusLiteral",
]
