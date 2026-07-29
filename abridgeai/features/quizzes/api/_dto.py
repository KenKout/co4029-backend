"""Pydantic v2 DTOs for the quizzes feature's cross-feature read surface.

These are the typed payload shapes returned by
:mod:`abridgeai.features.quizzes.api.public`. Consumers in other
features (``interviews``, ``spaced_repetition``) bind to these DTOs
rather than to the underlying ORM models — that's what keeps the
features-independent import-linter contract honest.

Conventions
-----------
* All fields use canonical Python types (``UUID``, ``Decimal``,
  ``datetime``).
* ``model_config = ConfigDict(from_attributes=True, frozen=True)`` so
  callers can ``model_validate`` ORM rows directly and so DTOs are
  immutable post-construction (consumers cannot accidentally mutate
  state and write it back).
* No ORM-level relationships are exposed. If a consumer needs a
  related entity, the public surface returns a separate DTO function.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# GenerationRun discriminator + status enums
# ---------------------------------------------------------------------------
#
# Mirrors the CHECK constraint declared on
# ``GenerationRun.__table_args__`` (see ``features/quizzes/models.py``):
#
#     generation_type IN ('quiz', 'interview', 'knowledge_graph',
#                         'material_index')
#
# The api layer publishes these as ``Literal`` types so callers get
# narrowed errors at static-check time when they pass an unsupported
# kind.
GenerationRunKind = Literal[
    "quiz", "interview", "knowledge_graph", "material_index", "interview_evaluation"
]
GenerationRunSourceScopeKind = Literal["lesson", "module", "course"]
GenerationRunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class QuestionWithQuizDTO(BaseModel):
    """Quiz question + parent-quiz context, used by SR remediation.

    Returns just the fields SR remediation needs to assemble follow-up
    prompts and KG retrieval seeds. Soft-deleted rows are filtered by
    the ORM loader-criteria, so a soft-deleted question yields
    ``None`` from the loader (not a populated DTO).
    """

    model_config = ConfigDict(from_attributes=True, frozen=True)

    question_id: UUID
    quiz_id: UUID
    prompt_text: str
    source_refs: list[Any]
    course_id: UUID
    module_id: UUID
    initial_ef: Decimal | None = None


class GradeReviewResultDTO(BaseModel):
    """Outcome of grading one card-review answer (cross-feature).

    Returned by :func:`quizzes.api.public.grade_review_answer` to the SR review
    loop. Carries the correctness signal the SM-2 scheduler needs plus the
    feedback the learner sees after answering (correct option / answer text +
    explanation). Feedback fields are only meaningful post-submit, so this DTO
    is never part of the pre-answer question payload.
    """

    model_config = ConfigDict(from_attributes=True, frozen=True)

    is_correct: bool
    #: Ids of the option(s) that were correct (MCQ/true_false). Empty for
    #: free-text types.
    correct_option_ids: list[UUID] = []
    #: Canonical answer rendered as text for free-text / non-option types
    #: (short_answer, fill_blank, numerical, matching, ordering). None for MCQ.
    correct_answer_text: str | None = None
    #: Teacher explanation for the question, if any (already sanitized).
    explanation: str | None = None


class AttemptScoreDTO(BaseModel):
    """Subset of ``QuizAttempt`` used by interviews evaluation.

    The interviews Gap Report needs ``score_percent`` per quiz attempt
    to compute a student's per-quiz historical averages. Other
    attempt fields (timestamps, answers) are not exposed here — they
    stay encapsulated in the quizzes feature.
    """

    model_config = ConfigDict(from_attributes=True, frozen=True)

    attempt_id: UUID
    quiz_id: UUID
    student_id: UUID
    status: str
    score_percent: Decimal | None
    passed: bool | None


class GenerationRunDTO(BaseModel):
    """Snapshot of a ``generation_runs`` row.

    Consumers (interviews AI pipeline, quiz workers) read these fields
    to dispatch + render run state. Mutation goes through the api
    surface (status updates remain INSIDE the quizzes feature; this
    DTO is read-only).
    """

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    generation_type: GenerationRunKind
    source_scope_kind: GenerationRunSourceScopeKind
    course_id: UUID | None
    module_id: UUID | None
    lesson_id: UUID | None
    requested_by: UUID | None
    status: GenerationRunStatus
    config_json: dict[str, Any]
    dedup_key: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "AttemptScoreDTO",
    "GenerationRunDTO",
    "GenerationRunKind",
    "GenerationRunSourceScopeKind",
    "GenerationRunStatus",
    "GradeReviewResultDTO",
    "QuestionWithQuizDTO",
]
