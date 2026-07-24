"""Teacher-facing (authoring) DTOs for the quizzes feature (T5.2).

Each Authoring schema inherits from its Public counterpart and widens /
adds fields that are only safe to expose to quiz authors:

* :class:`QuizQuestionOptionAuthoring` re-introduces the ``is_correct``
  flag — teachers grading questions must see which option is right.
  This is the inverse of the public schema's security invariant; the
  authoring router (T5.10 teacher slice) is the only consumer.
* widened ``status`` enum (``draft`` / ``archived`` in addition to
  ``published``);
* generation-pipeline metadata (``review_status``,
  ``original_generated_payload``, ``source_refs``, ``reviewed_by`` /
  ``reviewed_at``, ``published_at``);
* analytics + SR-scheduler inputs (``difficulty``, ``bloom_level``,
  ``expected_response_time_ms``, ``expected_ef_ceiling``);
* shuffle / SR-config toggles (``shuffle_questions`` /
  ``shuffle_options`` / ``initial_ef`` / ``min_ef_for_unlock`` /
  ``coverage_threshold`` / ``reminders_enabled`` /
  ``generation_instructions`` / ``generation_run_id``);
* audit columns from :class:`~abridgeai.core.db.AuditedByMixin` and
  :class:`~abridgeai.core.db.SoftDeleteMixin`.

Field drops vs plan body §5443-5447 (per T5.1 ORM ground truth, §A13)
---------------------------------------------------------------------
The plan body suggested several fields that do NOT exist in the T5.1
ORM:

* :class:`QuizAuthoring`: ``generation_mode`` (no column —
  ``generation_run_id`` carries the run linkage instead);
  ``source_module_id`` (no column — sourced lessons live on the
  ``QuizSourceLesson`` link table); ``internal_notes`` (no column);
  ``draft_count`` (not modelled — the ``review_status='pending'`` count
  is computed by the analytics query layer T5.4, not stored).
* :class:`QuizQuestionAuthoring`: ``validation_verdicts`` (no column —
  validation outcomes live in the generation-run record);
  ``regeneration_count`` (no column — derivable from
  ``QuizQuestionRevision`` row count via a query); ``draft_state`` (no
  column — ``review_status`` covers the lifecycle).

If a future migration adds them, this module is the only place that
needs updating.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from abridgeai.features.quizzes.schemas.public import (
    QuizPublic,
    QuizQuestionOptionPublic,
    QuizQuestionPublic,
)


class QuizQuestionOptionAuthoring(QuizQuestionOptionPublic):
    """Authoring projection of one ``QuizQuestionOption`` row.

    Re-introduces ``is_correct`` (intentionally hidden in
    :class:`QuizQuestionOptionPublic` to prevent answer leakage). Also
    surfaces audit + soft-delete metadata from the mixin column set.
    """

    is_correct: bool
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class QuizQuestionAuthoring(QuizQuestionPublic):
    """Authoring projection of one ``QuizQuestion`` row.

    Inherits the public projection and widens with the full review +
    analytics + audit column set, plus rebinds ``options`` to the
    authoring variant so ``is_correct`` flows through nested payloads.
    """

    options: list[QuizQuestionOptionAuthoring] = []  # type: ignore[assignment]
    explanation: str | None = None
    # Phase 3: render discriminator for explanation (plain | markdown | html).
    explanation_format: str = "plain"
    difficulty: Literal["easy", "medium", "hard"] | None = None
    bloom_level: (
        Literal["remember", "understand", "apply", "analyze", "evaluate", "create"] | None
    ) = None
    review_status: Literal["pending", "approved", "edited", "rejected"]
    expected_response_time_ms: int | None = None
    expected_ef_ceiling: Decimal | None = None
    source_refs: list[Any] = []
    original_generated_payload: dict[str, Any] | None = None
    imported_from_question_id: UUID | None = None
    reviewed_by: UUID | None = None
    reviewed_at: datetime | None = None
    published_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class QuizAuthoring(QuizPublic):
    """Authoring projection of :class:`Quiz`.

    Inherits the public schema and widens ``status`` to the full
    Literal (the ORM CHECK constraint allows all three values), plus
    surfaces course / module linkage, shuffle toggles, SR-config
    knobs, generation-run linkage, and the audit + soft-delete column
    set.
    """

    status: Literal["draft", "published", "archived"]  # type: ignore[assignment]
    course_id: UUID
    module_id: UUID
    # Moodle-style headline-score policy (migration 0033). Patchable via
    # PATCH /teacher/quizzes/{id}; surfaced so the Settings tab can edit it
    # and the results dashboard can label the headline column.
    grading_method: Literal["highest", "average", "first", "last"] = "highest"
    shuffle_questions: bool = False
    shuffle_options: bool = False
    initial_ef: Decimal | None = None
    min_ef_for_unlock: Decimal | None = None
    coverage_threshold: Decimal | None = None
    reminders_enabled: bool = False
    generation_instructions: str | None = None
    generation_run_id: UUID | None = None
    published_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class QuizForAuthoringPublic(BaseModel):
    """Composed authoring tree — quiz + every question (incl. drafts).

    Counterpart to :class:`~abridgeai.features.quizzes.schemas.public.QuizForTakingPublic`
    used by the teacher review screens (T5.10 authoring slice). Carries
    every question regardless of ``review_status`` and exposes the full
    authoring projection (with ``is_correct`` per option).
    """

    model_config = ConfigDict(from_attributes=True)

    quiz: QuizAuthoring
    questions: list[QuizQuestionAuthoring] = []


__all__ = [
    "QuizAuthoring",
    "QuizForAuthoringPublic",
    "QuizQuestionAuthoring",
    "QuizQuestionOptionAuthoring",
]
