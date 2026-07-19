"""Student-facing (public) DTOs for the interviews feature (T6.2).

These schemas back the learner slice of the interviews router (T6.12)
and represent the narrowest projection of the interview aggregate.

Security invariants for this module
-----------------------------------

* :class:`InterviewQuestionPublic` MUST NOT carry any of:

  - ``difficulty`` — gives away the expected complexity tier.
  - ``review_status``, ``ai_generated`` — internal review-pipeline
    metadata.
  - ``source_refs_json`` — the chunk_ids that grounded the question;
    leaking them gives the student the source material verbatim.
  - ``reviewed_by`` / ``reviewed_at`` — internal review audit.

  Authoring callers re-introduce all of the above via
  :class:`~abridgeai.features.interviews.schemas.authoring.InterviewQuestionAuthoring`.

* :class:`InterviewOutcomePublic` MUST NOT carry ``importance_weight``
  — exposing rubric weights lets a learner game which outcomes to
  prioritise. ``InterviewOutcomeAuthoring`` re-introduces it.

* :class:`InterviewConfigPublic.status`` narrows to
  ``Literal["published"]`` — draft / archived configs never surface.

* :class:`InterviewForTakingPublic`` carries ONLY the first question.
  Subsequent questions are revealed dynamically via
  ``POST /interviews/sessions/{id}/respond`` (per T6.12 session flow).
  Baking the full question list into the start payload would defeat
  the one-question-at-a-time interview UX.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585)
-----------------------------------------------------------------

* §A13 — baseline DDL is ground truth. Field names mirror T6.1 ORM
  columns: ``prompt_text`` (not ``prompt``), ``outcome_text`` (not
  ``criterion``), ``time_limit_minutes`` (interviews use **minutes**;
  quizzes use seconds). Persona literal is ``{strict, neutral,
  supportive}`` (NOT plan-body's ``friendly``). Supported modes are
  ``{voice, text, hybrid}``. Question types are the broader
  ``{conceptual, behavioral, technical, situational, system_design}``
  set (NOT plan-body's 3-value subset).
* §6.2 — questions revealed one-at-a-time per session flow. The
  schema enforces this by exposing ``first_question`` (singular) on
  :class:`InterviewForTakingPublic`, never ``questions: list``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

PersonaLiteral = Literal["strict", "neutral", "supportive"]
SupportedModesLiteral = Literal["voice", "text", "hybrid"]
QuestionTypeLiteral = Literal[
    "conceptual",
    "behavioral",
    "technical",
    "situational",
    "system_design",
]
OutcomeTypeLiteral = Literal["knowledge", "skill", "attitude"]


class _ORMModel(BaseModel):
    """Shared base — Pydantic v2 ORM-mode equivalent (`from_attributes=True`)."""

    model_config = ConfigDict(from_attributes=True)


class InterviewOutcomePublic(_ORMModel):
    """Student-facing projection of one ``InterviewOutcome`` row.

    SECURITY INVARIANT: ``importance_weight`` MUST NOT appear here.
    Retained for backwards schema compatibility. The learner taking endpoint
    returns no outcomes because the complete outcome set is protected rubric
    and coverage metadata.
    """

    id: UUID
    position: int
    outcome_text: str
    outcome_type: OutcomeTypeLiteral


class InterviewQuestionPublic(_ORMModel):
    """Student-facing projection of one ``InterviewQuestion`` row.

    Carries only what the learner needs to render and answer the
    question: identity, prompt, and the question-type discriminator.

    Authoring-only fields hidden here (re-added in
    :class:`~abridgeai.features.interviews.schemas.authoring.InterviewQuestionAuthoring`):

    * ``difficulty`` — expected-complexity hint.
    * ``review_status``, ``ai_generated`` — pipeline metadata.
    * ``source_refs_json`` — grounding chunks (would leak source).
    * ``reviewed_by`` / ``reviewed_at`` — review audit.
    * ``linked_outcome_id``, ``position`` — internal authoring layout.
    * Audit + soft-delete columns.
    """

    id: UUID
    prompt_text: str
    question_type: QuestionTypeLiteral


class InterviewConfigPublic(_ORMModel):
    """Student-facing interview-config summary.

    ``status`` narrows to ``Literal["published"]`` — the public catalog
    only ever surfaces published configs. The authoring counterpart
    :class:`~abridgeai.features.interviews.schemas.authoring.InterviewConfigAuthoring`
    widens to the full enum so teachers can see drafts.

    Authoring-only fields hidden here:

    * ``supplementary_instructions`` — author-only notes.
    * ``generation_run_id`` — pipeline linkage.
    * ``min_outcomes_to_pass`` — exposing the pass threshold lets a
      student stop trying once they've cleared it; it remains teacher-only.
    * Audit + soft-delete columns.
    """

    id: UUID
    course_id: UUID
    module_id: UUID
    title: str
    status: Literal["published"]
    persona: PersonaLiteral | None = None
    supported_modes: SupportedModesLiteral
    time_limit_minutes: int | None = None
    max_attempts: int | None = None
    cooldown_hours: int | None = None
    lock_quiz_ef_until_pass: bool
    published_at: datetime | None = None


class InterviewForTakingPublic(_ORMModel):
    """Composed take-interview start payload.

    Carries the published config and the **first** question only. Outcome and
    coverage metadata are deliberately absent from the learner contract.
    Subsequent questions are revealed via
    ``POST /interviews/sessions/{id}/respond`` per T6.12 session flow.

    Plan §6.2 explicit MUST NOT: do not bake all questions into the
    start payload. The ``first_question`` singular field enforces this
    at the schema layer.
    """

    config: InterviewConfigPublic
    first_question: InterviewQuestionPublic | None = None


__all__ = [
    "InterviewConfigPublic",
    "InterviewForTakingPublic",
    "InterviewOutcomePublic",
    "InterviewQuestionPublic",
    "OutcomeTypeLiteral",
    "PersonaLiteral",
    "QuestionTypeLiteral",
    "SupportedModesLiteral",
]
