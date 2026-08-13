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

    The taking endpoint still returns no outcomes — the complete outcome set is
    protected rubric and coverage metadata, and ``InterviewForTakingPublic``
    exposes only the count. ``importance_weight`` and ``min_outcomes_to_pass``
    remain hidden everywhere, because those are what make a rubric gameable.
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
    # Deepgram Aura voice for English sessions (NULL = deployment default). Safe
    # to expose: it only names the spoken voice, nothing gameable. Vietnamese
    # sessions ignore it (browser voice), so the UI shows it for English only.
    tts_voice: str | None = None
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

    ``outcome_count`` is a SAFE expectation-setting signal: it exposes only
    *how many* rubric criteria the interview assesses, never their text,
    ``importance_weight``, or the ``min_outcomes_to_pass`` threshold — all of
    which remain teacher-only per the security invariants above. A bare count
    lets the learner UI show "assessed on N criteria" without leaking anything
    gameable.
    """

    config: InterviewConfigPublic
    first_question: InterviewQuestionPublic | None = None
    outcome_count: int = 0


class InterviewProgressRead(BaseModel):
    """Per-interview completion state for one student, for course-learn.

    Interviews were graded per attempt long before this existed
    (``interview_sessions.pass_verdict``), but the verdict never reached the
    curriculum, so an interview item stayed pending forever.

    Completion rule (user decision, 2026-08-06): completed ⟺ at least one
    **non-practice** attempt has ``pass_verdict = TRUE``. This is intentionally
    STRICTER than the quiz rule, which also completes on "failed with every
    attempt consumed": here the tag is meant to read as *passed*, so a student
    who failed every attempt keeps the item pending. Practice runs never count,
    matching the attempt gate — rehearsing must not tick off a graded
    milestone.

    ``attempts_graded`` is exposed separately from ``attempts_used`` because a
    finished attempt is not necessarily a graded one: evaluation is an ARQ job,
    so a just-submitted attempt sits with ``pass_verdict IS NULL`` until the
    worker lands. A UI can use the gap to say "being marked" rather than
    implying a fail.

    SECURITY: deliberately carries no score, rubric aggregate, outcome text or
    ``min_outcomes_to_pass``. The learner contract for interviews withholds all
    of those (see :class:`InterviewForTakingPublic`), and a progress payload is
    not a licence to leak them — a pass/fail boolean is the whole signal the
    curriculum needs.
    """

    interview_config_id: UUID
    attempts_used: int
    attempts_in_flight: int = 0
    attempts_graded: int = 0
    passed: bool = False
    completed: bool


__all__ = [
    "InterviewConfigPublic",
    "InterviewForTakingPublic",
    "InterviewOutcomePublic",
    "InterviewProgressRead",
    "InterviewQuestionPublic",
    "OutcomeTypeLiteral",
    "PersonaLiteral",
    "QuestionTypeLiteral",
    "SupportedModesLiteral",
]
