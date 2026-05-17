"""Interview-session (taking-flow) DTOs (T6.2).

Back the learner-facing session endpoints (T6.12):

* ``POST /interviews/{id}/sessions`` — start a new session. Body:
  :class:`InterviewSessionStartRequest`. Response:
  :class:`InterviewSessionStartResponse` (returns the FIRST question
  only).
* ``POST /interviews/sessions/{id}/respond`` — submit one answer.
  Body: :class:`InterviewSubmitAnswerRequest`. Response:
  :class:`InterviewSubmitAnswerResponse` (returns the NEXT question
  or ``is_finished=True``).
* ``POST /interviews/sessions/{id}/finish`` — close the session.
  Response: :class:`InterviewSessionFinishResponse` (rubric verdict +
  per-outcome breakdown).
* ``GET  /interviews/sessions/{id}`` — read the student's own
  session. Response: :class:`InterviewSessionPublic`.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585)
-----------------------------------------------------------------

* §6.2 — questions are revealed one-at-a-time. The start response
  carries ``first_question`` (singular); the submit response carries
  ``next_question`` (singular). Never ``questions: list``.
* §A13 — baseline ``InterviewSession.status`` enum is the 5-value set
  ``{in_progress, completed, timed_out, abandoned, failed}`` — NOT
  the plan-body's 3-value or 5-value-but-different subset. The Public
  literal mirrors baseline byte-for-byte.
* §A13 — baseline ``InterviewSession.input_mode`` enum is
  ``{voice, text, hybrid}``. Inherited wisdom §39 confirmed
  ``input_mode`` (NOT ``delivery_format``) is the column name.
* §C4 — :class:`InterviewSessionStartRequest` carries an
  ``idempotency_key`` so retried POSTs on flaky networks resolve to
  the same session row (mirrors the
  :class:`~abridgeai.features.quizzes.schemas.attempt.QuizAttemptStart`
  pattern).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from abridgeai.features.interviews.schemas.public import InterviewQuestionPublic

InputModeLiteral = Literal["voice", "text", "hybrid"]
SessionStatusLiteral = Literal[
    "in_progress",
    "completed",
    "timed_out",
    "abandoned",
    "failed",
]


class InterviewSessionStartRequest(BaseModel):
    """Body for ``POST /interviews/{id}/sessions``.

    The idempotency key is opaque to the schema layer; the service
    layer (T6.12) enforces uniqueness at the DB level (mirrors the
    ``QuizAttempt.idempotency_key`` UNIQUE pattern from T5.2).
    """

    model_config = ConfigDict(extra="forbid")

    input_mode: InputModeLiteral
    idempotency_key: UUID | None = None


class InterviewSessionStartResponse(BaseModel):
    """Response shape for ``POST /interviews/{id}/sessions``.

    Plan §6.2 invariant: the start response carries ``first_question``
    (singular), NEVER a ``questions: list``. Subsequent questions are
    revealed via ``POST /interviews/sessions/{id}/respond``.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    first_question: InterviewQuestionPublic | None = None
    time_remaining_seconds: int | None = None
    question_count_remaining: int | None = None


class InterviewSessionPublic(BaseModel):
    """Student's view of their own ``InterviewSession`` row.

    Fields hidden from the student:

    * ``internal_summary_json`` — teacher-only debrief (LLM rationale).
    * ``transcript_object_id`` / ``recording_object_id`` — exposed via
      a separate signed-URL endpoint, not on this DTO.
    * ``livekit_room_name`` / ``livekit_session_ref`` — runtime
      voice-routing metadata, internal.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    interview_config_id: UUID
    status: SessionStatusLiteral
    input_mode: InputModeLiteral
    attempt_number: int
    started_at: datetime
    ended_at: datetime | None = None
    resume_deadline_at: datetime | None = None
    current_question_index: int | None = None
    time_remaining_seconds: int | None = None
    pass_verdict: bool | None = None


class InterviewSubmitAnswerRequest(BaseModel):
    """Body for ``POST /interviews/sessions/{id}/respond``.

    ``audio_object_id`` is optional — populated for ``input_mode``
    ``voice`` / ``hybrid`` sessions where the answer is uploaded as a
    blob and transcribed downstream. Text-mode sessions populate
    ``answer_text`` only.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    session_question_id: UUID
    answer_text: str | None = None
    audio_object_id: UUID | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class InterviewSubmitAnswerResponse(BaseModel):
    """Response shape for ``POST /interviews/sessions/{id}/respond``.

    Plan §6.2 invariant: ``next_question`` is singular and is ``None``
    once the session reaches its question budget; the client then
    POSTs to ``/finish`` to close the session.

    ``ai_followup_text`` carries an optional probing follow-up
    generated by the interviewer agent (e.g. "Can you elaborate on
    point X?"). The follow-up is delivered ALONGSIDE the next
    structured question; the client renders both.
    """

    model_config = ConfigDict(from_attributes=True)

    next_question: InterviewQuestionPublic | None = None
    is_finished: bool
    ai_followup_text: str | None = None
    time_remaining_seconds: int | None = None


class InterviewRubricScore(BaseModel):
    """One outcome's verdict slice on the finish response."""

    model_config = ConfigDict(from_attributes=True)

    outcome_id: UUID
    outcome_text: str
    verdict_met: bool
    evidence_excerpt: str | None = None


class InterviewSessionFinishResponse(BaseModel):
    """Response shape for ``POST /interviews/sessions/{id}/finish``.

    The student's first glimpse of the rubric verdict. Per-outcome
    ``verdict_met`` flags surface here (so the student knows which
    outcomes they cleared) but the LLM ``hidden_reasoning`` from
    :class:`~abridgeai.features.interviews.models.InterviewOutcomeEvaluation`
    is NEVER returned — that's a teacher-side field.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    status: SessionStatusLiteral
    total_score: Decimal | None = None
    rubric_scores: list[InterviewRubricScore] = []
    pass_verdict: bool | None = None
    ended_at: datetime | None = None


__all__ = [
    "InputModeLiteral",
    "InterviewRubricScore",
    "InterviewSessionFinishResponse",
    "InterviewSessionPublic",
    "InterviewSessionStartRequest",
    "InterviewSessionStartResponse",
    "InterviewSubmitAnswerRequest",
    "InterviewSubmitAnswerResponse",
    "SessionStatusLiteral",
]
