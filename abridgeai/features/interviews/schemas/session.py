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

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
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
# Mirrors ``services.evaluation_state.EvaluationState``. Kept as a local literal
# so the schema layer does not import a service (import-linter contract).
EvaluationStateLiteral = Literal[
    "not_required",
    "pending",
    "succeeded",
    "exhausted",
]
InterviewFinishReasonLiteral = Literal["natural", "ended_early", "timed_out"]
InterviewLanguageLiteral = Literal["en", "vi"]
InterviewTurnActionLiteral = Literal[
    "answer",
    "repeat",
    "clarify",
    "explain_term",
    "hint",
]
InterviewAssistanceKindLiteral = Literal["repeat", "clarification", "term", "hint"]
InterviewOnboardingStageLiteral = Literal[
    "identity_check",
    "audio_check",
    "language_check",
    "preparation",
    "readiness",
    "completed",
]
InterviewOnboardingActionLiteral = Literal[
    "confirm_identity",
    "audio_clear",
    "confirm_language",
    "continue_setup",
    # Retained for compatibility with the original combined setup prompt.
    "confirm_setup",
    "needs_adjustment",
    "ready",
    "not_ready",
    # Identity correction during identity_check: the candidate rejects the
    # profile-derived name, then supplies the name the interviewer should use.
    "reject_identity",
    "set_name",
    # Skip the remaining setup steps and jump straight to the readiness
    # briefing (does NOT start the assessed timer — that still needs "ready").
    "skip_setup",
]


class InterviewSessionHistoryTurn(BaseModel):
    """One learner-visible turn restored when an active attempt resumes."""

    id: str
    role: Literal["ai", "user"]
    content_text: str
    kind: Literal[
        "opening",
        "briefing",
        "transition",
        "question",
        "followup",
        "clarification",
        "hint",
        "answer",
        "closing",
    ]
    created_at: datetime
    elapsed_seconds: int | None = None
    question_type: str | None = None
    is_follow_up: bool = False


class InterviewSessionStartRequest(BaseModel):
    """Body for ``POST /interviews/{id}/sessions``.

    The idempotency key is opaque to the schema layer; the service
    layer (T6.12) enforces uniqueness at the DB level (mirrors the
    ``QuizAttempt.idempotency_key`` UNIQUE pattern from T5.2).
    """

    model_config = ConfigDict(extra="forbid")

    # Accepted for backward compatibility and IGNORED: every session runs the
    # unified hybrid room (migration 0077). Remove once no old client bundle
    # can be mid-flight.
    input_mode: InputModeLiteral | None = None
    idempotency_key: UUID | None = None


class InterviewSessionStartResponse(BaseModel):
    """Response shape for ``POST /interviews/{id}/sessions``.

    Plan §6.2 invariant: the start response carries ``first_question``
    (singular), NEVER a ``questions: list``. Subsequent questions are
    revealed via ``POST /interviews/sessions/{id}/respond``.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    opening_text: str | None = None
    first_question: InterviewQuestionPublic | None = None
    time_remaining_seconds: int | None = None
    question_count_remaining: int | None = None
    onboarding_stage: InterviewOnboardingStageLiteral = "completed"
    interview_language: InterviewLanguageLiteral = "en"
    assessment_started_at: datetime | None = None
    history: list[InterviewSessionHistoryTurn] = Field(default_factory=list)


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
    interview_title: str | None = None
    course_id: UUID | None = None
    status: SessionStatusLiteral
    input_mode: InputModeLiteral
    attempt_number: int
    started_at: datetime
    assessment_started_at: datetime | None = None
    onboarding_stage: InterviewOnboardingStageLiteral = "completed"
    interview_language: InterviewLanguageLiteral = "en"
    ended_at: datetime | None = None
    resume_deadline_at: datetime | None = None
    current_question_index: int | None = None
    time_remaining_seconds: int | None = None
    pass_verdict: bool | None = None
    # Is a verdict still coming? Derived server-side from the terminal status,
    # the verdict, and the recovery budget (see
    # ``services.evaluation_state.derive_evaluation_state``) — the frontend must
    # not re-derive it, because ``status='failed'`` is NOT terminal while the
    # recovery sweep can still re-drive the row. ``pending`` means keep polling;
    # ``succeeded`` / ``exhausted`` / ``not_required`` mean stop.
    evaluation_state: EvaluationStateLiteral = "not_required"
    # Proactive retake context (#7) — see InterviewSessionFinishResponse. Present
    # here too so the results screen survives a reload (the FE re-fetches the
    # session and must still know remaining attempts / cooldown).
    remaining_attempts: int | None = None
    retake_available_at: datetime | None = None
    can_retake: bool = True


class InterviewOnboardingRespondRequest(BaseModel):
    """One idempotent candidate response during pre-assessment onboarding."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "identity_check",
        "audio_check",
        "language_check",
        "preparation",
        "readiness",
    ]
    response_text: str | None = Field(default=None, max_length=1000)
    action: InterviewOnboardingActionLiteral | None = None
    language: InterviewLanguageLiteral | None = None
    turn_key: str = Field(min_length=1, max_length=200)


class InterviewOnboardingRespondResponse(BaseModel):
    """Next persisted onboarding turn, or question one after readiness."""

    onboarding_stage: InterviewOnboardingStageLiteral
    interview_language: InterviewLanguageLiteral
    ai_text: str | None = None
    is_complete: bool
    first_question: InterviewQuestionPublic | None = None
    assessment_started_at: datetime | None = None
    time_remaining_seconds: int | None = None


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
    # Optional explicit UI intent. Older clients omit this and retain the
    # existing answer-classification behavior.
    turn_action: InterviewTurnActionLiteral | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    # Optional client-provided idempotency key for THIS turn (safeguard #1).
    # A retry carrying the same key returns the previously-persisted step
    # response without inserting another answer, re-running the pipeline, or
    # bumping the runtime-state version. Opaque; the adaptive path enforces the
    # guarantee (DB-level partial-unique index + pre-insert check). When absent,
    # the server derives a stable key from (session_question_id + answer digest)
    # so legacy clients still get single-flight protection.
    turn_key: str | None = Field(default=None, max_length=200)


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

    # ── Adaptive interviewer structured fields (Slice 4, Phase 17) ───────────
    # ALL optional with None defaults so the contract stays additive: existing
    # clients that read only the four legacy fields above are unaffected. These
    # are populated ONLY when the adaptive path ran (flag on + text/hybrid +
    # orchestrator succeeded); on the legacy/sequential path they remain None.
    # Every field here is derived from the SAME canonical decision that produced
    # the legacy fields (single source of truth — never built independently).
    ai_turn_text: str | None = None
    language: str | None = None
    should_narrate: bool | None = None
    should_await_response: bool | None = None
    should_finish: bool | None = None
    assistance_kind: InterviewAssistanceKindLiteral | None = None

    # ── End-confirmation gate (Slice 4) ──────────────────────────────────────
    # pending_confirmation is True while the interviewer has asked the candidate
    # to confirm ending (action=request_end_confirmation) and is awaiting a
    # yes/no. interaction_state exposes the per-turn lifecycle axis (separate
    # from interview progress/phase) so the client can render the confirm UI.
    # Additive/optional; legacy + non-adaptive paths leave both None.
    pending_confirmation: bool | None = None
    interaction_state: str | None = None

    # ── Natural Interview Transitions ────────────────────────────────────────
    # A standardized between-turn transition, persisted as its own AI turn and
    # surfaced here so the client can show + narrate it BEFORE revealing the
    # next Question Card (or, when target is "closing", before the goodbye).
    # Additive/optional: legacy clients ignore these; the transition text is
    # ALSO carried in the legacy ai_followup_text for those clients.
    transition_id: str | None = None
    transition_text: str | None = None
    transition_target: Literal["next_question", "closing"] | None = None

    @classmethod
    def from_step_result(
        cls,
        result: Mapping[str, Any],
        *,
        time_remaining_seconds: int | None,
    ) -> InterviewSubmitAnswerResponse:
        """Build the turn response from ``take_session_step``'s result dict.

        THE single place this projection is defined. Two transports deliver a
        typed interview turn — REST ``/respond`` and the LiveKit control topic —
        and they must present identical state, so both call this rather than
        hand-listing fields at their own call site. Adding a field above without
        mapping it here shows up as a failing parity test, not as a field that
        silently reaches one client and not the other.

        ``next_question`` is projected through :class:`InterviewQuestionPublic`,
        so the ORM row never escapes: only ``id``, ``prompt_text`` and
        ``question_type`` are exposed, and authoring-only columns (difficulty,
        review status, source refs) cannot leak to a learner.

        ``time_remaining_seconds`` is a REQUIRED keyword rather than read from
        ``result``: the brain does not return it. It is computed by
        ``session_time_remaining_seconds`` against the session row, and passing
        it explicitly is what stops a caller from quietly publishing ``None``.
        """
        next_question = result.get("next_question")
        return cls(
            # ── legacy fields (always present; unchanged for existing clients) ─
            next_question=(
                InterviewQuestionPublic.model_validate(next_question)
                if next_question is not None
                else None
            ),
            is_finished=bool(result.get("is_finished")),
            ai_followup_text=result.get("followup_text"),
            time_remaining_seconds=time_remaining_seconds,
            # ── adaptive structured fields (None on the legacy/sequential path)
            ai_turn_text=result.get("ai_turn_text"),
            language=result.get("language"),
            should_narrate=result.get("should_narrate"),
            should_await_response=result.get("should_await_response"),
            should_finish=result.get("should_finish"),
            assistance_kind=result.get("assistance_kind"),
            # ── End-confirmation gate (None on legacy/sequential path) ────────
            pending_confirmation=result.get("pending_confirmation"),
            interaction_state=result.get("interaction_state"),
            # ── Natural Interview Transitions (None when no transition) ───────
            transition_id=result.get("transition_id"),
            transition_text=result.get("transition_text"),
            transition_target=result.get("transition_target"),
        )


class InterviewRubricScore(BaseModel):
    """One outcome's verdict slice on the finish response."""

    model_config = ConfigDict(from_attributes=True)

    outcome_id: UUID
    outcome_text: str
    verdict_met: bool
    evidence_excerpt: str | None = None


class InterviewSessionFinishRequest(BaseModel):
    """Optional close context supplied by ceremony-aware clients.

    Older clients may continue sending no request body; that is treated as a
    normal completion.
    """

    model_config = ConfigDict(extra="forbid")

    reason: InterviewFinishReasonLiteral = "natural"


class InterviewSessionFinishResponse(BaseModel):
    """Response shape for ``POST /interviews/sessions/{id}/finish``.

    Thesis §4.3: the student-facing result is **binary pass/fail ONLY** — no
    score, no per-outcome breakdown, no rubric. ``pass_verdict`` is the single
    meaningful signal here.

    ``total_score`` and ``rubric_scores`` are retained in the schema for
    backward compatibility but are ALWAYS ``None`` / empty on this learner
    response; the rubric total + per-outcome ``verdict_met`` + LLM
    ``hidden_reasoning`` are teacher-only and live in
    ``InterviewSession.internal_summary_json`` /
    :class:`~abridgeai.features.interviews.models.InterviewOutcomeEvaluation`.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    status: SessionStatusLiteral
    closing_text: str | None = None
    # Deprecated for students (always None) — see §4.3. Kept for API stability.
    total_score: Decimal | None = None
    # Deprecated for students (always []) — see §4.3. Kept for API stability.
    rubric_scores: list[InterviewRubricScore] = []
    pass_verdict: bool | None = None
    # Same derived label as :class:`InterviewSessionPublic` — see
    # ``services.evaluation_state.derive_evaluation_state``. Present HERE too
    # because ``/finish`` is what the results screen reads first, and it is the
    # only shape it has until the verdict poll returns. Reading ``status`` alone
    # made the screen freeze a recoverable grader failure as terminal: ARQ can
    # stamp ``status='failed'`` before this response is built, and the recovery
    # sweep re-drives exactly those rows.
    evaluation_state: EvaluationStateLiteral = "not_required"
    ended_at: datetime | None = None
    # ── Proactive retake context (#7) ────────────────────────────────────────
    # Surfaced so the results screen can show "N attempts left" and a cooldown
    # countdown instead of only learning the ceiling reactively via a 429/409.
    # remaining_attempts is None when max_attempts is unset (unlimited);
    # retake_available_at is None when no cooldown is currently blocking.
    remaining_attempts: int | None = None
    retake_available_at: datetime | None = None
    can_retake: bool = True


__all__ = [
    "InputModeLiteral",
    "InterviewFinishReasonLiteral",
    "InterviewRubricScore",
    "InterviewSessionFinishRequest",
    "InterviewSessionFinishResponse",
    "InterviewSessionHistoryTurn",
    "InterviewSessionPublic",
    "InterviewSessionStartRequest",
    "InterviewSessionStartResponse",
    "InterviewSubmitAnswerRequest",
    "InterviewSubmitAnswerResponse",
    "SessionStatusLiteral",
]
