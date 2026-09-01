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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from abridgeai.features.interviews.schemas.public import (
    InterviewConfigPublic,
    InterviewOutcomePublic,
    InterviewQuestionPublic,
    OutcomeTypeLiteral,
    PersonaLiteral,
    QuestionTypeLiteral,
)
from abridgeai.features.interviews.schemas.session import (
    InputModeLiteral,
    SessionStatusLiteral,
)

DifficultyLiteral = Literal["junior", "mid_level", "senior"]
ReviewStatusLiteral = Literal["pending", "approved", "edited", "rejected"]
ConfigStatusLiteral = Literal["draft", "published", "archived"]
GenerationRunStatusLiteral = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]
SecurityResponsePolicyLiteral = Literal[
    "continue_and_log",
    "warn_and_continue",
    "end_and_flag",
]
# Deepgram Aura-2 English voices offered in the teacher UI. MUST stay in sync
# with ``services.narration.ALLOWED_TTS_VOICES`` (the runtime allow-list) and
# the frontend voice dropdown. NULL = deployment default
# (settings.deepgram_tts_model_en). English-only: Vietnamese has no server TTS.
TtsVoiceLiteral = Literal[
    "aura-2-thalia-en",
    "aura-2-andromeda-en",
    "aura-2-helena-en",
    "aura-2-apollo-en",
    "aura-2-arcas-en",
    "aura-2-aries-en",
    "aura-2-asteria-en",
    "aura-2-athena-en",
    "aura-2-hera-en",
    "aura-2-hyperion-en",
    "aura-2-luna-en",
    "aura-2-orion-en",
    "aura-2-orpheus-en",
    "aura-2-ophelia-en",
    "aura-2-zeus-en",
    "aura-2-vesta-en",
]


# --------------------------------------------------------------------------- #
# Config — create / patch / authoring projection
# --------------------------------------------------------------------------- #


OpeningStyleLiteral = Literal["brief", "standard", "comfort"]
# Which professional identity the interviewer presents as. Orthogonal to
# ``persona`` (which is tone): a supportive engineering manager and a strict one
# are both coherent. Kept out of the ``persona`` column so its three-value CHECK
# constraint and preset invariants stay untouched.
InterviewerRoleLiteral = Literal[
    "generic_assistant",
    "backend_tech_lead",
    "staff_engineer",
    "eng_manager",
    "hr_screener",
]


class PersonaProfileWrite(BaseModel):
    """Optional per-trait persona overrides on a config (Phase 3).

    Every field is optional so a teacher can nudge one dial (e.g. warmth) and
    leave the rest to the ``persona`` preset. Traits are 0-4; the service merges
    this over the preset and clamps, so an out-of-range or partial payload is
    still safe. TONE ONLY — these never reach difficulty, selection, or scoring.
    """

    model_config = ConfigDict(extra="forbid")

    warmth: int | None = Field(default=None, ge=0, le=4)
    directness: int | None = Field(default=None, ge=0, le=4)
    verbosity: int | None = Field(default=None, ge=0, le=4)
    formality: int | None = Field(default=None, ge=0, le=4)
    ack_frequency: int | None = Field(default=None, ge=0, le=4)
    opening_style: OpeningStyleLiteral | None = None
    # Rides in the same JSONB blob as the trait overrides — no migration, the
    # precedent being ``opening_style``, which is likewise a non-numeric enum.
    # Absent → the generic assistant, i.e. the pre-identity wording verbatim.
    interviewer_role: InterviewerRoleLiteral | None = None


class PersonaProfileRead(BaseModel):
    """The RESOLVED persona profile (preset merged with any overrides).

    Teacher-only projection: what tone the interviewer will actually use. Never
    exposed on a learner-facing schema (same rule as importance_weight), because
    it reveals nothing a candidate needs and keeping it teacher-side avoids any
    perception that tone affects grading.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    warmth: int
    directness: int
    verbosity: int
    formality: int
    ack_frequency: int
    opening_style: OpeningStyleLiteral
    interviewer_role: InterviewerRoleLiteral


class InterviewConfigCreate(BaseModel):
    """Body for ``POST /teacher/interviews``."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=255)
    course_id: UUID
    module_id: UUID
    persona: PersonaLiteral | None = None
    # Optional per-trait overrides layered on the persona preset (Phase 3).
    persona_profile: PersonaProfileWrite | None = None
    tts_voice: TtsVoiceLiteral | None = None
    time_limit_minutes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    cooldown_hours: int | None = Field(default=None, ge=1)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=50)
    max_hints_per_question: int = Field(default=3, ge=0, le=10)
    supplementary_instructions: str | None = None
    security_response_policy: SecurityResponsePolicyLiteral = "warn_and_continue"
    security_max_consecutive_attempts: int = Field(default=3, ge=2, le=20)
    security_custom_refusal_en: str | None = Field(default=None, max_length=500)
    security_custom_refusal_vi: str | None = Field(default=None, max_length=500)
    security_incident_summary_enabled: bool = True


class InterviewConfigUpdate(BaseModel):
    """Body for ``PATCH /teacher/interviews/{id}`` — partial fields.

    ``status`` is intentionally omitted here — status transitions
    (draft → published, published → archived, archived → draft) go
    through dedicated endpoints (``/publish``, ``/archive``,
    ``/unarchive``) so the service layer can enforce business rules.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=255)
    persona: PersonaLiteral | None = None
    # Optional per-trait overrides layered on the persona preset (Phase 3).
    # NOTE: send an explicit empty object {} would clear nothing here; the
    # router treats None as "unchanged" and a present object as "replace".
    persona_profile: PersonaProfileWrite | None = None
    tts_voice: TtsVoiceLiteral | None = None
    time_limit_minutes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    cooldown_hours: int | None = Field(default=None, ge=1)
    min_outcomes_to_pass: int | None = Field(default=None, ge=1)
    max_follow_ups_per_question: int | None = Field(default=None, ge=0, le=50)
    max_hints_per_question: int | None = Field(default=None, ge=0, le=10)
    supplementary_instructions: str | None = None
    security_response_policy: SecurityResponsePolicyLiteral | None = None
    security_max_consecutive_attempts: int | None = Field(default=None, ge=2, le=20)
    security_custom_refusal_en: str | None = Field(default=None, max_length=500)
    security_custom_refusal_vi: str | None = Field(default=None, max_length=500)
    security_incident_summary_enabled: bool | None = None


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
    # Resolved persona profile (preset merged with persona_profile_json
    # overrides) — teacher-only tone projection, never on a learner schema.
    persona_profile_resolved: PersonaProfileRead | None = None
    tts_voice: TtsVoiceLiteral | None = None
    min_outcomes_to_pass: int | None = None
    security_response_policy: SecurityResponsePolicyLiteral = "warn_and_continue"
    security_max_consecutive_attempts: int = 3
    security_custom_refusal_en: str | None = None
    security_custom_refusal_vi: str | None = None
    security_incident_summary_enabled: bool = True
    generation_run_id: UUID | None = None
    draft_question_count: int | None = None
    published_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def _populate_aggregates(cls, data: Any) -> Any:  # noqa: ANN401 -- ORM/Pydantic boundary
        """Count active questions when the caller passes the raw ORM model with
        the ``questions`` relationship already loaded. Skipped when data is a
        Pydantic model (nested in InterviewForAuthoringPublic) or when the
        relationship is unloaded — accessing an unloaded relationship under an
        AsyncSession raises MissingGreenlet, which previously surfaced as a 500 on
        GET /teacher/interview-configs/{config_id}. Callers that need the
        aggregate set must eager-load (selectinload) ``InterviewConfig.questions``.

        A sibling ``total_importance_weight`` sum lived here until 2026-09-01. No
        client ever read it — the teacher UI shows per-outcome weights from the
        outcome rows themselves (``WeightStepper``) — so it was removed rather
        than kept as an aggregate nobody renders.
        """
        if isinstance(data, BaseModel):
            return data
        from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

        try:
            insp = sa_inspect(data)
        except Exception:  # noqa: BLE001 — defensive: only treat real ORM instances
            return data
        unloaded: set[str] = getattr(insp, "unloaded", set()) if insp is not None else set()
        if "questions" not in unloaded:
            questions = getattr(data, "questions", None)
            if questions is not None and getattr(data, "draft_question_count", None) is None:
                data.draft_question_count = sum(1 for q in questions if q.deleted_at is None)
        # Resolve the effective persona profile (preset + any per-trait
        # overrides) for the teacher projection. Best-effort: a bad override can
        # never raise (profile_from_config is defensive) and a failure here just
        # leaves the field None rather than 500-ing the authoring GET.
        if getattr(data, "persona_profile_resolved", None) is None:
            try:
                from abridgeai.features.interviews.orchestrator.interviewer_identity import (  # noqa: PLC0415
                    identity_from_config,
                )
                from abridgeai.features.interviews.orchestrator.persona import (  # noqa: PLC0415
                    profile_from_config,
                )

                profile_json = getattr(data, "persona_profile_json", None)
                resolved = profile_from_config(
                    getattr(data, "persona", None),
                    profile_json,
                ).clamped()
                # Identity is resolved separately from tone but projected on the
                # same object, so the teacher UI reads one shape.
                data.persona_profile_resolved = {
                    **resolved.as_prompt_traits(),
                    "interviewer_role": identity_from_config(profile_json).role.value,
                }
            except Exception:  # noqa: BLE001 — projection is best-effort
                return data
        return data


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


class InterviewOutcomeUpdate(BaseModel):
    """Teacher-editable fields of an existing interview outcome."""

    model_config = ConfigDict(extra="forbid")

    outcome_text: str | None = None
    outcome_type: OutcomeTypeLiteral | None = None
    importance_weight: int | None = Field(default=None, ge=1, le=5)
    position: int | None = Field(default=None, ge=1)


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
    model_answer: str | None = None
    linked_outcome_id: UUID | None = None
    position: int | None = Field(default=None, ge=1)


class InterviewQuestionUpdate(BaseModel):
    """Teacher-editable fields of an existing interview question.

    Ownership, all-angle grouping, AI provenance, source attribution, audit,
    soft-delete, and embedding fields are server-owned and intentionally absent.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_text: str | None = None
    question_type: QuestionTypeLiteral | None = None
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    linked_outcome_id: UUID | None = None
    position: int | None = Field(default=None, ge=1)
    review_status: ReviewStatusLiteral | None = None


class InterviewQuestionDuplicateCheckRequest(BaseModel):
    """Body for ``POST /teacher/interviews/{id}/questions/check-duplicate``."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    exclude_question_id: UUID | None = None
    """Set when editing an existing question, so it is not matched against itself."""


class InterviewQuestionDuplicateCheck(BaseModel):
    """Advisory duplicate verdict for a proposed question.

    Purely informational — the teacher can save regardless. ``enabled`` is False
    when ``interview_dedup_enabled`` is off, and ``error`` is non-empty when the
    check could not run; in both cases ``is_duplicate`` is False, so a client must
    read those before telling the teacher the question is unique.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    is_duplicate: bool
    duplicate_of_id: UUID | None = None
    duplicate_of_text: str = ""
    rationale: str = ""
    error: str = ""


class InterviewQuestionAuthoring(InterviewQuestionPublic):
    """Authoring projection of :class:`InterviewQuestion`.

    Inherits the public projection and widens with the full review +
    pipeline + audit column set.
    """

    interview_config_id: UUID
    linked_outcome_id: UUID | None = None
    # Authoring-only logical-question grouping for all-angle generation.
    variant_group_id: UUID | None = None
    position: int | None = None
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    review_status: ReviewStatusLiteral
    ai_generated: bool
    source_refs_json: list[Any] = []
    source_module_ids: list[UUID] = []
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

    There is deliberately NO generation-mode discriminator. One pipeline
    (retrieval → generation → validation → persistence) serves every run;
    scoping is expressed by the fields themselves — ``source_module_ids`` /
    ``source_lesson_ids`` narrow the material, ``target_outcome_ids`` narrows
    the rubric criteria, ``focus_topics`` overrides the retrieval anchors. A
    ``mode`` field existed until 2026-08-30 but no stage ever read it, so the
    teacher's three-way choice ("topic" / "outcome-based" / "coverage") had
    identical behaviour; it was removed rather than left as a decorative dial.
    """

    model_config = ConfigDict(extra="forbid")

    course_id: UUID
    module_id: UUID
    source_module_ids: list[UUID] = []
    source_lesson_ids: list[UUID] = []
    question_count: int = Field(default=5, ge=1, le=50)
    # Retrieval anchors: when non-empty these REPLACE the KG-concept /
    # lesson-title anchors in ``ai.stages.retrieval.anchors``, so they steer
    # which chunks the questions are grounded in.
    focus_topics: list[str] = []
    # Rendered into the GENERATION prompt as a hard exclusion list
    # (``ai/stages/generation/prompts/user.j2``), mirroring the quiz stages.
    avoid_topics: list[str] = []
    persona: PersonaLiteral | None = None
    supplementary_instructions: str | None = None
    # Optional subset of the interview's OWN rubric outcomes
    # (:class:`InterviewOutcome`) to target. Empty = use every outcome (the
    # prior behaviour). When non-empty, the generation pipeline filters the
    # outcomes it feeds the ideation stage to just these ids, so a teacher can
    # focus a run on specific criteria. Flows into ``config_json`` via the
    # router's ``model_dump`` — the pipeline reads it back from there.
    target_outcome_ids: list[UUID] = []
    # Role-conditioned variant generation (Slice 21). ``None`` keeps the
    # legacy mixed type-mix; ``all_angles`` multiplies each logical question
    # into one variant per interviewer angle; ``role_only`` pins every
    # question to the config role's preferred type. Flows into
    # ``config_json`` verbatim via the service's ``model_dump`` — the
    # generation pipeline reads it back from there.
    variant_strategy: Literal["all_angles", "role_only"] | None = None


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


class SecuritySessionSummary(BaseModel):
    """Teacher-only redacted security metrics; no student response content."""

    assessment_count: int = 0
    blocked_attempt_count: int = 0
    repeated_attempt_count: int = 0
    output_leakage_prevented: int = 0
    security_fallback_rate: float = 0.0
    average_classification_latency_ms: float | None = None
    session_flagged: bool = False


class InterviewSessionSummary(BaseModel):
    """One row in the teacher's per-config attempts list.

    Teacher-only: surfaces the student identity + binary verdict so a teacher
    can pick an attempt to review (thesis p77 "Teachers can review…").
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    student_id: UUID
    student_name: str | None = None
    attempt_number: int
    status: SessionStatusLiteral
    input_mode: InputModeLiteral
    pass_verdict: bool | None = None
    started_at: datetime
    ended_at: datetime | None = None
    security_summary: SecuritySessionSummary | None = None


class InterviewSessionTeacherRead(BaseModel):
    """One row in a teacher's cross-config / cross-student sessions list.

    Widens :class:`InterviewSessionSummary` with the parent config's id +
    title so a single row is self-describing across multiple interview
    configs in the same course. Backs the course-wide "Assessments" tab
    and the per-student profile's interview-attempts section
    (student-dashboard brainstorm, 2026-07-11).
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    interview_config_id: UUID
    interview_config_title: str
    student_id: UUID
    student_name: str | None = None
    attempt_number: int
    status: SessionStatusLiteral
    input_mode: InputModeLiteral
    pass_verdict: bool | None = None
    started_at: datetime
    ended_at: datetime | None = None
    security_summary: SecuritySessionSummary | None = None


class InterviewTranscriptTurn(BaseModel):
    """One question/answer turn in a teacher-facing transcript view."""

    model_config = ConfigDict(from_attributes=True)

    role: Literal["user", "ai", "system"]
    question_prompt: str | None = None
    content_text: str | None = None
    has_audio: bool = False
    created_at: datetime


class InterviewTranscriptRead(BaseModel):
    """Full ordered transcript of a session for teacher remediation review."""

    session_id: UUID
    turns: list[InterviewTranscriptTurn] = []


class InterviewIntegrityEvent(BaseModel):
    """One FR-5.8 proctoring signal recorded during an interview session.

    Projection of :class:`features.interviews.models.AssessmentIntegrityEvent`
    rows scoped to ``assessment_kind='interview'``. Surfaced only in the
    teacher-facing gap-report review — never to the student. Mirrors the
    quiz-side ``QuizAttemptIntegrityEvent``.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_type: str
    severity: str
    metadata_json: dict[str, Any] = {}
    created_at: datetime


class InterviewIntegrityRead(BaseModel):
    """Teacher-facing integrity timeline for one interview session (FR-5.8)."""

    session_id: UUID
    events: list[InterviewIntegrityEvent] = []


# --------------------------------------------------------------------------- #
# Adaptive readiness (Slice 5) — advisory authoring analysis
# --------------------------------------------------------------------------- #


class AdaptiveModeRolloutStatus(BaseModel):
    """Whether the adaptive interviewer is live for each input mode.

    Reflects the deployment flag matrix (master + per-mode switches), NOT a
    per-config toggle — teachers see it read-only so they understand which
    modes will actually run the adaptive brain for their published interview.
    """

    text: bool
    hybrid: bool
    voice: bool


class ReadinessWarningRead(BaseModel):
    """One advisory readiness finding. ``code`` drives the localized UI copy."""

    code: str
    level: Literal["info", "warning"]
    affected_ids: list[str] = Field(default_factory=list)
    count: int = 0


class AdaptiveReadinessRead(BaseModel):
    """Adaptive-readiness report for the authoring workspace.

    ``warnings`` are ADVISORY — they never block publishing (only the existing
    hard requirements do). ``blocks_publish`` is always False here and is kept
    explicit so the client never mistakes a warning for a publish gate.
    """

    config_id: UUID
    warnings: list[ReadinessWarningRead] = Field(default_factory=list)
    rollout: AdaptiveModeRolloutStatus
    blocks_publish: bool = False


# --------------------------------------------------------------------------- #
# Course-scoped question bank (§QBank-1) — reusable interview questions
# --------------------------------------------------------------------------- #


class InterviewQuestionBankItemCreate(BaseModel):
    """Body for ``POST /teacher/courses/{course_id}/interview-question-bank``.

    Either supplied directly by a teacher or forwarded from an existing
    interview question ("add to bank"). ``source_config_id`` is provenance
    only.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_text: str
    question_type: QuestionTypeLiteral
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    source_config_id: UUID | None = None


class InterviewQuestionBankItemRead(BaseModel):
    """Read projection of a course-scoped interview question-bank item."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    course_id: UUID
    prompt_text: str
    question_type: QuestionTypeLiteral
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None
    variant_group_id: UUID | None = None
    source_config_id: UUID | None = None
    created_at: datetime
    updated_at: datetime


LogicalQuestionAngleLiteral = Literal[
    "technical", "system_design", "situational", "behavioral"
]


class InterviewQuestionBankLogicalGroupCreate(BaseModel):
    """Atomically add one complete, four-angle logical question to the bank."""

    model_config = ConfigDict(extra="forbid")

    items: list[InterviewQuestionBankItemCreate] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _validate_angles(self) -> InterviewQuestionBankLogicalGroupCreate:
        angles = [item.question_type for item in self.items]
        expected = {"technical", "system_design", "situational", "behavioral"}
        if set(angles) != expected or len(set(angles)) != 4:
            raise ValueError("logical question requires exactly one of each four angle types")
        return self


class InterviewQuestionBankSiblingCreate(InterviewQuestionBankItemCreate):
    """One missing angle to append to a bank singleton or partial group."""

    question_type: LogicalQuestionAngleLiteral


class InterviewQuestionBankImportRequest(BaseModel):
    """Import bank entries; selecting one grouped child imports every sibling."""

    model_config = ConfigDict(extra="forbid")

    item_ids: list[UUID] = Field(min_length=1)


class InterviewQuestionBankImportResult(BaseModel):
    """Questions copied into a target config by one atomic bank import."""

    created: list[InterviewQuestionAuthoring]
    imported_group_count: int = 0


class InterviewQuestionBankItemUpdate(BaseModel):
    """Partial edit of a bank item (management page). All fields optional."""

    model_config = ConfigDict(extra="forbid")

    prompt_text: str | None = None
    question_type: QuestionTypeLiteral | None = None
    difficulty: DifficultyLiteral | None = None
    model_answer: str | None = None


# Suppress unused-import warning — Decimal is exported in case downstream
# generation-config payloads carry numeric thresholds. Keep available.
_DECIMAL_AVAILABLE = Decimal


__all__ = [
    "AdaptiveModeRolloutStatus",
    "AdaptiveReadinessRead",
    "ConfigStatusLiteral",
    "DifficultyLiteral",
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
    "InterviewQuestionBankImportRequest",
    "InterviewQuestionBankImportResult",
    "InterviewQuestionBankItemCreate",
    "InterviewQuestionBankItemRead",
    "InterviewQuestionBankItemUpdate",
    "InterviewQuestionBankLogicalGroupCreate",
    "InterviewQuestionBankSiblingCreate",
    "LogicalQuestionAngleLiteral",
    "InterviewQuestionCreate",
    "InterviewSessionSummary",
    "InterviewSessionTeacherRead",
    "InterviewTranscriptRead",
    "InterviewTranscriptTurn",
    "ReviewStatusLiteral",
    "SecurityResponsePolicyLiteral",
    "SecuritySessionSummary",
]
