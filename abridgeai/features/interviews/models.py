"""Interviews feature ORM models.

Ports the legacy ``backend/app/models/assessment.py`` interview aggregate
to the feature-first layout. The schema source of truth is
``migrations/versions/0001_baseline_schema.py`` lines 773-902 (the seven
``interview_*`` tables + ``gap_reports``) and lines 1081-1096
(``assessment_integrity_events``).

Reconciliation directives applied (plan §266-585)
-------------------------------------------------

* §A13 — baseline DDL is canonical. Plan body for T6.1 lists 5 ORM
  classes (``InterviewConfig``, ``InterviewQuestion``,
  ``InterviewSession``, ``InterviewResponse``, ``GapReport``); baseline
  declares **9** interview-related tables. Port all 9. The plan-body
  ``InterviewResponse`` is reconciled to baseline's
  ``InterviewSessionMessage`` (per-utterance log) plus
  ``InterviewSessionQuestion`` (asked-question audit), and the rubric
  scoring is reconciled to a separate ``InterviewOutcome`` /
  ``InterviewOutcomeEvaluation`` pair (rubric criteria + per-criterion
  verdicts) rather than a flat ``rubric_scores JSONB`` column. This
  matches the baseline DDL verbatim.

* §A13 baseline-canon column-shape decisions on ``InterviewConfig``:
  - ``status VARCHAR(20)`` CHECK ``{'draft', 'published', 'archived'}``.
  - ``persona VARCHAR(20)`` CHECK ``{'strict', 'neutral', 'supportive'}``
    (baseline DDL line 784, NOT the plan-body's "{strict, friendly,
    neutral}" — baseline wins).
  - ``supported_modes`` was dropped (migration 0077): every session runs
    the native agent in one room, mic is an in-room toggle. The legacy
    ``input_mode`` column belongs to the *session* row and survives for
    analytics; new sessions are always written ``hybrid``.
  - ``min_outcomes_to_pass`` PRESERVED (legacy ORM line 199 + baseline
    line 781; plan body did not list it).
  - ``time_limit_minutes`` PRESERVED (baseline NOT ``time_limit_seconds``
    like quizzes — interview baseline uses minutes).
  - ``generation_run_id`` FK SET NULL preserved.

* §A13 baseline-canon on ``InterviewQuestion``:
  - ``question_type`` CHECK ``{'conceptual', 'behavioral', 'technical',
    'situational', 'system_design'}`` (baseline line 814) — the
    plan-body claim of ``{technical, behavioral, situational}`` is
    reconciled to the broader baseline set.
  - ``difficulty`` CHECK ``{'junior', 'mid_level', 'senior'}`` (baseline
    line 817) — per inherited wisdom note (plan §39 "difficulty →
    integer 1-5"), the integerization is NOT for this field; baseline
    keeps it as a level string. Plan §39's note targets a future
    refactor on quiz_questions, not interview_questions.
  - ``review_status`` CHECK ``{'pending', 'approved', 'edited',
    'rejected'}`` (baseline line 819) — same shape as
    ``quiz_questions.review_status``.
  - ``ai_generated BOOLEAN DEFAULT TRUE`` preserved.
  - ``source_refs_json JSONB`` — name kept verbatim from baseline
    (legacy ORM line 241). Plan §36 rename ``source_refs_json →
    source_refs`` was applied to ``quiz_questions`` only via migration
    0007; interview_questions retains the ``_json`` suffix per
    baseline. Generating a parallel migration for interview rename is
    out of scope for T6.1.

* §A13 baseline-canon on ``InterviewSession``:
  - ``status`` CHECK ``{'in_progress', 'completed', 'timed_out',
    'abandoned', 'failed'}`` (baseline line 835) — note 5 values, NOT
    the 3 the plan body listed.
  - ``input_mode`` CHECK ``{'voice', 'text', 'hybrid'}`` (baseline line
    837). Inherited wisdom §39 says "KEEP ``input_mode`` (not
    delivery_format)" — confirmed.
  - ``UNIQUE (interview_config_id, student_id, attempt_number)``
    enforces one row per attempt per student per config.
  - ``transcript_object_id`` + ``recording_object_id`` are nullable FKs
    to ``storage_objects`` for voice-mode forward-compat (Phase 7+
    wires uploads). Plan §6.1 explicitly forbids making these NOT NULL.

* §A13 baseline-canon on ``InterviewSessionQuestion``:
  - Composite UNIQUE ``(session_id, sequence_no)`` enforces ordered
    asked-questions.
  - ``CreatedAtMixin`` ONLY (baseline DDL line 858 has no
    ``updated_at``) — immutable record of "this question was asked".

* §A13 baseline-canon on ``InterviewSessionMessage``:
  - ``role`` CHECK ``{'ai', 'user', 'system'}`` (baseline line 866 — note
    ``'ai'`` NOT ``'assistant'`` like OpenAI chat schema; baseline
    chose the abridgeai-internal vocabulary).
  - ``audio_object_id`` nullable FK to ``storage_objects`` (voice
    forward-compat).
  - ``latency_ms`` CHECK ``>= 0`` preserved.
  - ``TimestampMixin`` (baseline gives both ``created_at`` AND
    ``updated_at`` at line 873-874, so the ORM mirrors that — plan body
    "CreatedAt only for InterviewResponse" is overridden by §A13).

* §A13 baseline-canon on ``InterviewOutcomeEvaluation``:
  - UNIQUE ``(session_id, outcome_id)`` — one verdict per outcome per
    session.
  - ``hidden_reasoning`` carries the LLM rationale (NEVER returned in
    public schemas — Phase 6.2 schemas enforce).

* §A13 baseline-canon on ``GapReport``:
  - ``source_quiz_attempt_id`` and ``source_interview_session_id`` are
    BOTH nullable FK SET NULL — a gap report is anchored to one or the
    other (or both, for a discrepancy report combining quiz vs
    interview). The application layer enforces "at least one source"
    invariant; no DDL CHECK enforces it.
  - ``module_id`` is nullable FK SET NULL — gap reports may span an
    entire course not a specific module.
  - Plan body's "discrepancy analysis between quiz score and interview
    rubric" lives in ``report_json`` JSONB; no dedicated columns.

* §A13 baseline-canon on ``AssessmentIntegrityEvent`` (baseline lines
  1081-1096):
  - Polymorphic parent: either ``quiz_attempt_id`` OR
    ``interview_session_id`` non-null (CHECK enforces XOR).
  - ``CreatedAtMixin`` only — append-only proctoring log.
  - ``event_type`` CHECK ``{'focus_lost', 'tab_switch', 'fullscreen_exit',
    'warning_issued', 'reconnect', 'disconnect'}``.
  - ``severity`` CHECK ``{'info', 'warning', 'critical'}``.
  - Baseline already places this table in the
    ``assessment_integrity_events`` lane; per inherited wisdom §44 it
    belongs with the interview aggregate (closer to integrity/proctoring
    concerns) rather than with the quizzes feature.

Mixin policy (§A13 + plan §6.1)
--------------------------------

Baseline ``SOFT_DELETE_TABLES`` listing (migration 0001 line 41-72)
includes the three authoring-side interview tables:
``interview_configs``, ``interview_outcomes``, ``interview_questions``.
These three receive ``UUIDPrimaryKeyMixin + TimestampMixin +
AuditedByMixin + SoftDeleteMixin`` — same stack as the quiz authoring
trio (T5.1).

Note: ``interview_configs`` already carries a ``created_by`` column in
baseline DDL (line 790). The migration 0001 ``_add_audit_columns(...,
with_created_by=False)`` skips re-adding it but still adds
``updated_by`` + ``deleted_at`` + ``deleted_by``. The ORM
``AuditedByMixin`` declares the ``created_by`` column once — both the
hand-rolled DDL ``created_by`` and the mixin-declared
``created_by`` resolve to the same column at metadata-merge time
because they share table + column name + FK target.

The session/runtime side is HISTORICAL RECORD (plan §6.1 explicit
"Must NOT add SoftDeleteMixin to InterviewSession or InterviewResponse").
Mixins:
- ``InterviewSession`` → ``UUIDPrimaryKeyMixin + TimestampMixin``.
- ``InterviewSessionMessage`` → ``UUIDPrimaryKeyMixin + TimestampMixin``
  (baseline DDL ships ``updated_at``).
- ``InterviewOutcomeEvaluation`` → ``UUIDPrimaryKeyMixin + TimestampMixin``.
- ``InterviewSessionQuestion`` → ``UUIDPrimaryKeyMixin + CreatedAtMixin``
  (immutable).
- ``GapReport`` → ``UUIDPrimaryKeyMixin + TimestampMixin``.
- ``AssessmentIntegrityEvent`` → ``UUIDPrimaryKeyMixin + CreatedAtMixin``.

Forward references
------------------
- ``InterviewConfig.generation_run_id`` → string FK to
  ``generation_runs.id``. ``GenerationRun`` is owned by
  ``features/quizzes/models.py`` (T5.1 stop-gap port until T5.x moves it
  to ``features/ai/models.py``); the FK resolves at flush time once that
  module is imported.
- ``courses.id``, ``modules.id``, ``users.id``, ``storage_objects.id``,
  ``quiz_attempts.id`` are owned by other features. Tests must
  side-effect-import the owning models module before flushing
  cross-feature FKs (T3.5/T5.1 recipe).

No alembic migration generated
------------------------------
Baseline 0001 already creates all 9 tables verbatim. T6.1 ports ORM
classes only; no DDL drift is introduced. ``alembic upgrade head``
remains a no-op for this task.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import (
    HALFVEC,  # type: ignore[import-not-found,import-untyped,unused-ignore]
)
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class InterviewConfig(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "interview_configs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_interview_configs_status",
        ),
        CheckConstraint(
            "persona IS NULL OR persona IN ('strict', 'neutral', 'supportive')",
            name="ck_interview_configs_persona",
        ),
        CheckConstraint(
            "security_response_policy IN ('continue_and_log', 'warn_and_continue', 'end_and_flag')",
            name="ck_interview_configs_security_response_policy",
        ),
        CheckConstraint(
            "security_max_consecutive_attempts BETWEEN 2 AND 20",
            name="ck_interview_configs_security_max_attempts",
        ),
        CheckConstraint(
            "max_follow_ups_per_question BETWEEN 0 AND 50",
            name="ck_interview_configs_max_follow_ups",
        ),
        CheckConstraint(
            "max_hints_per_question BETWEEN 0 AND 10",
            name="ck_interview_configs_max_hints",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    # URL slug for student-facing breadcrumb links
    # (``/courses/<course-slug>/learn/<item-slug>``). Unique per module over
    # live rows (``uq_interview_configs_module_slug``, migration 0086);
    # generated from the title at create time and IMMUTABLE once published —
    # see ``services/published_freeze.py``.
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    max_attempts: Mapped[int | None] = mapped_column(Integer)
    # FR-5.3 retake cooldown: minimum hours between the end of one attempt and
    # the start of the next. NULL / <=0 disables the gate. Mirrors
    # ``Quiz.cooldown_hours``; enforced in ``services/taking.start_session``.
    cooldown_hours: Mapped[int | None] = mapped_column(Integer)
    min_outcomes_to_pass: Mapped[int | None] = mapped_column(Integer)
    time_limit_minutes: Mapped[int | None] = mapped_column(Integer)
    # Per-question budgets the interviewer operates under. Defaults mirror the
    # orchestrator constants these columns replaced (decision.py), so an
    # unmigrated config behaves exactly as before.
    max_follow_ups_per_question: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("2")
    )
    max_hints_per_question: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    persona: Mapped[str | None] = mapped_column(String(20))
    # Optional per-trait persona overrides (Phase 3). NULL = use the ``persona``
    # preset as-is. When present, it is a partial dict of the PersonaProfile
    # traits (warmth/directness/verbosity/formality/ack_frequency + optional
    # opening_style) merged over the preset in ``profile_from``; unknown keys are
    # ignored and values clamped to [0,4]. ``persona`` stays the preset selector,
    # so this is purely additive tone-shaping and never reaches scoring.
    persona_profile_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # TTS voice for English narration/voice sessions: a Deepgram Aura-2 model id
    # (one voice per model, e.g. 'aura-2-thalia-en'). NULL = the deployment
    # default (settings.deepgram_tts_model_en). Vietnamese has no server TTS, so
    # this only affects English sessions. Validated against a curated allow-list
    # in the authoring schema (see narration.ALLOWED_TTS_VOICES).
    tts_voice: Mapped[str | None] = mapped_column(String(40))
    supplementary_instructions: Mapped[str | None] = mapped_column(Text)
    security_response_policy: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'warn_and_continue'")
    )
    security_max_consecutive_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    security_custom_refusal_en: Mapped[str | None] = mapped_column(Text)
    security_custom_refusal_vi: Mapped[str | None] = mapped_column(Text)
    security_incident_summary_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("generation_runs.id", ondelete="SET NULL"),
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    questions: Mapped[list[InterviewQuestion]] = relationship(
        back_populates="config",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    outcomes: Mapped[list[InterviewOutcome]] = relationship(
        back_populates="config",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class InterviewOutcome(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "interview_outcomes"
    __table_args__ = (
        UniqueConstraint("interview_config_id", "position", name="uq_interview_outcomes_position"),
        CheckConstraint(
            "outcome_type IN ('knowledge', 'skill', 'attitude')",
            name="ck_interview_outcomes_type",
        ),
        CheckConstraint(
            "importance_weight BETWEEN 1 AND 5",
            name="ck_interview_outcomes_importance",
        ),
    )

    interview_config_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_configs.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    outcome_text: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_type: Mapped[str] = mapped_column(String(20), nullable=False)
    importance_weight: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

    config: Mapped[InterviewConfig] = relationship(back_populates="outcomes")
    evaluations: Mapped[list[InterviewOutcomeEvaluation]] = relationship(
        back_populates="outcome",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class InterviewQuestion(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "interview_questions"
    __table_args__ = (
        UniqueConstraint("interview_config_id", "position", name="uq_interview_questions_position"),
        CheckConstraint(
            "question_type IN ('conceptual', 'behavioral', 'technical', "
            "'situational', 'system_design')",
            name="ck_interview_questions_question_type",
        ),
        CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('junior', 'mid_level', 'senior')",
            name="ck_interview_questions_difficulty",
        ),
        CheckConstraint(
            "review_status IN ('pending', 'approved', 'edited', 'rejected')",
            name="ck_interview_questions_review_status",
        ),
    )

    interview_config_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_configs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    linked_outcome_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_outcomes.id", ondelete="SET NULL"),
    )
    # Peers generated from the same logical problem share this server-assigned
    # identifier. NULL preserves legacy, manual, imported, and role-only rows.
    variant_group_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    position: Mapped[int | None] = mapped_column(Integer)
    question_type: Mapped[str] = mapped_column(String(30), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str | None] = mapped_column(String(20))
    embedding: Mapped[list[float] | None] = mapped_column(
        HALFVEC(3072),
        nullable=True,
    )
    """Semantic vector of ``prompt_text``, for duplicate detection in the bank.

    Same width and the same ``EmbeddingClient`` as ``document_chunks.embedding``
    (see migration 0063). ``NULL`` means "not embedded yet" — such rows are skipped
    by the duplicate shortlist rather than treated as dissimilar, so a missing
    vector degrades detection instead of producing a wrong verdict.
    """
    model_answer: Mapped[str | None] = mapped_column(Text)
    """Teacher-facing reference answer. Authoring aid only — never exposed
    to learners and never used to auto-grade (scoring stays rubric-based
    against :class:`InterviewOutcome` criteria)."""
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    ai_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))
    source_refs_json: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    source_module_ids: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    """Module attribution for the Question Bank grouping. Populated from the
    generation run's ``source_module_ids`` (falling back to the config's own
    module). A question drawing from 2+ modules lands in the multi-module
    section of the bank."""
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    config: Mapped[InterviewConfig] = relationship(back_populates="questions")


class InterviewSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "interview_sessions"
    __table_args__ = (
        UniqueConstraint(
            "interview_config_id",
            "student_id",
            "attempt_number",
            name="uq_interview_sessions_number",
        ),
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'timed_out', 'abandoned', 'failed')",
            name="ck_interview_sessions_status",
        ),
        CheckConstraint(
            "input_mode IN ('voice', 'text', 'hybrid')",
            name="ck_interview_sessions_input_mode",
        ),
        CheckConstraint(
            "onboarding_stage IN ('identity_check', 'audio_check', "
            "'language_check', 'preparation', 'readiness', 'completed')",
            name="ck_interview_sessions_onboarding_stage",
        ),
        CheckConstraint(
            "interview_language IN ('en', 'vi')",
            name="ck_interview_sessions_language",
        ),
    )

    interview_config_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_configs.id", ondelete="CASCADE"),
        nullable=False,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'in_progress'")
    )
    input_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    livekit_room_name: Mapped[str | None] = mapped_column(String(255))
    livekit_session_ref: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    assessment_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    onboarding_stage: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'identity_check'")
    )
    interview_language: Mapped[str] = mapped_column(
        String(5), nullable=False, server_default=text("'en'")
    )
    # Session-scoped conversational name. Set when the candidate corrects the
    # profile-derived name during identity_check ("that isn't my name"). Never
    # written back to the user's profile — it only affects how the interviewer
    # addresses them within this session.
    preferred_name: Mapped[str | None] = mapped_column(String(60))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resume_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transcript_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
    )
    recording_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
    )
    pass_verdict: Mapped[bool | None] = mapped_column(Boolean)
    internal_summary_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    security_policy_version: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'2026-07-19'")
    )
    security_rules_version: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'1.0.0'")
    )
    security_prompt_version: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'1.0.0'")
    )
    output_guard_version: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'1.0.0'")
    )
    session_security_flagged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )

    questions: Mapped[list[InterviewSessionQuestion]] = relationship(
        back_populates="session",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    messages: Mapped[list[InterviewSessionMessage]] = relationship(
        back_populates="session",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    gap_report: Mapped[GapReport | None] = relationship(
        back_populates="session",
        cascade="save-update, merge, refresh-expire, expunge",
        uselist=False,
        foreign_keys="[GapReport.source_interview_session_id]",
    )
    integrity_events: Mapped[list[AssessmentIntegrityEvent]] = relationship(
        back_populates="session",
        cascade="save-update, merge, refresh-expire, expunge",
        foreign_keys="[AssessmentIntegrityEvent.interview_session_id]",
    )
    security_events: Mapped[list[InterviewSecurityEvent]] = relationship(
        back_populates="session",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class InterviewSessionQuestion(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "interview_session_questions"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "sequence_no", name="uq_interview_session_questions_sequence"
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    interview_question_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_questions.id", ondelete="SET NULL"),
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    session: Mapped[InterviewSession] = relationship(back_populates="questions")


class InterviewSessionMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "interview_session_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('ai', 'user', 'system')",
            name="ck_interview_session_messages_role",
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_question_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_session_questions.id", ondelete="SET NULL"),
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content_text: Mapped[str | None] = mapped_column(Text)
    audio_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    session: Mapped[InterviewSession] = relationship(back_populates="messages")


class InterviewOutcomeEvaluation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "interview_outcome_evaluations"
    __table_args__ = (
        UniqueConstraint("session_id", "outcome_id", name="uq_interview_outcome_evaluations"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    outcome_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_outcomes.id", ondelete="CASCADE"),
        nullable=False,
    )
    verdict_met: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hidden_reasoning: Mapped[str | None] = mapped_column(Text)
    evidence_excerpt: Mapped[str | None] = mapped_column(Text)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    outcome: Mapped[InterviewOutcome] = relationship(back_populates="evaluations")


class GapReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "gap_reports"

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
    )
    module_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="SET NULL"),
    )
    source_quiz_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_attempts.id", ondelete="SET NULL"),
    )
    source_interview_session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="SET NULL"),
    )
    student_summary: Mapped[str | None] = mapped_column(Text)
    teacher_summary: Mapped[str | None] = mapped_column(Text)
    report_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    session: Mapped[InterviewSession | None] = relationship(
        back_populates="gap_report",
        foreign_keys=[source_interview_session_id],
    )


class AssessmentIntegrityEvent(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "assessment_integrity_events"
    __table_args__ = (
        CheckConstraint(
            "assessment_kind IN ('quiz', 'interview')",
            name="ck_assessment_integrity_kind",
        ),
        CheckConstraint(
            "event_type IN ('focus_lost', 'tab_switch', 'fullscreen_exit', "
            "'warning_issued', 'reconnect', 'disconnect')",
            name="ck_assessment_integrity_event_type",
        ),
        CheckConstraint(
            "severity IN ('info', 'warning', 'critical')",
            name="ck_assessment_integrity_severity",
        ),
        CheckConstraint(
            "(assessment_kind = 'quiz' AND quiz_attempt_id IS NOT NULL "
            "AND interview_session_id IS NULL) OR "
            "(assessment_kind = 'interview' AND interview_session_id IS NOT NULL "
            "AND quiz_attempt_id IS NULL)",
            name="ck_assessment_integrity_parent_ref",
        ),
    )

    assessment_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    quiz_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_attempts.id", ondelete="CASCADE"),
    )
    interview_session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    session: Mapped[InterviewSession | None] = relationship(
        back_populates="integrity_events",
        foreign_keys=[interview_session_id],
    )


class InterviewSecurityEvent(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Redacted, idempotent prompt-injection/output-guard audit event."""

    __tablename__ = "interview_security_events"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_id", "event_type", name="uq_interview_security_event"),
        CheckConstraint(
            "event_type IN ('interview.security.assessed', "
            "'interview.security.blocked', "
            "'interview.security.output_leakage_blocked', "
            "'interview.security.repeated_attempt', "
            "'interview.security.session_flagged')",
            name="ck_interview_security_events_type",
        ),
        CheckConstraint(
            "confidence_band IN ('low', 'medium', 'high')",
            name="ck_interview_security_events_confidence_band",
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_band: Mapped[str] = mapped_column(String(10), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    fallback_status: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    normalized_fingerprint: Mapped[str | None] = mapped_column(String(64))
    protected_content_category: Mapped[str | None] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    rules_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    output_guard_version: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)

    session: Mapped[InterviewSession] = relationship(back_populates="security_events")


class InterviewRuntimeState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Persistent adaptive-interviewer runtime state (Phase 1).

    One row per :class:`InterviewSession` (1:1, UNIQUE ``session_id``). Holds
    the stateful orchestrator's view of the interview: phase, per-outcome
    coverage, asked/skipped/completed ledgers, follow-up counters, the last
    classified intent + answer analysis, and candidate signals. The entire
    structured payload lives in ``state_json`` (JSONB) so the shape can evolve
    without a migration per field; only the hot invariants that need DB-level
    guarantees get real columns:

    * ``phase`` — surfaced for querying/observability without unpacking JSON.
    * ``state_version`` — monotonic optimistic-lock counter. Every committed
      advancement bumps it; a stale update (client/LiveKit retry carrying an
      older version) is rejected so the interview never advances twice.
    * ``last_turn_idempotency_key`` — the idempotency key of the most recently
      processed student turn. A duplicate REST/LiveKit callback carrying the
      same key is a no-op replay, not a second advancement.

    Compatibility: this table is ADDITIVE and lazily initialised. Sessions that
    predate it (or run with ``adaptive_interviewer_enabled=false``) simply have
    no row; the sequential path never reads it. Nothing breaks if it is absent.
    """

    __tablename__ = "interview_runtime_states"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_interview_runtime_states_session"),
        CheckConstraint(
            "phase IN ('opening', 'warmup', 'core', 'deep_probe', 'closing', 'completed')",
            name="ck_interview_runtime_states_phase",
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    phase: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'opening'"))
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    last_turn_idempotency_key: Mapped[str | None] = mapped_column(String(255))
    state_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )


class InterviewQuestionBankItem(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Course-scoped reusable interview question (§QBank-1).

    A shared pool a teacher can add generated/authored interview questions
    into, then copy from into any interview config in the same course.
    Copy semantics: importing from the bank creates a fresh
    :class:`InterviewQuestion` row (edits diverge; deleting the bank item
    does not touch already-imported questions, and vice versa).

    Scope is course-level only. This mirrors the copyable subset of
    ``InterviewQuestion`` columns; it deliberately does NOT carry
    per-config state (position, review_status, linked_outcome_id, audit of
    the config question) since those are meaningless outside a config.
    """

    __tablename__ = "interview_question_bank_items"
    __table_args__ = (
        CheckConstraint(
            "question_type IN ('conceptual', 'behavioral', 'technical', "
            "'situational', 'system_design')",
            name="ck_iq_bank_question_type",
        ),
        CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('junior', 'mid_level', 'senior')",
            name="ck_iq_bank_difficulty",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(30), nullable=False)
    difficulty: Mapped[str | None] = mapped_column(String(20))
    model_answer: Mapped[str | None] = mapped_column(Text)
    # Free-text tags for filtering the bank (topic labels). Optional.
    tags_json: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # Provenance: which config this question was first added from (audit
    # only; SET NULL so deleting the source config keeps the bank item).
    source_config_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interview_configs.id", ondelete="SET NULL"),
    )


__all__ = [
    "AssessmentIntegrityEvent",
    "GapReport",
    "InterviewConfig",
    "InterviewOutcome",
    "InterviewOutcomeEvaluation",
    "InterviewQuestion",
    "InterviewQuestionBankItem",
    "InterviewRuntimeState",
    "InterviewSecurityEvent",
    "InterviewSession",
    "InterviewSessionMessage",
    "InterviewSessionQuestion",
]
