"""Quizzes feature ORM models.

Ports the legacy ``backend/app/models/assessment.py`` quiz aggregate to
the feature-first layout. The schema source of truth is
``migrations/versions/0001_baseline_schema.py`` plus migration 0007
(which applies the §C1 renames + nullable-T_exp flip + ``mcq`` →
``multiple_choice`` data migration that T5.1 demands).

Reconciliation directives applied (plan §420-444):

* §A13 — baseline DDL is canonical. Column names, types, defaults and
  CHECK constraints copy verbatim from baseline 0001 (lines 630-753)
  and migration 0007. Plan-body fabrications (``Quiz.lesson_id``,
  ``Quiz.passing_score INT``, ``Quiz.is_randomized``,
  ``Quiz.generation_mode`` column, ``Quiz.title VARCHAR(255)`` already
  match) are reconciled against baseline before mapping. Notable
  baseline-canon decisions:
  - ``passing_score_percent NUMERIC(5,2)`` (NOT ``passing_score INT``).
  - ``shuffle_questions`` / ``shuffle_options`` (NOT ``is_randomized``).
  - ``time_limit_seconds`` keeps baseline name.
  - ``allow_retakes`` / ``cooldown_hours`` / ``max_attempts`` are
    declared but NOT enforced here — Phase 7.5 / T7.5.11 owns
    enforcement (§C2).
  - ``Quiz.published_at`` / ``Quiz.created_by`` / ``Quiz.show_hints`` /
    ``Quiz.initial_ef`` / ``Quiz.min_ef_for_unlock`` /
    ``Quiz.coverage_threshold`` / ``Quiz.reminders_enabled`` /
    ``Quiz.generation_instructions`` / ``Quiz.generation_run_id`` are
    PRESERVED (plan §5363 listed only a subset).

* §C1 — column renames + domain rename:
  - ``QuizQuestion.expected_response_time_ms`` (was
    ``expected_response_ms``) maps as ``Integer | None`` — nullable in
    draft; T7.5.9 publish gate enforces NOT NULL at publish time.
  - ``QuizQuestion.source_refs`` (was ``source_refs_json``) maps as
    JSONB carrying chunk_ids for KG remediation (§C5).
  - ``QuizQuestion.hint_text`` is ``Text`` (unbounded) per §C1 (NOT
    ``String(500)`` from plan body).
  - ``QuizQuestion.question_type`` CHECK enforces
    ``{multiple_choice, true_false, short_answer, fill_blank, code}``
    after migration 0007 renames legacy ``mcq`` rows.
  - ``QuizAttemptAnswer.t_actual_ms`` (was ``response_time_ms``) maps
    per plan §5371.

* §C3 — ``QuizSourceLesson`` PRESERVED for the
  ``assert_ready_generation_sources`` publish gate (T5.13). Composite
  PK ``(quiz_id, lesson_id)`` + ``CreatedAtMixin`` only — pure link.

* §C4 — ``QuizAttempt.idempotency_key`` UNIQUE column PRESERVED for
  retry-safe attempt creation (FE depends on this).

* §C5 — Quiz Bank refactor (Question + QuestionBank tables,
  QuizQuestion-as-link-table) is OUT OF SCOPE for T5.1. Lands in T5.0
  follow-up. T5.1 ports the legacy ``quiz_questions`` shape verbatim.

* No ``Quiz.mode`` field (plan §5375 + §5381). Every quiz is SR; SR
  state lives in ``StudentQuizCardState`` (Phase 7.5 / T7.5.1).

* ``StudentQuizCardState`` lives in ``features/quizzes`` baseline-side
  (table ``student_quiz_card_state`` is in ``SOFT_DELETE_TABLES``? —
  actually it is NOT; baseline keeps it lean) but is intentionally
  NOT declared here. T7.5.1 ports it alongside the SR scheduler so
  the FK to ``QuizQuestion`` is set up with the algorithm.

Mixin policy (§A13 + draft "Soft-delete eligibility")
------------------------------------------------------
Baseline ``SOFT_DELETE_TABLES`` listing for the quiz aggregate:
``quizzes``, ``quiz_questions``, ``quiz_question_options``. These
three receive ``UUIDPrimaryKeyMixin`` + ``TimestampMixin`` +
``AuditedByMixin`` + ``SoftDeleteMixin``.

``quiz_attempts`` is NOT in the list per plan §5378 (attempts are
historical record, not content) — keeps ``UUIDPrimaryKeyMixin`` +
``TimestampMixin``.

``quiz_attempt_answers`` is NOT in the list either — answers are
immutable per attempt. Baseline DDL nevertheless gives the table both
``created_at`` and ``updated_at`` columns, so the ORM uses
``TimestampMixin`` (NOT ``CreatedAtMixin``) to mirror the actual DDL
shape. The plan body's "CreatedAt only" instruction is overridden by
§A13 (baseline > spec).

``quiz_question_revisions`` and ``quiz_source_lessons`` are
append-only link / log tables — ``UUIDPrimaryKeyMixin`` +
``CreatedAtMixin`` (revisions, with own ``created_by``) and
``CreatedAtMixin`` only (source lessons).

Forward references
------------------
``GenerationRun`` / ``AIModelCall`` continue to live in
``backend/app/models/ai_processing.py`` until T5.x ports them to
``features/ai/models.py``. ``Quiz.generation_run_id`` therefore uses a
string FK reference to ``generation_runs.id``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
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


class Quiz(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "quizzes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_quizzes_status",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="NO ACTION"),
        nullable=False,
    )
    module_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("modules.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    time_limit_seconds: Mapped[int | None] = mapped_column(Integer)
    passing_score_percent: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), nullable=False, server_default=text("70.00")
    )
    allow_retakes: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )
    max_attempts: Mapped[int | None] = mapped_column(Integer)
    cooldown_hours: Mapped[int | None] = mapped_column(Integer)
    shuffle_questions: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    shuffle_options: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    show_hints: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))
    initial_ef: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    min_ef_for_unlock: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    coverage_threshold: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    reminders_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    generation_instructions: Mapped[str | None] = mapped_column(Text)
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("generation_runs.id", ondelete="SET NULL"),
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    questions: Mapped[list[QuizQuestion]] = relationship(
        back_populates="quiz",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    attempts: Mapped[list[QuizAttempt]] = relationship(
        back_populates="quiz",
        cascade="save-update, merge, refresh-expire, expunge",
    )
    source_lessons: Mapped[list[QuizSourceLesson]] = relationship(
        back_populates="quiz",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class QuizSourceLesson(CreatedAtMixin, Base):
    __tablename__ = "quiz_source_lessons"

    quiz_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
        primary_key=True,
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("lessons.id", ondelete="NO ACTION"),
        primary_key=True,
    )

    quiz: Mapped[Quiz] = relationship(back_populates="source_lessons")


class QuizQuestion(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "quiz_questions"
    __table_args__ = (
        UniqueConstraint("quiz_id", "position", name="uq_quiz_questions_position"),
        CheckConstraint(
            "question_type IN ('multiple_choice', 'true_false', "
            "'short_answer', 'fill_blank', 'code')",
            name="ck_quiz_questions_question_type",
        ),
        CheckConstraint(
            "review_status IN ('pending', 'approved', 'edited', 'rejected')",
            name="ck_quiz_questions_review_status",
        ),
        CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('easy', 'medium', 'hard')",
            name="ck_quiz_questions_difficulty",
        ),
        CheckConstraint(
            "bloom_level IS NULL OR bloom_level IN "
            "('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create')",
            name="ck_quiz_questions_bloom_level",
        ),
    )

    quiz_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    question_type: Mapped[str] = mapped_column(String(30), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    hint_text: Mapped[str | None] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text)
    difficulty: Mapped[str | None] = mapped_column(String(20))
    bloom_level: Mapped[str | None] = mapped_column(String(20))
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    expected_response_time_ms: Mapped[int | None] = mapped_column(Integer)
    expected_ef_ceiling: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    source_refs: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    original_generated_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    quiz: Mapped[Quiz] = relationship(back_populates="questions")
    revisions: Mapped[list[QuizQuestionRevision]] = relationship(
        back_populates="question",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class QuizQuestionOption(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    __tablename__ = "quiz_question_options"
    __table_args__ = (
        UniqueConstraint("question_id", "option_key", name="uq_quiz_question_options_key"),
        UniqueConstraint("question_id", "position", name="uq_quiz_question_options_position"),
    )

    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="NO ACTION"),
        nullable=False,
    )
    option_key: Mapped[str] = mapped_column(String(5), nullable=False)
    option_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    position: Mapped[int] = mapped_column(Integer, nullable=False)


class QuizQuestionRevision(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "quiz_question_revisions"
    __table_args__ = (
        UniqueConstraint("question_id", "revision_no", name="uq_quiz_question_revisions_number"),
        CheckConstraint(
            "source_kind IN ('ai', 'teacher')",
            name="ck_quiz_question_revisions_source_kind",
        ),
    )

    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    source_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    question: Mapped[QuizQuestion] = relationship(back_populates="revisions")


class QuizAttempt(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "quiz_attempts"
    __table_args__ = (
        UniqueConstraint("quiz_id", "student_id", "attempt_number", name="uq_quiz_attempts_number"),
        CheckConstraint(
            "status IN ('in_progress', 'submitted', 'graded', 'abandoned', 'expired')",
            name="ck_quiz_attempts_status",
        ),
    )

    quiz_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quizzes.id", ondelete="NO ACTION"),
        nullable=False,
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'in_progress'")
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    time_taken_seconds: Mapped[int | None] = mapped_column(Integer)
    score_points: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    score_percent: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    passed: Mapped[bool | None] = mapped_column(Boolean)
    idempotency_key: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), unique=True)

    quiz: Mapped[Quiz] = relationship(back_populates="attempts")
    answers: Mapped[list[QuizAttemptAnswer]] = relationship(
        back_populates="attempt",
        cascade="save-update, merge, refresh-expire, expunge",
    )


class QuizAttemptAnswer(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "quiz_attempt_answers"
    __table_args__ = (
        UniqueConstraint("attempt_id", "question_id", name="uq_quiz_attempt_answers_question"),
    )

    attempt_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_attempts.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="NO ACTION"),
        nullable=False,
    )
    selected_option_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_question_options.id", ondelete="SET NULL"),
    )
    answer_text: Mapped[str | None] = mapped_column(Text)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hint_used: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    t_actual_ms: Mapped[int | None] = mapped_column(Integer)
    points_awarded: Mapped[Decimal] = mapped_column(
        Numeric(8, 2), nullable=False, server_default=text("0")
    )
    sm2_q_value: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    ef_before: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    ef_after: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    interval_before_days: Mapped[int | None] = mapped_column(Integer)
    interval_after_days: Mapped[int | None] = mapped_column(Integer)

    attempt: Mapped[QuizAttempt] = relationship(back_populates="answers")


__all__ = [
    "Quiz",
    "QuizAttempt",
    "QuizAttemptAnswer",
    "QuizQuestion",
    "QuizQuestionOption",
    "QuizQuestionRevision",
    "QuizSourceLesson",
]
