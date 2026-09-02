"""ORM models for the course-scoped curated Quiz Question Bank."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    AuditedByMixin,
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class QuizQuestionBankItem(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """Portable question content reusable within one course.

    Quiz placement and review state intentionally stay on ``QuizQuestion``.
    Importing a bank item creates a snapshot rather than a live link.
    """

    __tablename__ = "quiz_question_bank_items"
    __table_args__ = (
        CheckConstraint(
            "question_type IN ('multiple_choice', 'true_false', "
            "'short_answer', 'fill_blank', 'code', 'numerical', "
            "'matching', 'ordering')",
            name="ck_quiz_bank_question_type",
        ),
        CheckConstraint(
            "status IN ('draft', 'approved', 'archived')",
            name="ck_quiz_bank_status",
        ),
        CheckConstraint(
            "difficulty IS NULL OR difficulty IN ('easy', 'medium', 'hard')",
            name="ck_quiz_bank_difficulty",
        ),
        CheckConstraint(
            "bloom_level IS NULL OR bloom_level IN "
            "('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create')",
            name="ck_quiz_bank_bloom_level",
        ),
    )

    course_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    learning_outcome_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("course_learning_outcomes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_question_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question_type: Mapped[str] = mapped_column(String(30), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    hint_text: Mapped[str | None] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text)
    difficulty: Mapped[str | None] = mapped_column(String(20))
    bloom_level: Mapped[str | None] = mapped_column(String(20))
    expected_response_time_ms: Mapped[int | None] = mapped_column(Integer)
    expected_ef_ceiling: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    source_refs: Mapped[Any] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    original_generated_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    prompt_format: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'plain'")
    )
    hint_format: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'plain'")
    )
    explanation_format: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'plain'")
    )
    single_answer: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    answer_numbering: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'abc'")
    )
    numeric_answer: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    numeric_tolerance: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6), server_default=text("0")
    )
    match_pairs: Mapped[Any | None] = mapped_column(JSONB)
    match_distractors: Mapped[Any | None] = mapped_column(JSONB)
    ordering_sequence: Mapped[Any | None] = mapped_column(JSONB)
    category_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("question_categories.id", ondelete="SET NULL"),
        nullable=True,
    )


class QuizQuestionBankOption(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    __tablename__ = "quiz_question_bank_options"
    __table_args__ = (
        UniqueConstraint("bank_item_id", "option_key", name="uq_quiz_bank_options_key"),
        UniqueConstraint("bank_item_id", "position", name="uq_quiz_bank_options_position"),
    )

    bank_item_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_question_bank_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    option_key: Mapped[str] = mapped_column(String(5), nullable=False)
    option_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    option_format: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default=text("'plain'")
    )
    grade_fraction: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    feedback_text: Mapped[str | None] = mapped_column(Text)
    feedback_format: Mapped[str | None] = mapped_column(String(16))


__all__ = ["QuizQuestionBankItem", "QuizQuestionBankOption"]
