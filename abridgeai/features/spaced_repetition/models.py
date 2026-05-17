from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    PGUUID,
    Base,
    CreatedAtMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class StudentCardState(TimestampMixin, Base):
    __tablename__ = "student_card_state"
    __table_args__ = (
        CheckConstraint("ef >= 1.3", name="ck_student_card_state_ef_floor"),
        CheckConstraint(
            "interval_days >= 0",
            name="ck_student_card_state_interval_nonneg",
        ),
        CheckConstraint(
            "repetition_count >= 0",
            name="ck_student_card_state_repetition_nonneg",
        ),
        CheckConstraint(
            "last_q IS NULL OR (last_q BETWEEN 0 AND 5)",
            name="ck_student_card_state_last_q_range",
        ),
        CheckConstraint(
            "total_reviews >= 0",
            name="ck_student_card_state_total_reviews_nonneg",
        ),
        Index(
            "ix_student_card_state_due_at",
            "student_id",
            "due_at",
        ),
        Index("ix_student_card_state_question_id", "question_id"),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="CASCADE"),
        primary_key=True,
    )

    ef: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False, server_default=text("2.5"))
    interval_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    repetition_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    last_q: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    calibration_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("TRUE")
    )
    total_reviews: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))


class CardReview(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "card_reviews"
    __table_args__ = (
        CheckConstraint(
            "q_derived BETWEEN 0 AND 5",
            name="ck_card_reviews_q_derived_range",
        ),
        CheckConstraint(
            "t_actual_ms >= 0",
            name="ck_card_reviews_t_actual_nonneg",
        ),
        CheckConstraint(
            "t_exp_ms >= 0",
            name="ck_card_reviews_t_exp_nonneg",
        ),
        CheckConstraint("ef_before >= 1.3", name="ck_card_reviews_ef_before_floor"),
        CheckConstraint("ef_after >= 1.3", name="ck_card_reviews_ef_after_floor"),
        CheckConstraint(
            "interval_before >= 0",
            name="ck_card_reviews_interval_before_nonneg",
        ),
        CheckConstraint(
            "interval_after >= 0",
            name="ck_card_reviews_interval_after_nonneg",
        ),
        CheckConstraint(
            "n_before >= 0",
            name="ck_card_reviews_n_before_nonneg",
        ),
        CheckConstraint(
            "n_after >= 0",
            name="ck_card_reviews_n_after_nonneg",
        ),
        Index(
            "ix_card_reviews_student_created",
            "student_id",
            "created_at",
        ),
        Index(
            "ix_card_reviews_question_created",
            "question_id",
            "created_at",
        ),
    )

    student_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_questions.id", ondelete="CASCADE"),
        nullable=False,
    )
    quiz_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("quiz_attempts.id", ondelete="SET NULL"),
        nullable=True,
    )

    t_actual_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    t_exp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    rho: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hint_used: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("FALSE"))
    q_derived: Mapped[int] = mapped_column(Integer, nullable=False)

    ef_before: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    ef_after: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    interval_before: Mapped[int] = mapped_column(Integer, nullable=False)
    interval_after: Mapped[int] = mapped_column(Integer, nullable=False)
    n_before: Mapped[int] = mapped_column(Integer, nullable=False)
    n_after: Mapped[int] = mapped_column(Integer, nullable=False)


__all__ = ["CardReview", "StudentCardState"]
