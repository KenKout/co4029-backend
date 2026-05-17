"""Spaced repetition tables (T7.5.1)

Revision ID: 0010_spaced_repetition
Revises: 0009_progress_tracking
Create Date: 2026-05-17 18:00:00.000000

T7.5.1 introduces two tables for the new
``abridgeai/features/spaced_repetition/`` slice:

* ``student_card_state`` — per-(student, question) SM-2 hybrid state.
  Composite primary key ``(student_id, question_id)`` plus
  ``TimestampMixin``. Distinct from the legacy
  ``student_quiz_card_state`` table; per Reconciliation §A13 we leave
  the legacy table untouched and add this new shape alongside it.
* ``card_reviews`` — append-only review event log
  (``UUIDPrimaryKey + CreatedAt`` only).

Both tables match the SQLAlchemy models in
``abridgeai/features/spaced_repetition/models.py`` exactly. Constraint
and index names match the model ``__table_args__`` so ORM creates
round-trip with this DDL.

Downgrade drops both tables.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0010_spaced_repetition"
down_revision = "0009_progress_tracking"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "student_card_state",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "student_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "question_id",
            UUID(as_uuid=True),
            sa.ForeignKey("quiz_questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ef",
            sa.Numeric(4, 3),
            nullable=False,
            server_default=sa.text("2.5"),
        ),
        sa.Column(
            "interval_days",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "repetition_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "due_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("last_q", sa.Integer(), nullable=True),
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "calibration_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column(
            "total_reviews",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.PrimaryKeyConstraint(
            "student_id", "question_id", name="pk_student_card_state"
        ),
        sa.CheckConstraint("ef >= 1.3", name="ck_student_card_state_ef_floor"),
        sa.CheckConstraint(
            "interval_days >= 0", name="ck_student_card_state_interval_nonneg"
        ),
        sa.CheckConstraint(
            "repetition_count >= 0",
            name="ck_student_card_state_repetition_nonneg",
        ),
        sa.CheckConstraint(
            "last_q IS NULL OR (last_q BETWEEN 0 AND 5)",
            name="ck_student_card_state_last_q_range",
        ),
        sa.CheckConstraint(
            "total_reviews >= 0",
            name="ck_student_card_state_total_reviews_nonneg",
        ),
    )
    op.create_index(
        "ix_student_card_state_due_at",
        "student_card_state",
        ["student_id", "due_at"],
    )
    op.create_index(
        "ix_student_card_state_question_id",
        "student_card_state",
        ["question_id"],
    )

    op.create_table(
        "card_reviews",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("uuid_generate_v4()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "student_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "question_id",
            UUID(as_uuid=True),
            sa.ForeignKey("quiz_questions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "quiz_attempt_id",
            UUID(as_uuid=True),
            sa.ForeignKey("quiz_attempts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("t_actual_ms", sa.Integer(), nullable=False),
        sa.Column("t_exp_ms", sa.Integer(), nullable=False),
        sa.Column("rho", sa.Numeric(8, 4), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column(
            "hint_used",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column("q_derived", sa.Integer(), nullable=False),
        sa.Column("ef_before", sa.Numeric(4, 3), nullable=False),
        sa.Column("ef_after", sa.Numeric(4, 3), nullable=False),
        sa.Column("interval_before", sa.Integer(), nullable=False),
        sa.Column("interval_after", sa.Integer(), nullable=False),
        sa.Column("n_before", sa.Integer(), nullable=False),
        sa.Column("n_after", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "q_derived BETWEEN 0 AND 5",
            name="ck_card_reviews_q_derived_range",
        ),
        sa.CheckConstraint(
            "t_actual_ms >= 0", name="ck_card_reviews_t_actual_nonneg"
        ),
        sa.CheckConstraint("t_exp_ms >= 0", name="ck_card_reviews_t_exp_nonneg"),
        sa.CheckConstraint("ef_before >= 1.3", name="ck_card_reviews_ef_before_floor"),
        sa.CheckConstraint("ef_after >= 1.3", name="ck_card_reviews_ef_after_floor"),
        sa.CheckConstraint(
            "interval_before >= 0",
            name="ck_card_reviews_interval_before_nonneg",
        ),
        sa.CheckConstraint(
            "interval_after >= 0",
            name="ck_card_reviews_interval_after_nonneg",
        ),
        sa.CheckConstraint(
            "n_before >= 0", name="ck_card_reviews_n_before_nonneg"
        ),
        sa.CheckConstraint("n_after >= 0", name="ck_card_reviews_n_after_nonneg"),
    )
    op.create_index(
        "ix_card_reviews_student_created",
        "card_reviews",
        ["student_id", "created_at"],
    )
    op.create_index(
        "ix_card_reviews_question_created",
        "card_reviews",
        ["question_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_card_reviews_question_created", table_name="card_reviews")
    op.drop_index("ix_card_reviews_student_created", table_name="card_reviews")
    op.drop_table("card_reviews")

    op.drop_index(
        "ix_student_card_state_question_id", table_name="student_card_state"
    )
    op.drop_index("ix_student_card_state_due_at", table_name="student_card_state")
    op.drop_table("student_card_state")
